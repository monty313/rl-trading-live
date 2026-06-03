"""
tests/unit/test_day_boundary_and_lot_fix.py
────────────────────────────────────────────────────────────────────────────
Regression tests for the two bugs fixed in this change:

ISSUE 1 — the perfect every-other-day zero-trade pattern. Root cause: a DD
breach (`breach_now`) is measured against the CLOSING day's peak at the top of
step(), but it was OR'd back into `_day_halted` AFTER the new-day reset had just
cleared the halt — so the brand-new day opened permanently halted, traded ZERO
bars, cleared at its own boundary, and the cycle repeated (odd/even alternation).
The fix gates the halt with (~new_day): a calendar rollover starts a FRESH day
with halt=False, while intraday breaches still halt as before. These tests
assert:
  • bars-per-day is honored (one day counter increment per BARS_PER_DAY bars),
  • after a DD-halt day the NEXT day starts with halt=False for ALL episodes,
  • across >=4 consecutive gate-on days there is NO alternating odd/even
    zero-trade signature and NO zero-trade day on an episode that had gate-on
    bars (the day-boundary invariant).

ISSUE 2 — the lot head being stuck near the floor. The audit showed the
mechanics are correct (mapping, env sizing, PnL formula, proportional scaler);
the fix makes the "initial mean lot is mid-range" invariant hold BY CONSTRUCTION
(zeroed lot-head bias) and keeps the full [MIN_LOT, max_lot] range reachable and
size exploration alive. These tests assert:
  • initial mean lot is mid-range (≈1.0 at MAX_LOT=2.0), not pinned at floor,
  • the lot head can output the full [MIN_LOT, MAX_LOT] range,
  • a chosen lot flows into env position sizing and yields proportional PnL
    (0.5 lots on a 10-pip move ≈ $50),
  • the proportional scaler == 1.0 EXACTLY at the trained baseline.
"""
import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv, proportional_lot_scale
from core.agent.ppo import PPOAgent, ActorCritic
from core.agent.action_space import (DIRECTION_DIM, FLAT, BUY, SELL, EXIT_HOLD,
                                     MIN_LOT, map_lot)

DEV = torch.device("cpu")
BPD = 1440


# ── synthetic data that makes the phase1 CCI gate fire on many bars/day ──────
def _gated_ohlcv(days: int) -> np.ndarray:
    n = BPD * days + 200
    t = np.arange(n)
    close = 1.10 + 0.01 * np.sin(t / 50.0) + 0.002 * np.sin(t / 7.0)
    high = close + 5e-4
    low = close - 5e-4
    vol = np.ones(n) * 100.0
    return np.stack([close, high, low, close, vol], axis=1).astype(np.float32)


def _gated_cfg(B: int, days: int) -> dict:
    return {
        "BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
        "EPISODE_BARS": BPD * days, "DAILY_TARGET_PCT": 0.025,
        "DAILY_MAX_DD_PCT": 0.010, "MAX_TRADES_PER_DAY": 100000,
        "MAX_LOT": 2.0, "FEATURES": None,
    }


_PHASE1 = {"name": "phase1_cci_align", "mask": "phase1_cci_align",
           "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15]}


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 1 — day-boundary / halt-reset regressions
# ════════════════════════════════════════════════════════════════════════════
def test_bars_per_day_is_1440_and_day_counter_increments_once_per_day():
    """The env's day counter advances EXACTLY once per BARS_PER_DAY (=1440) bars,
    not every 720 (off-by-half) and not twice per day (double-rollover)."""
    days = 4
    env = BatchedFTMOEnv(_gated_ohlcv(days), _gated_cfg(4, days), DEV, phase=_PHASE1)
    assert env.bars_per_day == 1440
    agent = PPOAgent(env.state_dim, env.cfg, DEV)
    state = env.reset()
    last_day_idx = 0
    increments_at = []
    for step in range(BPD * days):
        out = agent.select_actions(state, mask=env.current_direction_mask())
        state, _r, _d, info = env.step(out)
        di = int(info["day_idx"][0].item())
        if di != last_day_idx:
            increments_at.append(step + 1)   # 1-based bar number where it ticked
            last_day_idx = di
    # one increment per day, each landing precisely on a 1440-bar boundary
    assert increments_at == [BPD * k for k in range(1, days + 1)], increments_at


def test_halt_flag_resets_next_day_for_all_episodes():
    """After a forced DD-halt on day 0, the NEXT calendar day must open with
    _day_halted == False for EVERY episode (force-entry works again next day).
    This is the exact invariant the every-other-day bug violated."""
    B, days = 6, 3
    env = BatchedFTMOEnv(_gated_ohlcv(days), _gated_cfg(B, days), DEV, phase=_PHASE1)
    env.reset()
    # Force a hard DD breach on EVERY episode partway through day 0 by slamming
    # equity well below the day's peak, then step one bar so the breach is detected.
    env._day_high_eq.fill_(env.initial_equity)
    env._equity.fill_(env.initial_equity * 0.90)        # -10% << 1% DD limit
    flat = {"direction": torch.full((B,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(B), "exit": torch.full((B,), EXIT_HOLD,
                                                          dtype=torch.long)}
    s, r, d, info = env.step(flat)
    assert bool(env._day_halted.all().item()), "expected all episodes halted mid-day"

    # Step to the end of day 0 (the calendar boundary) — halt stays set within day.
    while int(env._step_i[0].item()) % BPD != 0:
        env.step(flat)
    # The boundary step itself cleared halt for the fresh day; assert False for ALL.
    assert not bool(env._day_halted.any().item()), \
        "next day must start with halt=False for all episodes"
    # And it must STAY clear at the very first bar of the new day.
    env.step(flat)
    assert not bool(env._day_halted.any().item())


def test_no_alternating_odd_even_zero_trade_pattern():
    """Across >=4 consecutive gate-on days, assert NO day produces zero trades on
    an episode that had gate-on bars, and assert there is NO alternating odd/even
    zero-trade signature (the impossible pattern the bug produced)."""
    B, days = 8, 6
    env = BatchedFTMOEnv(_gated_ohlcv(days), _gated_cfg(B, days), DEV, phase=_PHASE1)
    agent = PPOAgent(env.state_dim, env.cfg, DEV)
    state = env.reset()

    gate_count = np.zeros((days, B), dtype=int)
    trades = np.zeros((days, B), dtype=int)
    for d in range(days):
        for _ in range(BPD):
            out = agent.select_actions(state, mask=env.current_direction_mask())
            gate_count[d] += env._gate_on_batch().cpu().numpy().astype(int)
            state, _r, _dn, info = env.step(out)
        trades[d] = info["trades_today"].cpu().numpy()

    # (1) Any (day, episode) cell with >=1 gate-on bar MUST have >=1 trade
    #     (force_in_and_gate guarantees an entry on every gate-on bar when flat).
    gated_cells = gate_count > 0
    assert (trades[gated_cells] >= 1).all(), (
        "a gate-on episode-day produced zero trades:\n"
        f"gate_on=\n{gate_count}\ntrades=\n{trades}")

    # (2) No alternating odd/even zero-trade signature: it is NOT the case that
    #     all even-index days are zero while all odd-index days are nonzero
    #     (or vice-versa) — that perfect parity pattern was the bug.
    day_has_trade = trades.sum(axis=1) > 0          # (days,) per-day any-trade
    day_has_gate = gate_count.sum(axis=1) > 0
    even_idx = [d for d in range(days) if d % 2 == 0 and day_has_gate[d]]
    odd_idx = [d for d in range(days) if d % 2 == 1 and day_has_gate[d]]
    even_all_zero = len(even_idx) > 0 and all(not day_has_trade[d] for d in even_idx)
    odd_all_zero = len(odd_idx) > 0 and all(not day_has_trade[d] for d in odd_idx)
    assert not even_all_zero, f"even gate-on days all zero-trade: {trades.sum(1)}"
    assert not odd_all_zero, f"odd gate-on days all zero-trade: {trades.sum(1)}"

    # (3) Every gate-on DAY (aggregated over episodes) trades.
    for d in range(days):
        if day_has_gate[d]:
            assert day_has_trade[d], f"day {d} had gate-on bars but zero trades"


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 2 — lot-path regressions
# ════════════════════════════════════════════════════════════════════════════
def _lot_cfg():
    return {"BATCH_SIZE_ENV": 32, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
            "DAILY_TARGET_PCT": 0.025, "DAILY_MAX_DD_PCT": 0.010, "MAX_LOT": 2.0,
            "FEATURES": None}


def test_initial_mean_lot_is_mid_range_not_floor():
    """The lot head's INITIAL mean lot must sit ~mid-range (≈1.0 at MAX_LOT=2.0),
    not pinned at the 0.01 floor. Guaranteed by the zeroed lot-head bias."""
    for seed in (0, 1, 7, 123):
        torch.manual_seed(seed)
        net = ActorCritic(state_dim=128)
        x = torch.randn(64, 128)
        _dl, _el, lot_mean, _v = net(x)
        lot_raw = torch.sigmoid(lot_mean.squeeze(-1))
        mapped = MIN_LOT + lot_raw * (2.0 - MIN_LOT)
        m = float(mapped.mean().item())
        assert 0.7 < m < 1.3, f"initial mean lot {m:.3f} not mid-range (seed {seed})"
        assert m > 0.1, "initial lot must NOT be pinned at the floor"


def test_lot_head_reaches_full_range():
    """sigmoid(lot_pre) -> map_lot must reach BOTH ends of [MIN_LOT, MAX_LOT]."""
    lo = map_lot(float(torch.sigmoid(torch.tensor(-8.0))), 2.0)
    hi = map_lot(float(torch.sigmoid(torch.tensor(8.0))), 2.0)
    mid = map_lot(float(torch.sigmoid(torch.tensor(0.0))), 2.0)
    assert lo == MIN_LOT, f"low end {lo} != {MIN_LOT}"
    assert hi == 2.0, f"high end {hi} != 2.0"
    assert abs(mid - 1.0) < 0.02, f"mid {mid} not ~1.0"


def test_chosen_lot_flows_to_env_sizing_and_proportional_pnl():
    """A chosen lot must drive env position size and produce proportional PnL:
    0.5 lots on a clean 10-pip move ≈ $50 (pnl = price_move * lots * 100000)."""
    B = 4
    n = 400
    # Flat price, then a single clean +10-pip step so PnL is exactly attributable.
    close = np.full(n, 1.10, dtype=np.float64)
    close[200:] = 1.1010                                   # +10 pips (0.0010)
    ohlcv = np.stack([close, close + 1e-4, close - 1e-4, close,
                      np.ones(n) * 100], axis=1).astype(np.float32)
    cfg = {"BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
           "DAILY_TARGET_PCT": 0.025, "DAILY_MAX_DD_PCT": 0.010, "MAX_LOT": 2.0,
           "MAX_TRADES_PER_DAY": 100000, "FEATURES": None,
           # Isolate the proportional-PnL invariant from the Section-8 lot
           # curriculum clamp: the full [MIN_LOT, MAX_LOT] head must map 0.5->0.5.
           "LOT_CURRICULUM_ENABLED": False}
    env = BatchedFTMOEnv(ohlcv, cfg, DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    env.reset()
    # Pin the absolute bar so the price step is deterministic: abs_idx = start +
    # step_i must land on bar 199 (last 1.10 bar; bar 200 is the +10-pip bar).
    env._start.fill_(0)
    env._step_i.fill_(199)                                 # current bar = index 199
    eq0 = env._equity.clone()
    # lot_raw that maps to 0.5 lots: 0.5 = 0.01 + raw*(2.0-0.01)
    lot_raw = (0.5 - MIN_LOT) / (2.0 - MIN_LOT)
    buy = {"direction": torch.full((B,), BUY, dtype=torch.long),
           "lot_raw": torch.full((B,), float(lot_raw)),
           "exit": torch.full((B,), EXIT_HOLD, dtype=torch.long)}
    env.step(buy)                                          # open at 1.10 (bar 199)
    # position must be ~0.5 lots long
    assert torch.allclose(env._position, torch.full((B,), 0.5), atol=1e-3), \
        f"position {env._position} != 0.5 lots"
    # next bar is 1.1010 -> +10 pips -> mark-to-market gain ≈ $50/episode
    gain = (env._equity - eq0)
    assert torch.allclose(gain, torch.full((B,), 50.0), atol=1.0), \
        f"PnL {gain} not ≈ $50 for 0.5 lots on 10-pip move"


def test_proportional_scaler_is_one_at_baseline():
    """At the trained baseline (2.5% / 1%) the proportional lot scaler is EXACTLY
    1.0 — it must not silently shrink (or grow) lots in the default regime."""
    assert proportional_lot_scale(0.025, 0.010, 0.025, 0.010) == 1.0
    # agent wrapper agrees and respects the on/off toggle
    agent = PPOAgent(state_dim=128, cfg=_lot_cfg(), device=DEV)
    assert agent.proportional_scale(0.025, 0.010) == 1.0
    # tighter DD scales DOWN (<1); higher target scales UP (>1) — sanity of sign.
    assert proportional_lot_scale(0.025, 0.005, 0.025, 0.010) < 1.0
    assert proportional_lot_scale(0.050, 0.010, 0.025, 0.010) > 1.0


def test_lot_head_does_not_collapse_to_floor_after_short_training():
    """After a few PPO updates on a gated phase, the deterministic mean lot must
    NOT have collapsed toward the 0.01 floor (size stays learnable/usable)."""
    B, days = 16, 4
    env = BatchedFTMOEnv(_gated_ohlcv(days),
                         {**_gated_cfg(B, days), "ROLLOUT_STEPS": 256}, DEV,
                         phase=_PHASE1)
    agent = PPOAgent(env.state_dim, env.cfg, DEV)
    state = env.reset()
    for _ in range(1500):
        out = agent.select_actions(state, mask=env.current_direction_mask())
        s, r, d, info = env.step(out)
        agent.store(state, out, r, d, env.current_direction_mask())
        state = s
        if len(agent.buffer) >= 256:
            agent.update()
    _dl, _el, lot_mean, _v = agent.net(state)
    mean_lot = (MIN_LOT + torch.sigmoid(lot_mean.squeeze(-1))
                * (2.0 - MIN_LOT)).mean().item()
    assert mean_lot > 0.2, f"lot head collapsed toward floor (mean lot {mean_lot:.3f})"
