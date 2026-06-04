"""
tests/unit/test_audit_pass1_env.py
────────────────────────────────────────────────────────────────────────────
PASS-1 AUDIT — Step 3 (environment correctness) HAND-CALCULATED regression
tests. Every expected number below is computed by hand from the FTMO/PnL rules,
so a silent change to the trade economics breaks a test with a concrete number.

PnL convention (proven in code): pnl = price_move * lots * 100_000, with
price_move signed by position direction. EURUSD pip = 0.0001, so 10 pips = 0.0010
=> 0.10 lots * 0.0010 * 100000 = $10. Commission (forex EURUSD) = $5 round-trip
per standard lot ($2.50 per side).

Covered:
  • BUY pip PnL (exact)            • SELL pip PnL (exact, opposite sign)
  • commission charged on open+close (round-trip $)
  • open→close roundtrip equity    • open→reverse equity
  • daily PASS met / daily FAIL (loss) / zero-trade day = FAIL
  • DD halt then resume next day   • day-rollover baseline snapshot-before-reset
  • target_pct/max_dd_pct never hardcoded (config drives classification)
  • deterministic seeded replay (same seed => identical trajectory)
  • OPTIONAL entry friction (P1): half-spread+slippage worsens the entry when on,
    and is a no-op (exact $50 on 0.5 lots/10 pips) when OFF (the default).
"""
import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.ppo import PPOAgent
from core.agent.action_space import (FLAT, BUY, SELL, EXIT_HOLD, EXIT_CLOSE,
                                      MIN_LOT)

DEV = torch.device("cpu")
BPD = 1440


def _lot_raw_for(lots: float, max_lot: float = 2.0) -> float:
    """Inverse of map_lot: the raw [0,1] that maps to `lots` with no curriculum."""
    return (lots - MIN_LOT) / (max_lot - MIN_LOT)


def _two_level_ohlcv(n, step_at, p0, p1):
    close = np.full(n, p0, dtype=np.float64)
    close[step_at:] = p1
    return np.stack([close, close + 1e-6, close - 1e-6, close,
                     np.ones(n) * 100], axis=1).astype(np.float32)


# Production forex commission table (used only where we want to assert the
# commission itself; the pure pip-PnL tests run commission-free so the expected
# dollar figure is the clean market PnL with no cost term to subtract).
_FOREX_COMM = {"forex": {"kind": "per_lot_round_trip", "value": 5.00}}


def _base_cfg(B=4, days=1, commission=False, **extra):
    cfg = {"BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
           "EPISODE_BARS": BPD * days, "DAILY_TARGET_PCT": 0.025,
           "DAILY_MAX_DD_PCT": 0.010, "MAX_TRADES_PER_DAY": 100000,
           "MAX_LOT": 2.0, "FEATURES": None, "LOT_CURRICULUM_ENABLED": False}
    if commission:
        cfg["COMMISSION"] = _FOREX_COMM
    cfg.update(extra)
    return cfg


_ANY = {"entry_conditions": {"buy": "any", "sell": "any"}}


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: BUY / SELL pip PnL
# ════════════════════════════════════════════════════════════════════════════
def test_buy_pnl_exact_10_pips_half_lot():
    """0.5 lots BUY, +10 pip move => +$50 mark-to-market (friction OFF)."""
    B = 3
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.1010)       # +10 pips
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = env._equity.clone()
    buy = {"direction": torch.full((B,), BUY, dtype=torch.long),
           "lot_raw": torch.full((B,), _lot_raw_for(0.5)),
           "exit": torch.full((B,), EXIT_HOLD, dtype=torch.long)}
    env.step(buy)
    assert torch.allclose(env._position, torch.full((B,), 0.5), atol=1e-3)
    gain = env._equity - eq0
    assert torch.allclose(gain, torch.full((B,), 50.0), atol=1.0), \
        f"BUY 0.5 lots / +10 pips expected +$50, got {gain}"


def test_sell_pnl_exact_10_pips_half_lot():
    """0.5 lots SELL, +10 pip move (price RISES) => -$50 (short loses)."""
    B = 3
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.1010)       # +10 pips up
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = env._equity.clone()
    sell = {"direction": torch.full((B,), SELL, dtype=torch.long),
            "lot_raw": torch.full((B,), _lot_raw_for(0.5)),
            "exit": torch.full((B,), EXIT_HOLD, dtype=torch.long)}
    env.step(sell)
    assert torch.allclose(env._position, torch.full((B,), -0.5), atol=1e-3)
    gain = env._equity - eq0
    assert torch.allclose(gain, torch.full((B,), -50.0), atol=1.0), \
        f"SELL 0.5 lots / +10 pips expected -$50, got {gain}"


def test_sell_profits_when_price_falls():
    """0.5 lots SELL, -10 pip move (price FALLS) => +$50 (short wins)."""
    B = 2
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.0990)       # -10 pips down
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = env._equity.clone()
    sell = {"direction": torch.full((B,), SELL, dtype=torch.long),
            "lot_raw": torch.full((B,), _lot_raw_for(0.5)),
            "exit": torch.full((B,), EXIT_HOLD, dtype=torch.long)}
    env.step(sell)
    gain = env._equity - eq0
    assert torch.allclose(gain, torch.full((B,), 50.0), atol=1.0), \
        f"SELL into a -10-pip fall expected +$50, got {gain}"


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: commission on open + close (round-trip)
# ════════════════════════════════════════════════════════════════════════════
def test_commission_round_trip_on_flat_price():
    """Open then close 1.0 lot on a FLAT price: zero market PnL, so the equity
    delta is exactly the round-trip commission. EURUSD = $5 RT/std-lot."""
    B = 1
    ohlcv = _two_level_ohlcv(400, 999, 1.10, 1.10)         # never moves
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, commission=True), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(100)
    eq0 = env._equity.clone()
    # open 1.0 lot
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(1.0)]),
           "exit": torch.tensor([EXIT_HOLD])}
    env.step(buy)
    after_open = env._equity.clone()
    open_cost = float((eq0 - after_open)[0].item())
    # close the same 1.0 lot (flat price => only close commission)
    close_a = {"direction": torch.tensor([FLAT]), "lot_raw": torch.tensor([0.0]),
               "exit": torch.tensor([EXIT_CLOSE])}
    env.step(close_a)
    rt_cost = float((eq0 - env._equity)[0].item())
    # $2.50 per side, $5.00 round trip for 1.0 lot.
    assert abs(open_cost - 2.5) < 0.25, f"open commission {open_cost} != ~$2.50"
    assert abs(rt_cost - 5.0) < 0.5, f"round-trip commission {rt_cost} != ~$5.00"


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: open → reverse equity (realize then flip)
# ════════════════════════════════════════════════════════════════════════════
def test_open_then_reverse_realizes_pnl_and_flips_side():
    """Open BUY 0.5 lots at 1.10; price → 1.1010 (+10 pips); REVERSE to SELL.
    The +$50 on the long is realized into balance and the new position is short."""
    B = 1
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.1010)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = float(env._equity[0].item())
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(0.5)]),
           "exit": torch.tensor([EXIT_HOLD])}
    env.step(buy)                                          # now at bar 200, +10 pips MTM
    # reverse to SELL on bar 200 (price 1.1010): realizes the +$50 long gain.
    sell = {"direction": torch.tensor([SELL]), "lot_raw": torch.tensor([_lot_raw_for(0.5)]),
            "exit": torch.tensor([EXIT_HOLD])}
    env.step(sell)
    assert float(env._position[0].item()) < 0, "reverse must flip to SHORT"
    # equity reflects the +$50 realized long gain (commission-free cfg here).
    eq_now = float(env._equity[0].item())
    assert eq_now > eq0 + 40.0, f"reverse did not realize the long gain: {eq_now-eq0}"


# ════════════════════════════════════════════════════════════════════════════
# P0 REGRESSION: unrealized MTM must NOT compound into the realized balance
# ════════════════════════════════════════════════════════════════════════════
def test_held_winner_equity_does_not_compound():
    """A winning position HELD at a CONSTANT price must show the SAME marked equity
    every bar — the unrealized PnL is marked once, not re-added each bar. The prior
    code folded mark-to-market back into the balance (self._equity = equity_now with
    equity_now = self._equity + mtm), so a 1-lot long at +30 pips grew +$300 EVERY
    bar (10,300 → 10,600 → 10,900 …). That P0 double-count made holding a winner
    inflate equity without bound and turned PASS trivial. Here we hold 1.0 lot long
    at a flat +30-pip price for 50 bars and assert equity stays pinned at ≈+$300 and
    the realized balance never moves while the position is open."""
    B = 1
    # +30 pips at bar 1, then FLAT forever; DD limit is wide enough not to halt a
    # winning long (this is a gain, DD only triggers on a drawdown from the peak).
    ohlcv = _two_level_ohlcv(400, 1, 1.10, 1.1030)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.zero_()
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(1.0)]),
           "exit": torch.tensor([EXIT_HOLD])}
    hold = {"direction": torch.tensor([FLAT]), "lot_raw": torch.tensor([0.0]),
            "exit": torch.tensor([EXIT_HOLD])}
    eq0 = float(env._equity[0].item())
    env.step(buy)                                          # enter; price now +30 pips
    bal_after_open = float(env._balance[0].item())
    eqs = []
    for _ in range(50):
        env.step(hold)
        eqs.append(float(env._equity[0].item()))
    # 1.0 lot * 30 pips (0.0030) * 100000 = $300 unrealized — marked ONCE.
    for e in eqs:
        assert abs((e - eq0) - 300.0) < 2.0, \
            f"held winner equity {e-eq0:+.1f} != +$300 (compounding double-count?)"
    # the realized balance must NOT move while the position stays open (no realize).
    assert all(abs(float(env._balance[0].item()) - bal_after_open) < 1e-6 for _ in [0]), \
        "balance changed while merely holding an open position"
    # and it certainly must not have grown by +$300 PER bar (the old bug).
    assert (eqs[-1] - eq0) < 350.0, \
        f"equity compounded to {eqs[-1]-eq0:+.1f} — MTM double-counted per bar"


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: daily PASS / FAIL / zero-trade FAIL
# ════════════════════════════════════════════════════════════════════════════
def test_zero_trade_day_is_fail():
    """A day with NO trades must classify FAIL (tier_fail True, passed False) at
    the day boundary — never PASS, never an 'OK/SKIP' free pass."""
    B = 2
    ohlcv = _two_level_ohlcv(BPD + 50, 999, 1.10, 1.10)
    # ungated phase so there is NO force-entry; agent stays flat all day.
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, days=1), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.zero_()
    flat = {"direction": torch.full((B,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(B), "exit": torch.full((B,), EXIT_HOLD,
                                                          dtype=torch.long)}
    info = None
    for _ in range(BPD):
        _s, _r, _d, info = env.step(flat)
    assert bool(info["day_closed"].all().item()), "expected the day to close at BPD"
    assert int(info["trades_today"].sum().item()) == 0, "expected zero trades"
    assert bool(info["tier_fail"].all().item()), "zero-trade day must be tier_fail"
    assert not bool(info["passed"].any().item()), "zero-trade day must not PASS"


def test_losing_day_fails_target_unmet():
    """A day that ends below day_start_equity (a loss) cannot meet the +$ target,
    so it must FAIL and report a negative daily_return."""
    B = 1
    # Drop 5 pips and hold; a single small long loses, ends under target.
    ohlcv = _two_level_ohlcv(BPD + 50, 5, 1.10, 1.0995)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, days=1), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.zero_()
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(1.0)]),
           "exit": torch.tensor([EXIT_HOLD])}
    info = None
    for i in range(BPD):
        act = buy if i == 0 else {"direction": torch.tensor([FLAT]),
                                  "lot_raw": torch.tensor([0.0]),
                                  "exit": torch.tensor([EXIT_HOLD])}
        _s, _r, _d, info = env.step(act)
    assert bool(info["tier_fail"].all().item()), "losing day must FAIL"
    assert float(info["daily_return"][0].item()) < 0, "daily_return should be negative"


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: DD halt then resume next day
# ════════════════════════════════════════════════════════════════════════════
def test_dd_halt_then_resume_next_day():
    """Force a hard DD breach mid-day 0 => halted that day; the next calendar day
    must open with halt cleared (the (~new_day) fix) so trading resumes."""
    B = 4
    ohlcv = _two_level_ohlcv(BPD * 2 + 50, 999, 1.10, 1.10)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, days=2), DEV, phase=_ANY)
    env.reset()
    env._day_high_eq.fill_(env.initial_equity)
    # Drive the breach via the REALIZED balance: marked equity (used for DD) is
    # recomputed each bar as balance + open-position MTM, so balance is the
    # persistent source of truth (a flat episode has zero MTM => equity == balance).
    env._balance.fill_(env.initial_equity * 0.90)          # -10% << 1% limit
    env._equity.fill_(env.initial_equity * 0.90)
    flat = {"direction": torch.full((B,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(B), "exit": torch.full((B,), EXIT_HOLD,
                                                          dtype=torch.long)}
    env.step(flat)
    assert bool(env._day_halted.all().item()), "expected halt after DD breach"
    while int(env._step_i[0].item()) % BPD != 0:
        env.step(flat)
    assert not bool(env._day_halted.any().item()), "next day must clear the halt"


# ════════════════════════════════════════════════════════════════════════════
# HAND-CALC: day-rollover baseline snapshot BEFORE reset
# ════════════════════════════════════════════════════════════════════════════
def test_day_rollover_reports_closing_day_baseline():
    """On the calendar boundary, info['day_start_eq'] must be the CLOSING day's
    opening equity (the pre-reset snapshot), not the post-reset baseline — else
    daily_return would read ~0 on every boundary and PASS could never fire."""
    B = 1
    ohlcv = _two_level_ohlcv(BPD + 50, 1, 1.10, 1.1010)    # +10 pips immediately
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, days=1), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.zero_()
    start_eq = float(env._equity[0].item())
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(1.0)]),
           "exit": torch.tensor([EXIT_HOLD])}
    info = None
    for i in range(BPD):
        act = buy if i == 0 else {"direction": torch.tensor([FLAT]),
                                  "lot_raw": torch.tensor([0.0]),
                                  "exit": torch.tensor([EXIT_HOLD])}
        _s, _r, _d, info = env.step(act)
    assert abs(float(info["day_start_eq"][0].item()) - start_eq) < 1e-3, \
        "closing-day baseline must be the pre-reset day-start equity"


# ════════════════════════════════════════════════════════════════════════════
# CONFIG-DRIVEN classification (nothing hardcoded to 2.5% / 1%)
# ════════════════════════════════════════════════════════════════════════════
def test_target_pct_is_config_driven_not_hardcoded():
    """Changing DAILY_TARGET_PCT changes the fixed-$ daily target proportionally
    (initial_equity * target_pct), proving it is not hardcoded to 0.025."""
    env1 = BatchedFTMOEnv(_two_level_ohlcv(400, 999, 1.1, 1.1),
                          _base_cfg(DAILY_TARGET_PCT=0.025), DEV, phase=_ANY)
    env2 = BatchedFTMOEnv(_two_level_ohlcv(400, 999, 1.1, 1.1),
                          _base_cfg(DAILY_TARGET_PCT=0.05), DEV, phase=_ANY)
    assert abs(env1.daily_increment - env1.initial_equity * 0.025) < 1e-6
    assert abs(env2.daily_increment - env2.initial_equity * 0.05) < 1e-6
    assert env2.daily_increment > env1.daily_increment


# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC seeded replay
# ════════════════════════════════════════════════════════════════════════════
def test_deterministic_seeded_replay():
    """Same torch seed + same env config => identical equity trajectory. A silent
    nondeterminism (unseeded sampling, dict ordering) would diverge the run."""
    def run():
        torch.manual_seed(20240601)
        np.random.seed(20240601)
        ohlcv = _two_level_ohlcv(BPD + 50, 200, 1.10, 1.1005)
        env = BatchedFTMOEnv(ohlcv, _base_cfg(B=4, days=1), DEV, phase=_ANY)
        agent = PPOAgent(env.state_dim, env.cfg, DEV)
        state = env.reset()
        eqs = []
        for _ in range(300):
            out = agent.select_actions(state, mask=env.current_direction_mask())
            state, _r, _d, info = env.step(out)
            eqs.append(float(info["equity"].sum().item()))
        return eqs
    a, b = run(), run()
    assert a == b, "seeded replay diverged — nondeterminism in the core loop"


# ════════════════════════════════════════════════════════════════════════════
# OPTIONAL entry friction (P1): no-op when OFF, pessimistic when ON
# ════════════════════════════════════════════════════════════════════════════
def test_entry_friction_off_is_frictionless_default():
    """DEFAULT (ENTRY_FRICTION_ENABLED unset): entry fills exactly at close, so
    0.5 lots / +10 pips is exactly +$50 — the historical, frictionless economics."""
    B = 1
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.1010)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = env._equity.clone()
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(0.5)]),
           "exit": torch.tensor([EXIT_HOLD])}
    env.step(buy)
    gain = float((env._equity - eq0)[0].item())
    assert abs(gain - 50.0) < 1.0, f"friction-off must be exactly +$50, got {gain}"


def test_entry_friction_on_worsens_buy_entry():
    """With ENTRY_FRICTION_ENABLED, a BUY entry price is worse (higher) by
    half-spread + slippage, so the SAME +10-pip move nets LESS than the
    frictionless +$50. EURUSD default friction = 0.0001*(0.5*1.0+0.5)=0.0001
    => 1 pip worse entry => 0.5 lots * 1 pip = $5 less, so ≈ +$45."""
    B = 1
    ohlcv = _two_level_ohlcv(400, 200, 1.10, 1.1010)
    env = BatchedFTMOEnv(ohlcv, _base_cfg(B=B, ENTRY_FRICTION_ENABLED=True),
                         DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.fill_(199)
    eq0 = env._equity.clone()
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([_lot_raw_for(0.5)]),
           "exit": torch.tensor([EXIT_HOLD])}
    env.step(buy)
    gain = float((env._equity - eq0)[0].item())
    assert 43.0 < gain < 48.0, \
        f"friction-on BUY should net ≈+$45 (1-pip worse entry), got {gain}"
    assert gain < 50.0, "entry friction must make the entry strictly worse"
