"""
tests/unit/test_audit_pass1_reward.py
────────────────────────────────────────────────────────────────────────────
PASS-1 AUDIT — Step 4 (reward) ADVERSARIAL tests. The reward is the steering
wheel: if a losing/flat/giveback day can out-score a profitable/PASS day, the
agent learns the wrong thing. These run a FULL trading day on scripted prices,
sum the per-step reward, and assert the ORDERING the FTMO objective demands:

  • a profitable day scores strictly higher than an identical-structure LOSING day
  • a PASS day (>= target) scores strictly higher than a FAIL day
  • a flat / no-trade day FAILS and scores below a profitable day
  • a give-back day (ran up, gave it all back to a loss) scores below a day that
    simply stayed flat-positive — surrendering gains is penalized
  • reward stays FINITE (no NaN/Inf) under an extreme price spike

Comparisons are RELATIVE (ordering), which is exactly what the optimizer cares
about, and robust to the exact magnitude of the 14-section shaping.
"""
import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import FLAT, BUY, EXIT_HOLD, MIN_LOT

DEV = torch.device("cpu")
BPD = 1440


def _cfg(B=1, **extra):
    cfg = {"BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
           "EPISODE_BARS": BPD, "DAILY_TARGET_PCT": 0.025,
           "DAILY_MAX_DD_PCT": 0.010, "MAX_TRADES_PER_DAY": 100000,
           "MAX_LOT": 2.0, "FEATURES": None, "LOT_CURRICULUM_ENABLED": False}
    cfg.update(extra)
    return cfg


_ANY = {"entry_conditions": {"buy": "any", "sell": "any"}}


def _lot_raw_for(lots, max_lot=2.0):
    return (lots - MIN_LOT) / (max_lot - MIN_LOT)


def _run_day(close_path, lots=1.0, side=BUY, enter_bar=0):
    """Run ONE full BPD-day on the given 1m close path (len >= BPD+2). Open a
    `lots`-lot position in `side` at `enter_bar` and HOLD; return the summed
    per-step reward and the closing-day info dict."""
    n = len(close_path)
    close = np.asarray(close_path, dtype=np.float64)
    ohlcv = np.stack([close, close + 1e-6, close - 1e-6, close,
                      np.ones(n) * 100], axis=1).astype(np.float32)
    env = BatchedFTMOEnv(ohlcv, _cfg(B=1), DEV, phase=_ANY)
    env.reset(); env._start.fill_(0); env._step_i.zero_()
    total = 0.0
    info = None
    for i in range(BPD):
        if i == enter_bar:
            act = {"direction": torch.tensor([side]),
                   "lot_raw": torch.tensor([_lot_raw_for(lots)]),
                   "exit": torch.tensor([EXIT_HOLD])}
        else:
            act = {"direction": torch.tensor([FLAT]),
                   "lot_raw": torch.tensor([0.0]),
                   "exit": torch.tensor([EXIT_HOLD])}
        _s, r, _d, info = env.step(act)
        total += float(r[0].item())
    return total, info


def _flat_then(level, change_at, new_level, n=BPD + 5):
    p = np.full(n, level, dtype=np.float64)
    p[change_at:] = new_level
    return p


# ════════════════════════════════════════════════════════════════════════════
# profitable day reward > identical losing day
# ════════════════════════════════════════════════════════════════════════════
def test_profitable_day_outscores_losing_day():
    """Same structure (1 lot long, enter at bar 0), only the price direction
    differs: +30 pips (profit) vs -30 pips (loss). The profitable day MUST score
    strictly higher."""
    up = _flat_then(1.10, 1, 1.1030)      # +30 pips
    down = _flat_then(1.10, 1, 1.0970)    # -30 pips
    r_up, info_up = _run_day(up, lots=1.0, side=BUY)
    r_down, info_down = _run_day(down, lots=1.0, side=BUY)
    assert r_up > r_down, f"profitable day {r_up} !> losing day {r_down}"
    assert float(info_up["daily_return"][0].item()) > 0
    assert float(info_down["daily_return"][0].item()) < 0


# ════════════════════════════════════════════════════════════════════════════
# PASS day reward > FAIL day
# ════════════════════════════════════════════════════════════════════════════
def test_pass_day_outscores_fail_day():
    """A day that clears the +2.5% target (PASS/EXCEED) must score strictly above
    a day that ends slightly down (FAIL). On $10k @ 2.5% the target is +$250; a
    1-lot long that runs +30 pips ≈ +$300 PASSES; +0... a small loss FAILS."""
    pass_path = _flat_then(1.10, 1, 1.1030)     # +30 pips ≈ +$300 > $250 target
    fail_path = _flat_then(1.10, 1, 1.0995)     # -5 pips ≈ -$50 (FAIL)
    r_pass, info_pass = _run_day(pass_path, lots=1.0, side=BUY)
    r_fail, info_fail = _run_day(fail_path, lots=1.0, side=BUY)
    assert bool(info_pass["passed"][0].item()), "expected a PASS day"
    assert bool(info_fail["tier_fail"][0].item()), "expected a FAIL day"
    assert r_pass > r_fail, f"PASS reward {r_pass} !> FAIL reward {r_fail}"


# ════════════════════════════════════════════════════════════════════════════
# flat / no-trade day = FAIL and scores below a profitable day
# ════════════════════════════════════════════════════════════════════════════
def test_flat_no_trade_day_fails_and_underperforms_profit():
    """A no-trade day classifies FAIL and must score below a profitable day. We
    run a flat price with NO entry (enter_bar past the day) vs a profitable day."""
    flat_path = np.full(BPD + 5, 1.10, dtype=np.float64)
    r_flat, info_flat = _run_day(flat_path, lots=1.0, side=BUY, enter_bar=BPD + 1)
    assert int(info_flat["trades_today"][0].item()) == 0, "expected zero trades"
    assert bool(info_flat["tier_fail"][0].item()), "no-trade day must FAIL"
    up = _flat_then(1.10, 1, 1.1030)
    r_up, _ = _run_day(up, lots=1.0, side=BUY)
    assert r_up > r_flat, f"profitable {r_up} !> flat/no-trade {r_flat}"


# ════════════════════════════════════════════════════════════════════════════
# give-back day penalized vs staying flat-positive
# ════════════════════════════════════════════════════════════════════════════
def test_giveback_day_penalized_vs_held_gain():
    """Two days that both peak at +30 pips intraday. Day A HOLDS the gain to the
    close (ends +30). Day B GIVES IT BACK and ends slightly negative. The giveback
    day must score strictly LOWER — surrendering an intraday run is penalized
    (Section 3.2 wipeout + 3.3 cross-day giveback)."""
    held = _flat_then(1.10, 1, 1.1030)                 # up +30 and stay
    give = np.full(BPD + 5, 1.10, dtype=np.float64)
    give[1:200] = 1.1030                               # run to +30 pips early
    give[200:] = 1.0995                                # then give it back to -5
    r_held, info_held = _run_day(held, lots=1.0, side=BUY)
    r_give, info_give = _run_day(give, lots=1.0, side=BUY)
    assert float(info_held["daily_return"][0].item()) > 0
    assert float(info_give["daily_return"][0].item()) < 0
    assert r_held > r_give, f"held-gain {r_held} !> giveback {r_give}"


# ════════════════════════════════════════════════════════════════════════════
# reward finite under extreme prices
# ════════════════════════════════════════════════════════════════════════════
def test_reward_finite_under_extreme_spike():
    """An extreme intrabar price spike must not produce NaN/Inf reward. We slam the
    price to a huge value mid-day while long and assert every step reward is
    finite."""
    spike = np.full(BPD + 5, 1.10, dtype=np.float64)
    spike[700:702] = 5.00          # absurd +390% spike then back
    spike[702:] = 1.10
    total, info = _run_day(spike, lots=1.0, side=BUY)
    assert np.isfinite(total), f"summed reward not finite: {total}"
    assert torch.isfinite(info["equity"]).all(), "equity went non-finite"
