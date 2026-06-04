"""
tests/integration/test_commission_in_equity.py
────────────────────────────────────────────────────────────────────────────
PASS-2 ADDENDUM regression: commission must be reflected in the SAME equity that
drives BOTH the 1% trailing-DD check AND the 2.5% daily-target / 5-tier
classification. There must be no path where commission hits a balance/PnL that
is not the equity the DD/target read.

Hand-calc anchor (forex EURUSD, CFG COMMISSION = $5.00/std-lot ROUND TRIP):
    2.0 lots round trip  ->  open side $5.00  +  close side $5.00  =  $10.00.

We drive a controlled scenario on a FLAT price series (every close identical, so
mark-to-market PnL is exactly $0 and the only thing that can move equity is
commission). We open 2.0 lots on bar 0 and close on bar 1, then assert:

  (a) realized balance dropped by exactly $10.00 (open $5 + close $5);
  (b) the equity read by the DD trailing check is the post-commission equity
      (day_high - equity reflects the $10 give-back, NOT $0);
  (c) the daily tier classification's `progress` uses the post-commission equity
      (a day whose ONLY activity is paying $10 commission ends BELOW target ->
      its progress is negative, i.e. it is a FAIL, not an OK/PASS).

This is a true regression guard: if commission were ever deducted from a side
ledger that equity_now does not source from, the equity here would stay at the
initial 100_000 and (a)/(b)/(c) would all fail.
"""
from __future__ import annotations

import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import BUY, FLAT, EXIT_HOLD, EXIT_CLOSE

DEV = torch.device("cpu")
B = 4
PX = 1.10000           # constant price -> zero mark-to-market PnL


def _flat_series(n: int = 400) -> np.ndarray:
    """Perfectly flat OHLCV at PX so the ONLY equity mover is commission."""
    row = [PX, PX, PX, PX, 100.0]
    return np.asarray([row] * n, dtype=np.float32)


def _cfg() -> dict:
    return {
        "BATCH_SIZE_ENV": B, "LOOKBACK": 20,
        "BARS_PER_DAY": 60, "EPISODE_BARS": 180,
        "INITIAL_EQUITY": 100_000.0, "MAX_LOT": 2.0,
        "DAILY_TARGET_PCT": 0.025, "DAILY_MAX_DD_PCT": 0.01,
        "MAX_TRADES_PER_DAY": 800,
        "USE_AMP": False, "USE_TORCH_COMPILE": False,
        # forex $5/std-lot round trip is the active path (mirrors settings.py)
        "COMMISSION": {"forex": {"kind": "per_lot_round_trip", "value": 5.00}},
        "CONTRACT_SIZE": 100_000.0,
    }


# free phase (no gate / no force-entry) so we fully control entries/exits
_PHASE = {"name": "free", "mask": None, "mask_type": "none"}


def _open_2lots_then_close():
    """Return (env, eq_after_open, eq_after_close) for a 2.0-lot round trip on a
    flat series, with the curriculum window pinned to exactly [2.0, 2.0] so
    lot_raw=1.0 maps to precisely 2.0 lots."""
    env = BatchedFTMOEnv(_flat_series(), _cfg(), DEV, phase=_PHASE)
    env.reset()
    # Pin the curriculum window to exactly 2.0 lots (lot_raw -> 2.0 exactly).
    env._lot_lo, env._lot_hi = 2.0, 2.0

    ones = torch.ones(env.B)
    # Bar 0: OPEN 2.0 lots long (pays the OPEN commission = $5.00).
    env.step({"direction": (BUY * ones).long(),
              "lot_raw": ones.float(),
              "exit": (EXIT_HOLD * ones).long()})
    eq_after_open = float(env._equity[0].item())

    # Bar 1: CLOSE the position (pays the CLOSE commission = $5.00).
    env.step({"direction": (FLAT * ones).long(),
              "lot_raw": torch.zeros(env.B).float(),
              "exit": (EXIT_CLOSE * ones).long()})
    eq_after_close = float(env._equity[0].item())
    return env, eq_after_open, eq_after_close


def test_open_commission_hits_equity_exactly_five_dollars():
    """OPEN side of a 2.0-lot round trip costs exactly $5.00, felt in equity."""
    _, eq_after_open, _ = _open_2lots_then_close()
    # flat price => zero MTM => equity drop is PURE open commission.
    assert abs((100_000.0 - eq_after_open) - 5.00) < 1e-6, (
        f"open commission not reflected in equity: drop="
        f"{100_000.0 - eq_after_open:.6f}, expected 5.00")


def test_round_trip_commission_is_ten_dollars_in_equity():
    """Full 2.0-lot round trip costs exactly $10.00 ($5 open + $5 close),
    and that $10 is visible in the equity (balance) the env exposes."""
    env, _, eq_after_close = _open_2lots_then_close()
    assert abs((100_000.0 - eq_after_close) - 10.00) < 1e-6, (
        f"round-trip commission wrong: drop={100_000.0 - eq_after_close:.6f}, "
        f"expected 10.00")
    # position is flat, so balance == equity exactly (no open MTM component).
    assert abs(float(env._balance[0].item()) - eq_after_close) < 1e-6


def test_commission_drop_feeds_trailing_dd_check():
    """The 1% trailing-DD check reads the POST-commission equity: with the
    high-water mark set at the initial 100_000 and equity now 99_990 after the
    $10 round trip, the trailing draw-down used must equal $10 / day_high, NOT
    zero. If commission hit a side ledger, dd_used would be ~0 here."""
    env, _, eq_after_close = _open_2lots_then_close()
    day_high = float(env._day_high_eq[0].item())
    dd_used = (day_high - eq_after_close) / (day_high + 1e-8)
    expected_dd = 10.00 / (day_high + 1e-8)
    assert abs(dd_used - expected_dd) < 1e-9, (
        f"trailing DD did not see commission: dd_used={dd_used:.3e}, "
        f"expected={expected_dd:.3e}")
    assert dd_used > 0.0, "commission produced ZERO trailing draw-down (leak!)"


def test_commission_drop_feeds_daily_tier_progress():
    """The 5-tier daily classification's `progress` = (equity - day_start) /
    daily_increment uses the POST-commission equity. A day whose only activity
    is paying $10 commission ends $10 BELOW its opening equity, so progress is
    negative -> the day is a FAIL (progress < 0.5), never OK/PASS."""
    env, _, eq_after_close = _open_2lots_then_close()
    day_start = float(env._day_start_eq[0].item())
    inc = float(env._daily_increment_t[0].item())   # 100_000 * 0.025 = 2_500
    progress = (eq_after_close - day_start) / max(inc, 1e-9)
    # equity is 99_990, day_start 100_000 -> progress = -10 / 2500 < 0.
    assert progress < 0.0, (
        f"tier progress ignored commission: progress={progress:.6f} "
        f"(equity={eq_after_close}, day_start={day_start}, inc={inc})")
    assert progress < 0.5, "commission-only day must classify as FAIL, not OK/PASS"


def test_bigger_size_moves_dd_and_target_equity_more():
    """A sized-UP position incurs LARGER commission and therefore moves the
    DD/target equity MORE: 2.0 lots ($10 RT) must drop equity by exactly 4x the
    drop of 0.5 lots ($2.50 RT). Confirms the deduction scales with lots on the
    SAME equity the DD/target read."""
    def round_trip_drop(lots: float) -> float:
        env = BatchedFTMOEnv(_flat_series(), _cfg(), DEV, phase=_PHASE)
        env.reset()
        env._lot_lo, env._lot_hi = lots, lots
        ones = torch.ones(env.B)
        env.step({"direction": (BUY * ones).long(), "lot_raw": ones.float(),
                  "exit": (EXIT_HOLD * ones).long()})
        env.step({"direction": (FLAT * ones).long(),
                  "lot_raw": torch.zeros(env.B).float(),
                  "exit": (EXIT_CLOSE * ones).long()})
        return 100_000.0 - float(env._equity[0].item())

    drop_2lot = round_trip_drop(2.0)     # $10.00
    drop_half = round_trip_drop(0.5)     # $2.50
    assert abs(drop_2lot - 10.00) < 1e-6
    assert abs(drop_half - 2.50) < 1e-6
    assert abs(drop_2lot - 4.0 * drop_half) < 1e-6   # scales linearly with lots
