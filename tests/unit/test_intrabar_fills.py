"""Unit tests for intrabar fill approximation."""
from core.env.intrabar_fills import compute_fill
from core.agent.action_space import BUY, SELL, HOLD
from tests.fixtures.sample_configs import minimal_trading_policy

POLICY = minimal_trading_policy()
BAR = {"open": 1.1750, "high": 1.1760, "low": 1.1740, "close": 1.1755}


def test_buy_fill_above_open():
    f = compute_fill(BAR, BUY, sl_pips=10, tp_pips=20, instrument="EURUSD", policy=POLICY)
    assert f["entry"] > BAR["open"]
    assert f["sl"] < f["entry"]      # SL below entry for BUY
    assert f["tp"] > f["entry"]      # TP above entry for BUY


def test_sell_fill_below_open():
    f = compute_fill(BAR, SELL, sl_pips=10, tp_pips=20, instrument="EURUSD", policy=POLICY)
    assert f["entry"] < BAR["open"]
    assert f["sl"] > f["entry"]      # SL above entry for SELL
    assert f["tp"] < f["entry"]      # TP below entry for SELL


def test_hold_returns_open():
    f = compute_fill(BAR, HOLD, sl_pips=10, tp_pips=20, instrument="EURUSD", policy=POLICY)
    assert f["entry"] == BAR["open"] and f["sl"] is None


def test_atr_caps_sl_width():
    # tiny ATR -> SL cannot be wider than 3*ATR from entry
    f = compute_fill(BAR, BUY, sl_pips=500, tp_pips=20, instrument="EURUSD",
                     policy=POLICY, atr_14=0.0005)
    assert f["entry"] - f["sl"] <= 3 * 0.0005 + 1e-9
