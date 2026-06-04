"""
tests/integration/test_live_runner_mt5_s9.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 9 — MOCKED MT5 live-runner resilience. EVERYTHING is mocked; no real
account, no network, no order ever leaves the process. Covers the full failure
surface a real MT5 link exhibits:

  • connection-failure retry/backoff (initialize fails N times then succeeds);
  • symbol-unavailable -> initialize returns False (never trades a wrong symbol);
  • order request FORMAT (symbol/volume/type/price/sl/tp present and typed);
  • DUPLICATE-order prevention (same bar+decision fires at most once);
  • reconnect during an open trade (link drops on send -> reconnect + resend);
  • REJECTION records NO PnL (status REJECTED, filled_volume 0);
  • PARTIAL fill reported as PARTIAL with the executed (< requested) volume;
  • live daily-halt STOPS orders (a halted guard blocks the gate);
  • EMERGENCY kill-switch stops all orders (and survives reset_day);
  • TRAIN/LIVE OBSERVATION PARITY (float-for-float feature matrix);
  • dry-run is the DEFAULT — orders are computed+logged but NOT transmitted;
    --live (live=True) is required to actually send.
"""
from __future__ import annotations

import numpy as np
import torch

from broker.mt5_adapter import MT5Adapter
from broker.live_runner import LiveRunner
from core.risk.trade_gate import TradeGate
from core.risk.daily_guard import DailyGuard
from core.settings import CFG
from core.agent.action_space import BUY, FLAT, EXIT_HOLD
from tests.mocks.mock_mt5 import MockMT5
from tests.fixtures.sample_configs import minimal_trading_policy
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")
BAR = {"time": "2025-01-01T10:00:00", "open": 1.1750, "high": 1.1760,
       "low": 1.1740, "close": 1.1755}


class _FakeAgent:
    """Deterministic stand-in for PPOAgent: always emits a fixed direction + lot."""
    def __init__(self, direction=BUY, lot_raw=0.5):
        self._dir, self._lot = direction, lot_raw

    def select_action(self, obs, deterministic=True, mask=None):
        return self._dir, self._lot, EXIT_HOLD

    def proportional_scale(self, target_pct, max_dd_pct):
        return 1.0


def _runner(agent, mock, *, live=False, halt=False, tmp_path=None):
    guard = DailyGuard("ftmo", 100_000, dict(CFG))
    if halt:
        guard.force_halt()
    gate = TradeGate(guard, log_path=str((tmp_path or ".") + "/gate.csv"))
    adapter = MT5Adapter(trade_gate=gate, mt5_module=mock)
    adapter.symbol = "EURUSD"
    return LiveRunner(agent, adapter, gate, guard, dict(CFG), policy={},
                      instrument="EURUSD",
                      accuracy_path=str((tmp_path or ".") + "/acc.json"),
                      heartbeat_path=str((tmp_path or ".") + "/hb.txt"),
                      live=live)


# ════════════════════════════════════════════════════════════════════════════
# Connection-failure retry/backoff + symbol-unavailable.
# ════════════════════════════════════════════════════════════════════════════
def test_connect_retries_then_succeeds():
    mock = MockMT5(known_symbols=("EURUSD",), fail_initialize_times=2)
    a = MT5Adapter(mt5_module=mock)
    cfg = minimal_trading_policy()["accounts"][0]
    cfg["symbol_aliases"] = {"EURUSD": ["EURUSD"]}
    ok = a.initialize_with_retry(cfg, max_attempts=5, base_delay=0.0)
    assert ok is True
    assert mock.init_calls == 3, "should have failed twice then connected on the 3rd"


def test_connect_gives_up_after_max_attempts():
    mock = MockMT5(known_symbols=("EURUSD",), fail_initialize_times=99)
    a = MT5Adapter(mt5_module=mock)
    cfg = minimal_trading_policy()["accounts"][0]
    cfg["symbol_aliases"] = {"EURUSD": ["EURUSD"]}
    assert a.initialize_with_retry(cfg, max_attempts=3, base_delay=0.0) is False
    assert mock.init_calls == 3


def test_symbol_unavailable_initialize_false():
    mock = MockMT5(known_symbols=("GBPUSD",))   # EURUSD not offered
    a = MT5Adapter(mt5_module=mock)
    cfg = minimal_trading_policy()["accounts"][0]
    cfg["symbol_aliases"] = {"EURUSD": ["EURUSD", "EURUSDm"]}
    cfg["instruments"] = ["EURUSD"]
    assert a.initialize(cfg) is False
    assert a.symbol is None, "must not bind to an unavailable symbol"


# ════════════════════════════════════════════════════════════════════════════
# Order format.
# ════════════════════════════════════════════════════════════════════════════
def test_order_request_format(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "FILLED"
    req = mock.sent_orders[-1]
    for k in ("symbol", "volume", "type", "price", "sl", "tp"):
        assert k in req, f"order request missing {k}"
    assert req["symbol"] == "EURUSD"
    assert isinstance(req["volume"], float) and req["volume"] > 0
    assert req["type"] in (mock.ORDER_TYPE_BUY, mock.ORDER_TYPE_SELL)


# ════════════════════════════════════════════════════════════════════════════
# Duplicate-order prevention.
# ════════════════════════════════════════════════════════════════════════════
def test_duplicate_order_suppressed(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    obs = torch.zeros(1, 16)
    r1 = r.step_bar(obs, BAR, max_lot=2.0, atr_14=0.001)
    r2 = r.step_bar(obs, BAR, max_lot=2.0, atr_14=0.001)   # same bar/decision
    assert r1["status"] == "FILLED"
    assert r2["status"] == "DUPLICATE"
    assert len(mock.sent_orders) == 1, "a duplicate bar must not transmit twice"


# ════════════════════════════════════════════════════════════════════════════
# Reconnect during an open trade.
# ════════════════════════════════════════════════════════════════════════════
def test_reconnect_on_link_drop_then_resend(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",), drop_link=True)
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    # Reconnect helper brings the link back up so the resend lands.
    orig_reconnect = r.adapter.reconnect
    def _reconnect(*a, **k):
        mock.set_link_up()
        return True
    r.adapter.reconnect = _reconnect
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "FILLED", "order should land after reconnect+resend"
    assert len(mock.sent_orders) == 1


def test_reconnect_fails_no_pnl(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",), drop_link=True)
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    r.adapter.reconnect = lambda *a, **k: False   # reconnect never succeeds
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "DISCONNECTED"
    assert res["filled_volume"] == 0.0, "a failed send must book NO fill / no PnL"
    assert len(mock.sent_orders) == 0


# ════════════════════════════════════════════════════════════════════════════
# Rejection records no PnL; partial fill reported.
# ════════════════════════════════════════════════════════════════════════════
def test_rejected_order_records_no_pnl(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",), reject_orders=True)
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "REJECTED"
    assert res["filled_volume"] == 0.0
    # a rejected order must NOT advance the dedup key (so a retry can fire).
    assert r._last_order_key is None


def test_partial_fill_reports_executed_volume(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",), partial_fill_ratio=0.5)
    r = _runner(_FakeAgent(direction=BUY, lot_raw=1.0), mock, live=True,
                tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "PARTIAL"
    assert 0.0 < res["filled_volume"] < res["requested_volume"]


# ════════════════════════════════════════════════════════════════════════════
# Daily-halt + emergency kill-switch stop orders.
# ════════════════════════════════════════════════════════════════════════════
def test_daily_halt_blocks_order(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, halt=True,
                tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "BLOCKED"
    assert len(mock.sent_orders) == 0


def test_kill_switch_stops_all_orders_and_survives_reset(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    r.engage_kill_switch("test")
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "KILLED"
    assert len(mock.sent_orders) == 0
    # A new day must NOT silently un-kill the runner.
    r.reset_day()
    res2 = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res2["status"] == "KILLED"


# ════════════════════════════════════════════════════════════════════════════
# Dry-run DEFAULT; --live required to transmit.
# ════════════════════════════════════════════════════════════════════════════
def test_dry_run_is_default_no_transmission(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=False, tmp_path=str(tmp_path))
    assert r.live is False, "dry-run must be the DEFAULT"
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "DRY_RUN"
    assert len(mock.sent_orders) == 0, "dry-run must NEVER transmit an order"


def test_live_flag_transmits(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=BUY), mock, live=True, tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "FILLED"
    assert len(mock.sent_orders) == 1


def test_cli_dry_run_default_and_live_flag():
    """The CLI must default to dry-run and only arm transmission on explicit --live."""
    import broker.live_runner as LR
    src = open(LR.__file__).read()
    assert 'action="store_true"' in src and '"--live"' in src, "--live flag missing"
    # default (no --live) returns cleanly without arming live mode
    import sys
    argv = sys.argv
    try:
        sys.argv = ["live_runner"]
        assert LR.main() == 0
    finally:
        sys.argv = argv


def test_flat_decision_never_sends(tmp_path):
    mock = MockMT5(known_symbols=("EURUSD",))
    r = _runner(_FakeAgent(direction=FLAT), mock, live=True, tmp_path=str(tmp_path))
    res = r.step_bar(torch.zeros(1, 16), BAR, max_lot=2.0, atr_14=0.001)
    assert res["status"] == "FLAT"
    assert len(mock.sent_orders) == 0


# ════════════════════════════════════════════════════════════════════════════
# TRAIN/LIVE OBSERVATION PARITY (float-for-float).
# ════════════════════════════════════════════════════════════════════════════
def test_observation_feature_parity_bit_for_bit():
    """The live feature builder and the training env feature builder are the SAME
    function (core.env.indicators.build_feature_matrix), so identical candles must
    yield a bit-for-bit identical float32 matrix — zero observation drift."""
    ohlcv = make_synthetic_ohlcv_array(n=300, seed=7)
    o, h, l, c, v = (ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])

    live_mat = LiveRunner.build_market_features(o, h, l, c, v, DEV)

    # The env's exact feature path (environment._ensure_feature_matrix -> the same
    # build_feature_matrix). Build it directly from the same module to compare.
    from core.env.indicators import build_feature_matrix
    train_mat = build_feature_matrix(o, h, l, c, v, DEV)

    assert live_mat.shape == train_mat.shape
    assert torch.equal(live_mat, train_mat), "live vs train features are not bit-identical"
