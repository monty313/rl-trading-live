"""Unit tests for MT5Adapter using the mock MT5 module (Colab/Linux safe)."""
import csv
from broker.mt5_adapter import MT5Adapter
from core.risk.trade_gate import TradeGate
from core.risk.daily_guard import DailyGuard
from core.settings import CFG
from tests.mocks.mock_mt5 import MockMT5
from tests.fixtures.sample_configs import minimal_trading_policy

ORDER = {"symbol": "EURUSD", "direction": "BUY", "lot": 0.1,
         "sl": 1.17, "tp": 1.18, "entry": 1.175}


def test_symbol_alias_resolves():
    mock = MockMT5(known_symbols=("EURUSDm",))
    a = MT5Adapter(mt5_module=mock)
    acct = minimal_trading_policy()["accounts"][0]
    acct["symbol_aliases"] = {"EURUSD": ["EURUSD", "EURUSDm"]}
    assert a.initialize(acct) is True
    assert a.symbol == "EURUSDm"


def test_send_order_calls_gate_and_blocks(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG)); g.force_halt()
    gate = TradeGate(g, log_path=str(tmp_path / "log.csv"))
    a = MT5Adapter(trade_gate=gate, mt5_module=MockMT5())
    res = a.send_order(ORDER)
    assert res["status"] == "BLOCKED"
    rows = list(csv.reader(open(tmp_path / "log.csv")))
    assert any("BLOCKED" in r for r in rows)


def test_send_order_fills_when_approved(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG))
    gate = TradeGate(g, log_path=str(tmp_path / "log.csv"))
    mock = MockMT5(known_symbols=("EURUSD",))
    a = MT5Adapter(trade_gate=gate, mt5_module=mock)
    a.symbol = "EURUSD"
    res = a.send_order(ORDER)
    assert res["status"] == "FILLED"
    assert len(mock.sent_orders) == 1
