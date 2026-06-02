"""Unit tests for TradeGate approval + BLOCKED logging."""
import csv
from core.risk.trade_gate import TradeGate
from core.risk.daily_guard import DailyGuard
from core.settings import CFG

ORDER = {"symbol": "EURUSD", "direction": "BUY", "lot": 0.1,
         "sl": 1.17, "tp": 1.18, "entry": 1.175}


def test_approve_true_when_not_halted(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG))
    gate = TradeGate(g, log_path=str(tmp_path / "log.csv"))
    assert gate.approve(ORDER) is True


def test_approve_false_when_halted(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG))
    g.force_halt()
    log = str(tmp_path / "log.csv")
    gate = TradeGate(g, log_path=log)
    assert gate.approve(ORDER) is False
    rows = list(csv.reader(open(log)))
    assert any("BLOCKED" in r for r in rows)   # BLOCKED written to log


def test_consent_blocks(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG))
    gate = TradeGate(g, log_path=str(tmp_path / "log.csv"))
    gate.set_consent(False)
    assert gate.approve(ORDER) is False
