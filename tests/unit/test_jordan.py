"""Unit tests for Jordan: IRAC markdown, persona fallback, consent flow, test access."""
import os
import torch
from jordan.irac_engine import generate_irac
from jordan import persona
from jordan.consent_flow import ConsentFlow
from core.risk.trade_gate import TradeGate
from core.risk.daily_guard import DailyGuard
from core.settings import CFG


def test_irac_has_four_sections():
    md = generate_irac("test_failure", {"test": "t_x", "assertion": "a==b",
                                        "file": "tests/unit/x.py"})
    for sec in ("**ISSUE**", "**RULE**", "**APPLICATION**", "**CONCLUSION**"):
        assert sec in md


def test_persona_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    out = persona.get_response({}, "how am I doing?")
    assert isinstance(out, str) and len(out) > 0


def test_consent_requires_two_steps(tmp_path):
    g = DailyGuard("ftmo", 100000, dict(CFG))
    gate = TradeGate(g, log_path=str(tmp_path / "log.csv"))
    cf = ConsentFlow(gate, reports_dir=str(tmp_path / "reports"))
    cf.begin()
    assert gate.consent_granted is False     # trades blocked while pending
    import pytest
    with pytest.raises(PermissionError):
        cf.approve_deploy("irac", "diff")    # cannot deploy before step 1
    cf.approve_idea()
    path = cf.approve_deploy("irac text", "diff text")
    assert os.path.exists(path)              # only file Jordan may write
    assert gate.consent_granted is True


def test_jordan_can_read_test_results(tmp_path):
    from tests.run_all_tests import get_last_results, jordan_summary
    r = get_last_results()
    assert "summary" in r
    assert isinstance(jordan_summary(), str)
