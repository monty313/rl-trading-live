"""Unit tests for DailyGuard FTMO + Beast halts and PASS/FAIL."""
from core.risk.daily_guard import DailyGuard
from core.settings import CFG


def _cfg():
    return dict(CFG)


def test_ftmo_halts_at_dd_threshold():
    g = DailyGuard("ftmo", 100000, _cfg())
    g.update(equity=100000 * (1 - 0.011))   # 1.1% DD > 1% limit
    assert g.get_status()["halted"] is True


def test_ftmo_no_halt_within_limit():
    g = DailyGuard("ftmo", 100000, _cfg())
    g.update(equity=100000 * (1 - 0.005))   # 0.5% DD
    assert g.get_status()["halted"] is False


def test_pass_recorded_when_target_hit():
    g = DailyGuard("ftmo", 100000, _cfg())
    g.update(equity=100000 * 1.026)         # +2.6% > +2.5% ($2,500) fixed target
    assert g.pass_fail() == "PASS"          # binary now: PASS or FAIL only


def test_pass_fail_is_binary_below_target():
    # A small green day that does NOT reach the fixed increment is a FAIL (no OK).
    g = DailyGuard("ftmo", 10000, _cfg())   # increment = $250
    g.update(equity=10000 + 20.86)          # the +$20.86 regression case
    assert g.pass_fail() == "FAIL"


def test_force_halt():
    g = DailyGuard("ftmo", 100000, _cfg())
    g.force_halt()
    assert g.get_status()["halted"] is True


def test_beast_trailing_from_peak():
    c = _cfg(); c["BEAST_TRAILING_DD_PCT"] = 0.05
    g = DailyGuard("beast", 100000, c)
    g.update(equity=110000)                 # new peak
    g.update(equity=110000 * (1 - 0.051))   # 5.1% from peak > 5%
    assert g.get_status()["halted"] is True


def test_trade_cap_halts():
    c = _cfg(); c["MAX_TRADES_PER_DAY"] = 800
    g = DailyGuard("ftmo", 100000, c)
    g.update(equity=100000, trade_count=800)
    assert g.get_status()["halted"] is True
