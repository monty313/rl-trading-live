"""Unit tests for PositionSizer clamping + floor."""
from core.risk.position_sizer import PositionSizer


def test_clamped_to_max_lot():
    s = PositionSizer()
    assert s.size(6, max_lot=0.5) == 0.5     # bucket 6 -> max_lot
    assert s.size(5, max_lot=0.10) == 0.10   # 0.50 bucket clamped to 0.10


def test_minimum_lot_floor():
    s = PositionSizer()
    assert s.size(0, max_lot=0.001) == 0.01  # floored at MT5 min


def test_no_crash_on_any_bucket():
    s = PositionSizer()
    for i in range(7):
        lot = s.size(i, max_lot=2.0, balance=100000, sl_pips=20)
        assert 0.01 <= lot <= 2.0
