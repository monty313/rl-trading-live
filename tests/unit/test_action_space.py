"""Unit tests for the PPO structured action space."""
from core.agent import action_space as A


def test_direction_and_exit_dims():
    assert A.DIRECTION_DIM == 3 and A.EXIT_DIM == 3
    assert (A.FLAT, A.BUY, A.SELL) == (0, 1, 2)
    assert A.HOLD == A.FLAT   # back-compat alias


def test_map_lot_range_and_clamp():
    assert A.map_lot(0.0, max_lot=2.0) == 0.01      # min floor
    assert A.map_lot(1.0, max_lot=2.0) == 2.0       # max
    mid = A.map_lot(0.5, max_lot=2.0)
    assert 0.01 < mid < 2.0
    assert A.map_lot(5.0, max_lot=0.10) == 0.10     # clamped above
    assert A.map_lot(-1.0, max_lot=2.0) == 0.01     # clamped below


def test_decode_structured_action():
    out = A.decode((A.BUY, 0.5, A.EXIT_CLOSE), max_lot=2.0)
    assert out["direction"] == A.BUY
    assert out["exit"] == A.EXIT_CLOSE
    assert 0.01 <= out["lot"] <= 2.0


def test_describe():
    d = A.describe(A.SELL, 1.0, A.EXIT_HOLD, max_lot=1.5)
    assert d["direction"] == "SELL" and d["lot"] == 1.5 and d["exit"] == "HOLD"
