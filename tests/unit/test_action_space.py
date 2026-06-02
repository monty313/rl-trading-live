"""Unit tests for the 756-action space: roundtrip, bounds, bucket resolution."""
from core.agent import action_space as A


def test_num_actions_is_756():
    assert A.NUM_ACTIONS == 756


def test_encode_decode_roundtrip_all_756():
    seen = set()
    for d in range(A.N_DIRECTION):
        for lot in range(A.N_LOT):
            for sl in range(A.N_SL):
                for tp in range(A.N_TP):
                    a = A.encode(d, lot, sl, tp)
                    assert A.decode(a) == (d, lot, sl, tp)
                    seen.add(a)
    assert seen == set(range(756))
    assert min(seen) == 0 and max(seen) == 755


def test_decode_encode_roundtrip_all_ids():
    for x in range(A.NUM_ACTIONS):
        assert A.encode(*A.decode(x)) == x


def test_lot_resolution_and_clamp():
    assert A.get_lot(0, max_lot=2.0) == 0.01
    assert A.get_lot(6, max_lot=2.0) == 2.0      # special bucket -> max_lot
    assert A.get_lot(5, max_lot=0.10) == 0.10    # clamped to max_lot
    assert A.get_lot(0, max_lot=0.005) == 0.01   # floored at MT5 min


def test_sl_tp_tables():
    assert A.get_sl_pips(0) == 5 and A.get_sl_pips(5) == 50
    assert A.get_tp_pips(0) == 5 and A.get_tp_pips(5) == 50


def test_out_of_range_raises():
    import pytest
    with pytest.raises(ValueError):
        A.decode(756)
    with pytest.raises(ValueError):
        A.encode(3, 0, 0, 0)
