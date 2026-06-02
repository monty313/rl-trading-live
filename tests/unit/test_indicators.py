"""Unit tests for the GPU-tensor feature matrix builder."""
import torch
from core.env.indicators import build_feature_matrix, NUM_FEATURES, COL
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array


def test_shape_and_finite():
    arr = make_synthetic_ohlcv_array(n=300)
    dev = torch.device("cpu")
    mat = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], dev)
    assert mat.shape == (300, NUM_FEATURES)
    assert torch.isfinite(mat).all()


def test_named_columns_present():
    for name in ["close", "sma_20", "ema_20", "cci_14", "atr_14",
                 "atr_14_ma", "rolling_high_20", "rolling_low_20"]:
        assert name in COL


def test_rolling_high_ge_close():
    arr = make_synthetic_ohlcv_array(n=200)
    dev = torch.device("cpu")
    mat = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], dev)
    # rolling 20-high should be >= the high at each bar's own position eventually
    assert (mat[:, COL["rolling_high_20"]] >= mat[:, COL["rolling_low_20"]]).all()


def test_device_placement_cpu():
    arr = make_synthetic_ohlcv_array(n=100)
    dev = torch.device("cpu")
    mat = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], dev)
    assert mat.device.type == "cpu"
