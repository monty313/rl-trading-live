"""Tests for the authoritative multi-indicator builder + resampler."""
import numpy as np
import pandas as pd
import torch
from core.env.indicators import (compute_indicators, resample_ohlcv,
                                  build_feature_matrix, FEATURE_COLUMNS, COL)
from tests.fixtures.sample_candles import make_synthetic_candles, make_synthetic_ohlcv_array


def test_compute_indicators_has_authoritative_columns():
    df = make_synthetic_candles(n=400).set_index("time")[
        ["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"})
    ind = compute_indicators(df)
    for col in ["cci30", "cci100", "cci30_sma1_sh8", "cci100_sma1_sh8",
                "atr14", "atr14_sma1_sh8", "atr45", "atr45_sma1_sh8",
                "bb20_upper", "bb20_mid", "bb200_upper", "bb200_mid",
                "high_sma4_sh8", "low_sma4_sh8", "sma2_sh0", "sma2_sh4",
                "bb20_upper_sma4_sh8", "adx14", "rsi7"]:
        assert col in ind.columns, f"missing {col}"


def test_cci300_and_cci900_removed():
    df = make_synthetic_candles(n=300).set_index("time")[
        ["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"})
    ind = compute_indicators(df)
    assert "cci300" not in ind.columns
    assert "cci900" not in ind.columns
    assert "cci900_sma20" not in ind.columns


def test_resample_to_15m():
    df = make_synthetic_candles(n=300).set_index("time")[
        ["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"})
    r15 = resample_ohlcv(df, 15)
    assert len(r15) < len(df)                  # fewer bars after resample
    assert (r15["high"] >= r15["low"]).all()   # valid OHLC
    assert resample_ohlcv(df, 1) is df         # 1m returns input unchanged


def test_feature_matrix_shape_and_columns():
    arr = make_synthetic_ohlcv_array(n=300)
    mat = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4],
                               torch.device("cpu"))
    assert mat.shape == (300, len(FEATURE_COLUMNS))
    assert torch.isfinite(mat).all()
    assert "cci30" in COL and "bb200_mid" in COL
