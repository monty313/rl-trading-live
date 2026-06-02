"""
Integration: indicators output must be BIT-IDENTICAL when built twice (the
parity guarantee that backs HARD RULE 10). Also checks the md5 parity helper.
"""
import torch
from core.env.indicators import build_feature_matrix
from backtest.engine import current_parity_hashes
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array


def test_indicators_bit_identical():
    arr = make_synthetic_ohlcv_array(n=600, seed=7)
    dev = torch.device("cpu")
    m1 = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], dev)
    m2 = build_feature_matrix(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], dev)
    assert torch.equal(m1, m2)   # bit-identical


def test_parity_hashes_present():
    h = current_parity_hashes()
    assert "indicators.py" in h and "intrabar_fills.py" in h
    assert len(h["indicators.py"]) == 32   # md5 hex length
