"""
Tests for core/env/legacy_multitf_state.py and the indicators_legacy port.
These are the load-bearing modules that make the DQN checkpoint usable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.env.legacy_multitf_state import (
    LEGACY_LOOKBACK,
    LEGACY_NUM_FEATURES,
    LEGACY_STATE_DIM,
    LEGACY_TF_FACTORS,
    LegacyMultiTFStateProvider,
    collapse_dqn_7_to_3_probs,
)
from core.env.indicators_legacy import build_feature_matrix as legacy_build_features


# ── invariants ──────────────────────────────────────────────────────────────
def test_legacy_state_dim_is_exactly_2166():
    """The whole point of this code path is to match the DQN checkpoint."""
    assert LEGACY_STATE_DIM == 2166
    assert LEGACY_LOOKBACK == 20
    assert LEGACY_NUM_FEATURES == 27
    assert LEGACY_TF_FACTORS == (1, 15, 60, 1440)


def _synthetic_ohlcv(T=5000, seed=0):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, T))
    high  = close + np.abs(rng.normal(0, 1e-4, T))
    low   = close - np.abs(rng.normal(0, 1e-4, T))
    op    = close + rng.normal(0, 1e-4, T)
    vol   = np.abs(rng.normal(0, 100, T))
    return np.column_stack([op, high, low, close, vol]).astype(np.float64)


# ── feature builder ─────────────────────────────────────────────────────────
def test_legacy_build_features_shape_and_count():
    ohlcv = _synthetic_ohlcv(T=2000)
    mat = legacy_build_features(*[ohlcv[:, i] for i in range(5)])
    assert mat.shape == (2000, 27)
    # No NaNs after warmup
    assert np.isfinite(mat[500:]).all()


# ── state provider ──────────────────────────────────────────────────────────
def test_provider_state_shape():
    ohlcv = _synthetic_ohlcv(T=5000)
    prov = LegacyMultiTFStateProvider(ohlcv, device="cpu")
    B = 4
    abs_idx = torch.tensor([3000, 3500, 4000, 4500], dtype=torch.long)
    zeros = torch.zeros(B)
    ones = torch.ones(B)
    s = prov.build_state(
        abs_idx=abs_idx,
        position=zeros,
        entry_px=zeros,
        equity=ones * 10000.0,
        initial_equity=ones * 10000.0,
        day_start_eq=ones * 10000.0,
        day_high_eq=ones * 10000.0,
        target_pct=ones * 0.025,
        max_dd_pct=ones * 0.01,
    )
    assert s.shape == (B, 2166)
    assert torch.isfinite(s).all()


def test_provider_trailing_features_order():
    """Order MUST be [position, unrealised, eq_chg, gap, dd_head, daily_ret] to match DQN."""
    ohlcv = _synthetic_ohlcv(T=3000)
    prov = LegacyMultiTFStateProvider(ohlcv, device="cpu")
    B = 2
    abs_idx = torch.tensor([2000, 2500], dtype=torch.long)
    s = prov.build_state(
        abs_idx=abs_idx,
        position=torch.tensor([1.0, -1.0]),       # long, short
        entry_px=torch.tensor([1.10, 1.10]),
        equity=torch.tensor([10100.0, 9900.0]),
        initial_equity=torch.tensor([10000.0, 10000.0]),
        day_start_eq=torch.tensor([10000.0, 10000.0]),
        day_high_eq=torch.tensor([10100.0, 10050.0]),
        target_pct=torch.tensor([0.025, 0.025]),
        max_dd_pct=torch.tensor([0.01, 0.01]),
    )
    # The last 6 columns are the trailing features.
    trailing = s[:, -6:]
    # position is the first trailing feature.
    assert trailing[0, 0].item() == pytest.approx(1.0)
    assert trailing[1, 0].item() == pytest.approx(-1.0)
    # eq_chg = (equity - initial) / initial → +0.01 long row, -0.01 short row.
    assert trailing[0, 2].item() == pytest.approx(0.01, abs=1e-4)
    assert trailing[1, 2].item() == pytest.approx(-0.01, abs=1e-4)


def test_provider_resample_matches_predecessor_logic():
    """Resampling = last bar of each TF window (idx = arange(tf-1, T, tf))."""
    ohlcv = _synthetic_ohlcv(T=10000)
    prov = LegacyMultiTFStateProvider(ohlcv, device="cpu")
    # For tf=15, the resampled tensor should have ceil(T/15) bars roughly.
    feat_15 = prov._resampled[15]
    assert feat_15.shape[0] == len([i for i in range(14, 10000, 15)])
    # For tf=1, identity.
    assert torch.equal(prov._resampled[1], prov._feat_1m)


# ── 7-action collapse ───────────────────────────────────────────────────────
def test_collapse_7_to_3_exact():
    """idx0=HOLD; 1-3 sum to BUY; 4-6 sum to SELL."""
    probs7 = torch.tensor([[0.1, 0.05, 0.10, 0.25, 0.05, 0.15, 0.30]])
    probs3 = collapse_dqn_7_to_3_probs(probs7)
    # BUY = 0.05 + 0.10 + 0.25 = 0.40
    # SELL = 0.05 + 0.15 + 0.30 = 0.50
    # HOLD = 0.10
    assert probs3[0, 0].item() == pytest.approx(0.40, abs=1e-6)
    assert probs3[0, 1].item() == pytest.approx(0.50, abs=1e-6)
    assert probs3[0, 2].item() == pytest.approx(0.10, abs=1e-6)
    assert probs3.sum(-1).item() == pytest.approx(1.0, abs=1e-6)


def test_collapse_batched():
    probs7 = torch.softmax(torch.randn(32, 7), dim=-1)
    probs3 = collapse_dqn_7_to_3_probs(probs7)
    assert probs3.shape == (32, 3)
    assert torch.allclose(probs3.sum(-1), torch.ones(32), atol=1e-5)


def test_collapse_rejects_wrong_dim():
    with pytest.raises(AssertionError):
        collapse_dqn_7_to_3_probs(torch.softmax(torch.randn(4, 5), dim=-1))
