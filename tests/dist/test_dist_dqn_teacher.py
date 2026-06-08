# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY TEST FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
"""Tests for DistDQNTeacher: freezing, batch inference, retirement freeze
value, and obs adapter behavior."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.dist_teacher.dist_dqn_teacher import DistDQNTeacher
from core.dist_teacher.dist_obs_adapter import DistObsAdapter, build_adapter_if_needed


def test_dist_loads_frozen(tiny_dqn_checkpoint):
    t = DistDQNTeacher(
        checkpoint_path=tiny_dqn_checkpoint,
        device="cpu",
        action_order=["BUY", "SELL", "HOLD"],
    )
    assert t.is_frozen is True
    # Every parameter must have requires_grad False
    for p in t.model.parameters():
        assert p.requires_grad is False


def test_dist_predict_probs_shape_and_sum(tiny_dqn_checkpoint):
    t = DistDQNTeacher(
        checkpoint_path=tiny_dqn_checkpoint,
        device="cpu",
        action_order=["BUY", "SELL", "HOLD"],
    )
    obs = torch.randn(8, 32)
    probs = t.predict_probs_batch(obs)
    assert probs.shape == (8, 3)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5)


def test_dist_no_weight_update(tiny_dqn_checkpoint):
    """50 batches of inference must not modify any parameter."""
    t = DistDQNTeacher(
        checkpoint_path=tiny_dqn_checkpoint,
        device="cpu",
        action_order=["BUY", "SELL", "HOLD"],
    )
    before = t._compute_checksum()
    for _ in range(50):
        t.predict_probs_batch(torch.randn(4, 32))
    after = t._compute_checksum()
    assert before == after
    assert t.verify_no_drift() is True


def test_dist_retirement_freeze_value_tracks_mean(tiny_dqn_checkpoint):
    t = DistDQNTeacher(
        checkpoint_path=tiny_dqn_checkpoint,
        device="cpu",
        action_order=["BUY", "SELL", "HOLD"],
    )
    # With no batches seen, fall back to uniform.
    fv = t.get_retirement_freeze_value()
    assert np.allclose(fv, np.full(3, 1.0 / 3.0), atol=1e-6)

    seen = []
    for _ in range(20):
        probs = t.predict_probs_batch(torch.randn(8, 32))
        seen.append(probs.mean(dim=0).detach().cpu().numpy())
    expected = np.mean(seen, axis=0)
    fv2 = t.get_retirement_freeze_value()
    # Should equal cumulative running mean (close to mean-of-means since
    # batch sizes are constant).
    assert np.allclose(fv2.sum(), 1.0, atol=1e-5)
    assert np.allclose(fv2, expected, atol=1e-5)


def test_dist_obs_adapter_slices_correctly():
    adapter = DistObsAdapter(ppo_obs_dim=40, dqn_input_dim=32)
    x = torch.arange(40, dtype=torch.float32).expand(5, 40)
    y = adapter.adapt(x)
    assert y.shape == (5, 32)
    assert torch.equal(y, x[..., :32])


def test_dist_obs_adapter_builder_returns_none_on_match():
    assert build_adapter_if_needed(32, 32) is None
    adapter = build_adapter_if_needed(40, 32)
    assert isinstance(adapter, DistObsAdapter)


def test_dist_action_order_reorders_columns(tiny_dqn_checkpoint):
    """Specifying a non-canonical action_order should reorder columns."""
    # Re-order BUY/SELL so that the teacher believes column 0 is SELL.
    t = DistDQNTeacher(
        checkpoint_path=tiny_dqn_checkpoint,
        device="cpu",
        action_order=["SELL", "BUY", "HOLD"],
    )
    # When we ask for probs they should be reordered into canonical
    # [BUY, SELL, HOLD] — so column 0 of output corresponds to checkpoint
    # column 1 (BUY in their world).
    fixed_input = torch.zeros(1, 32)
    out = t.predict_probs_batch(fixed_input)
    assert out.shape == (1, 3)
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-5)
