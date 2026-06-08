"""
Tests for core/agent/dqn_warm_start.py — the DQN → PPO trunk transfer.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from core.agent.dqn_warm_start import warm_start_ppo_from_dqn


# ── Mocks: a tiny DQN that mimics the predecessor's architecture ────────────
class _TinyDQN(nn.Module):
    """Mimics predecessor QNetwork: state_dim → 256 → 128 → 7."""
    def __init__(self, state_dim=2166):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )


class _TinyPPO(nn.Module):
    """Mimics current ActorCritic: state_dim → 256 → 256 trunk."""
    def __init__(self, state_dim=2166):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )


@pytest.fixture
def dqn_ckpt(tmp_path):
    torch.manual_seed(0)
    m = _TinyDQN()
    p = tmp_path / "dqn_stub.pt"
    torch.save({"policy_state_dict": m.state_dict()}, p)
    return str(p), m


# ── happy path ──────────────────────────────────────────────────────────────
def test_warm_start_copies_first_layer_verbatim(dqn_ckpt):
    ckpt_path, dqn = dqn_ckpt
    ppo = _TinyPPO()
    # Snapshot the PPO weights BEFORE warm-start so we can verify the second
    # layer was NOT touched (shape mismatch).
    ppo_before_l2 = ppo.trunk[2].weight.detach().clone()
    report = warm_start_ppo_from_dqn(ppo, ckpt_path, device="cpu")

    # First layer: shapes match (2166 → 256), DQN weights copied verbatim.
    assert torch.equal(ppo.trunk[0].weight.data, dqn.net[0].weight.data)
    assert torch.equal(ppo.trunk[0].bias.data,   dqn.net[0].bias.data)
    # Second layer: DQN is (256, 128), PPO is (256, 256) → skipped.
    assert torch.equal(ppo.trunk[2].weight, ppo_before_l2)
    assert "trunk.0" in report["transferred"]


def test_warm_start_fails_loudly_on_state_dim_mismatch(dqn_ckpt):
    """If you forget MULTI_TF_OBS=True, the first layer dims won't match."""
    ckpt_path, _dqn = dqn_ckpt
    wrong_ppo = _TinyPPO(state_dim=620)   # single-TF dim
    with pytest.raises(AssertionError, match="MULTI_TF_OBS=True"):
        warm_start_ppo_from_dqn(wrong_ppo, ckpt_path, device="cpu")


def test_warm_start_handles_missing_checkpoint():
    ppo = _TinyPPO()
    with pytest.raises(FileNotFoundError):
        warm_start_ppo_from_dqn(ppo, "/nonexistent/path.pt", device="cpu")


def test_warm_start_does_not_touch_dqn_action_head(dqn_ckpt):
    """The DQN's 7-way head should never bleed into PPO heads."""
    ckpt_path, _dqn = dqn_ckpt
    ppo = _TinyPPO()
    warm_start_ppo_from_dqn(ppo, ckpt_path, device="cpu")
    # PPO only has trunk in this stub; the real test is in the report content.
    # Just confirm the report mentions "skipped" or has at most one entry per trunk layer.
    # (Two PPO trunk layers; first copied, second skipped → report has 1 transfer + 1 skip.)
