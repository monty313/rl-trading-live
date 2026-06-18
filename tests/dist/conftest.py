# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY TEST HELPERS — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
"""Shared fixtures for the DIST pre-phase tests.

These build TINY stub objects so the tests do not need:
  - the real 1.74 GB DQN checkpoint,
  - real EURUSD data,
  - GPU acceleration.

The interface contracts they exercise are identical to the production
classes' contracts, so the tests validate real behavior — just with
small inputs.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn

# Make sure the repo root is importable when tests run from anywhere.
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── shared knobs ───────────────────────────────────────────────────────
PPO_OBS_DIM = 32
DQN_INPUT_DIM = 32  # match → no adapter
BATCH_SIZE = 4


# ── tiny stub DQN that we can dump to a tempfile and reload ────────────
class _TinyDQN(nn.Module):
    """Minimal MLP that mimics the DQN's policy head architecture."""

    def __init__(self, in_dim: int = DQN_INPUT_DIM, hidden: int = 16, out_dim: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)


def _build_state_dict_for_inferred_keys(model: _TinyDQN) -> dict:
    """Return a state_dict using the ``fc{i}`` prefix the loader knows."""
    sd = {}
    for i, layer in enumerate(model.layers):
        sd[f"layers.{i}.weight"] = layer.weight.detach().clone()
        sd[f"layers.{i}.bias"] = layer.bias.detach().clone()
    return sd


@pytest.fixture
def tiny_dqn_checkpoint(tmp_path) -> str:
    """Persist a fake DQN checkpoint with a recognizable state_dict."""
    torch.manual_seed(0)
    model = _TinyDQN()
    ckpt = {"policy_state_dict": _build_state_dict_for_inferred_keys(model)}
    path = tmp_path / "tiny_dqn.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture
def dist_config() -> Dict[str, Any]:
    """Minimal config dict that DistPhaseManager expects."""
    return {
        "dist_prephase_enabled": True,
        "dist_masking_enabled": True,
        "dist_teacher": {
            "checkpoint_path": "/tmp/does_not_matter.pt",
            "gdrive_file_id": "x",
            "action_order": ["BUY", "SELL", "HOLD"],
            "temperature": 1.0,
            "confidence_threshold": 0.55,
            "initial_distillation_weight": 0.30,
        },
        "dist_phase": {
            "prephase_max_daily_dd": 0.05,
            "prephase_daily_target": 0.02,
            "phase1_max_daily_dd": 0.01,
            "phase1_daily_target": 0.025,
            "required_gate_days": 10,
            "monotonic_fade": True,
            "gate_win_rate": 0.55,
            "gate_profit_factor": 1.3,
            "gate_expectancy_pips": 0.0,
            "gate_min_trades_per_day": 3,
            "signal2_agreement_normalized_min": 0.30,
            "signal2_rolling_window_days": 5,
            "signal3_solo_days_required": 3,
            "signal3_cooldown_days": 3,
            "graduation_record_path": str(
                Path(tempfile.gettempdir()) / "dist_grad_test.json"
            ),
        },
    }


# ── fake env that duck-types BatchedFTMOEnv enough for the wrapper ─────
class FakeBatchedEnv:
    """Tiny vectorized env stand-in.

    Implements only the surface used by DistPrePhaseWrapper:
      ``state_dim``, ``B``, ``device``, ``_position``,
      ``current_mask_and_force()``, ``reset()``, ``step(actions)``.
    """

    def __init__(self, B: int = BATCH_SIZE, base_dim: int = PPO_OBS_DIM):
        self.B = B
        self.state_dim = base_dim
        self.device = torch.device("cpu")
        self._position = torch.zeros(B)
        self._mask = torch.ones(B, 3)         # all directions allowed by default
        self._must_enter = torch.zeros(B, dtype=torch.bool)
        self._last_state = torch.randn(B, base_dim)

    def reset(self):
        self._position = torch.zeros(self.B)
        self._last_state = torch.randn(self.B, self.state_dim)
        return self._last_state.clone()

    def current_mask_and_force(self):
        return self._mask.clone(), self._must_enter.clone()

    def step(self, actions):
        direction = actions["direction"].long()
        # Update position to reflect "executed direction".
        new_pos = torch.zeros_like(self._position)
        new_pos = torch.where(direction == 1, torch.ones_like(new_pos), new_pos)  # BUY
        new_pos = torch.where(direction == 2, -torch.ones_like(new_pos), new_pos)  # SELL
        # If exit==CLOSE, force flat regardless.
        exit_act = actions.get("exit")
        if exit_act is not None:
            new_pos = torch.where(exit_act.long() == 2, torch.zeros_like(new_pos), new_pos)
        self._position = new_pos
        next_state = torch.randn(self.B, self.state_dim)
        reward = torch.zeros(self.B)
        done = torch.zeros(self.B, dtype=torch.bool)
        info = {"trades_today": torch.zeros(self.B, dtype=torch.long)}
        self._last_state = next_state
        return next_state, reward, done, info


@pytest.fixture
def fake_env() -> FakeBatchedEnv:
    return FakeBatchedEnv()
