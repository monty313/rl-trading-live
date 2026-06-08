# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY TEST FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
"""Mask-interaction tests. The dist wrapper does not REPLACE the env's masking
(see core/env/environment.py current_mask_and_force) — it only refuses to give
the distillation bonus when the DQN's preferred direction is masked. These
tests exercise that contract."""
from __future__ import annotations

import torch

from core.agent.action_space import BUY, SELL, FLAT, EXIT_HOLD
from core.dist_teacher import DistDQNTeacher, DistPrePhaseWrapper
from core.dist_phase import DistPhaseManager, DistPhase


def _bias_teacher(teacher, col: int, magnitude: float = 50.0):
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[col] = magnitude


def _setup(env, ckpt, cfg, masking=True):
    mgr = DistPhaseManager(cfg, start_phase=DistPhase.PRE_PHASE)
    teacher = DistDQNTeacher(
        checkpoint_path=ckpt,
        device="cpu",
        action_order=["BUY", "SELL", "HOLD"],
    )
    w = DistPrePhaseWrapper(
        env,
        teacher=teacher,
        dist_phase_manager=mgr,
        confidence_threshold=0.0,
        masking_enabled=masking,
    )
    return w, mgr, teacher


def _actions(B, direction):
    return {
        "direction": torch.full((B,), direction, dtype=torch.long),
        "lot_raw":   torch.full((B,), 0.5),
        "exit":      torch.full((B,), EXIT_HOLD, dtype=torch.long),
    }


def test_dist_bonus_zero_for_masked_dqn_action(fake_env, tiny_dqn_checkpoint, dist_config):
    """Mask BUY; force DQN to want BUY; PPO picks BUY (entry). Bonus must be 0."""
    # Mask convention (env): m[FLAT=0]=allow_flat, m[BUY=1]=allow_buy, m[SELL=2]=allow_sell.
    fake_env._mask = torch.tensor([[1.0, 0.0, 1.0]] * fake_env.B)  # BUY masked
    env, _mgr, teacher = _setup(fake_env, tiny_dqn_checkpoint, dist_config)
    _bias_teacher(teacher, col=0)  # canonical column 0 = BUY
    env.reset()
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    assert torch.allclose(info["dist_bonus"], torch.zeros_like(info["dist_bonus"]))


def test_dist_bonus_fires_when_dqn_action_allowed(fake_env, tiny_dqn_checkpoint, dist_config):
    """SELL masked but DQN wants BUY (allowed) — bonus should fire."""
    fake_env._mask = torch.tensor([[1.0, 1.0, 0.0]] * fake_env.B)  # SELL masked
    env, _mgr, teacher = _setup(fake_env, tiny_dqn_checkpoint, dist_config)
    _bias_teacher(teacher, col=0)  # BUY
    env.reset()
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    assert (info["dist_bonus"] > 0).all()


def test_dist_masking_disabled_flag_ignores_mask(fake_env, tiny_dqn_checkpoint, dist_config):
    """With masking disabled, the bonus may still fire even when DQN action is masked."""
    fake_env._mask = torch.tensor([[1.0, 0.0, 1.0]] * fake_env.B)  # BUY masked
    env, _mgr, teacher = _setup(
        fake_env, tiny_dqn_checkpoint, dist_config, masking=False
    )
    _bias_teacher(teacher, col=0)  # DQN wants BUY (masked)
    env.reset()
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    # With masking_enabled=False, agreement counts even on masked direction.
    assert (info["dist_bonus"] > 0).all()


def test_dist_bonus_only_for_buy_or_sell_not_flat(fake_env, tiny_dqn_checkpoint, dist_config):
    """Even if DQN says HOLD with high confidence, FLAT is not an entry → bonus 0."""
    env, _mgr, teacher = _setup(fake_env, tiny_dqn_checkpoint, dist_config)
    _bias_teacher(teacher, col=2)  # HOLD column
    env.reset()
    _, _, _, info = env.step(_actions(fake_env.B, FLAT))
    assert torch.allclose(info["dist_bonus"], torch.zeros_like(info["dist_bonus"]))
