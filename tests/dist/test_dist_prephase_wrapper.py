# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY TEST FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
"""Tests for DistPrePhaseWrapper: obs augmentation, entry-step bonus,
mask interaction, and base-reward integrity."""
from __future__ import annotations

import torch

from core.agent.action_space import BUY, SELL, FLAT, EXIT_HOLD, EXIT_CLOSE
from core.dist_teacher import DistDQNTeacher, DistPrePhaseWrapper
from core.dist_phase import DistPhaseManager, DistPhase


def _build(env, ckpt_path, dist_cfg, masking=True, phase=DistPhase.PRE_PHASE):
    mgr = DistPhaseManager(dist_cfg, start_phase=phase)
    teacher = DistDQNTeacher(
        checkpoint_path=ckpt_path,
        device="cpu",
        action_order=dist_cfg["dist_teacher"]["action_order"],
    )
    wrapped = DistPrePhaseWrapper(
        env,
        teacher=teacher,
        dist_phase_manager=mgr,
        confidence_threshold=0.0,    # always confident in tests
        masking_enabled=masking,
    )
    return wrapped, mgr, teacher


def _actions(B: int, direction_val: int, exit_val: int = EXIT_HOLD):
    return {
        "direction": torch.full((B,), direction_val, dtype=torch.long),
        "lot_raw":   torch.full((B,), 0.5),
        "exit":      torch.full((B,), exit_val, dtype=torch.long),
    }


def test_dist_obs_dim_extended(fake_env, tiny_dqn_checkpoint, dist_config):
    env, _mgr, _t = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    s = env.reset()
    assert s.shape[-1] == fake_env.state_dim + 3


def test_dist_obs_dim_constant_across_phases(fake_env, tiny_dqn_checkpoint, dist_config):
    env, mgr, _t = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    s_pre = env.reset()
    mgr.advance_to_phase_1()
    # Step once so the wrapper rebuilds obs in PHASE_1.
    actions = _actions(fake_env.B, BUY)
    s_phase1, _, _, _ = env.step(actions)
    # Force retirement to test post-graduation observation shape.
    mgr._retire_teacher()
    s_retired, _, _, _ = env.step(_actions(fake_env.B, FLAT))
    assert s_pre.shape == s_phase1.shape == s_retired.shape


def test_dist_bonus_fires_only_on_entry_steps(fake_env, tiny_dqn_checkpoint, dist_config):
    env, _mgr, _t = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    env.reset()

    # First step: was flat → BUY. Should be an entry step for ALL B rows.
    _, _, _, info1 = env.step(_actions(fake_env.B, BUY))
    entry_mask_1 = info1["dist_entry_step_mask"]
    assert bool(entry_mask_1.all().item())

    # Second step: BUY again (already long, no flip) → NOT entry. Bonus 0.
    _, _, _, info2 = env.step(_actions(fake_env.B, BUY))
    entry_mask_2 = info2["dist_entry_step_mask"]
    assert not bool(entry_mask_2.any().item())
    assert torch.allclose(info2["dist_bonus"], torch.zeros_like(info2["dist_bonus"]))


def test_dist_bonus_zero_when_disabled(fake_env, tiny_dqn_checkpoint, dist_config):
    cfg = dict(dist_config)
    cfg["dist_prephase_enabled"] = False
    env, _mgr, _t = _build(fake_env, tiny_dqn_checkpoint, cfg)
    env.reset()
    _, total_reward, _, info = env.step(_actions(fake_env.B, BUY))
    assert torch.allclose(info["dist_bonus"], torch.zeros_like(info["dist_bonus"]))
    # And total reward must equal the base reward (zero in fake env).
    assert torch.allclose(total_reward, torch.zeros_like(total_reward))


def test_dist_base_reward_never_modified(fake_env, tiny_dqn_checkpoint, dist_config):
    # Override fake env reward to a constant so we can verify additivity.
    base_r = torch.tensor([0.5, -0.25, 1.0, 0.0])
    orig_step = fake_env.step

    def step_with_reward(actions):
        next_state, _zero_r, done, info = orig_step(actions)
        return next_state, base_r.clone(), done, info

    fake_env.step = step_with_reward
    env, _mgr, _t = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    env.reset()
    _, total, _, info = env.step(_actions(fake_env.B, BUY))
    bonus = info["dist_bonus"]
    # Total = base + bonus, base untouched.
    assert torch.allclose(total - bonus, base_r, atol=1e-6)


def test_dist_bonus_zero_for_masked_dqn_action(fake_env, tiny_dqn_checkpoint, dist_config):
    """If DQN says BUY but BUY is masked, bonus must be 0."""
    # Mask convention (env): m[:, DIRECTION_DIM] with 1.0=allowed, 0.0=masked.
    # Columns are [FLAT=0, BUY=1, SELL=2]. So masking BUY = zeroing column 1.
    fake_env._mask = torch.tensor([[1.0, 0.0, 1.0]] * fake_env.B)  # BUY masked

    env, _mgr, teacher = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    # Force the teacher to return overwhelmingly-BUY by hacking weights.
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[0] = 50.0  # column 0 = BUY in canonical order

    env.reset()
    # PPO picks BUY (which is masked). DQN also wants BUY. Bonus must be 0.
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    assert torch.allclose(info["dist_bonus"], torch.zeros_like(info["dist_bonus"]))


def test_dist_bonus_zero_on_disagreement(fake_env, tiny_dqn_checkpoint, dist_config):
    env, _mgr, teacher = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    # Force teacher to overwhelmingly want SELL.
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[1] = 50.0  # column 1 = SELL

    env.reset()
    # PPO picks BUY → disagreement → bonus 0.
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    assert torch.allclose(info["dist_bonus"], torch.zeros_like(info["dist_bonus"]))


def test_dist_bonus_fires_on_flip(fake_env, tiny_dqn_checkpoint, dist_config):
    env, _mgr, teacher = _build(fake_env, tiny_dqn_checkpoint, dist_config)
    # Strong BUY signal from teacher.
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[0] = 50.0  # BUY column

    env.reset()
    # First go SHORT — entry step but DQN wants BUY → no bonus.
    _, _, _, _ = env.step(_actions(fake_env.B, SELL))
    # Now flip to BUY — entry step (flip) AND agrees with DQN → bonus > 0.
    _, _, _, info = env.step(_actions(fake_env.B, BUY))
    assert (info["dist_bonus"] > 0).all()
