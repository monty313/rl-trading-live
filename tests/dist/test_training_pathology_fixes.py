"""
Mini-test for the four training-pathology fixes shipped on feature/multi-tf-obs:

  1. The strategy mask matches the user's 4-cell rule table exactly:
        gate OFF + flat       → {FLAT}
        gate OFF + in trade   → {FLAT}           (HOLD or EXIT only, no flips)
        gate ON  + flat       → {BUY, SELL}      (must_enter=True)
        gate ON  + in trade   → {FLAT, BUY, SELL} (everything)

  2. PPO can exit any bar it wants — no anti-flicker cooldown anywhere.

  3. The dist bonus fires on HOLD bars when the held position direction
     matches the DQN's top action (not only on entry steps).

  4. initial_distillation_weight is now 0.60 (the bigger pull).

These four together should move the needle vs the prior overnight run where
PPO had P:0 average position and ~50% win rate. The test is intentionally
fast (under 1 second) so we can re-run after every refactor.
"""
from __future__ import annotations

import torch

from core.agent.action_space import BUY, SELL, FLAT, EXIT_HOLD, EXIT_CLOSE
from core.env.conditions_engine import compute_action_mask
from core.dist_teacher import DistDQNTeacher, DistPrePhaseWrapper
from core.dist_phase import DistPhaseManager, DistPhase


# ── 1. Mask invariants ──────────────────────────────────────────────────────
# Piggyback on the real MASK_REGISTRY (phase0_cci_extreme) and choose CCI rows
# so the gate condition fn returns True or False as desired.
DEV = torch.device("cpu")
_PHASE = {"mask": "phase0_cci_extreme", "mask_type": "force_in_and_gate",
          "gate_timeframes": [1, 15]}
_GATE_ON_ROW  = {"cci10": 150.0, "cci30": 120.0}   # extreme CCI -> condition True
_GATE_OFF_ROW = {"cci10":   0.0, "cci30":   0.0}   # calm CCI    -> condition False
_GATE_ON  = {1: _GATE_ON_ROW,  15: _GATE_ON_ROW}
_GATE_OFF = {1: _GATE_OFF_ROW, 15: _GATE_OFF_ROW}


def _allowed_dirs(mask):
    return {i for i, v in enumerate(mask.tolist()) if v == 1.0}


def test_mask_gate_off_flat_returns_flat_only():
    dir_mask, must_enter = compute_action_mask(_PHASE, _GATE_OFF, DEV, is_flat=True)
    assert _allowed_dirs(dir_mask) == {FLAT}
    assert must_enter is False


def test_mask_gate_off_in_trade_returns_flat_only_no_flips():
    """The new rule: in-trade + gate OFF → only HOLD/EXIT (mask=FLAT)."""
    dir_mask, must_enter = compute_action_mask(_PHASE, _GATE_OFF, DEV, is_flat=False)
    # User rule: no flips allowed when gate turns off mid-trade. PPO can still
    # exit via the exit head (EXIT_CLOSE) or hold (EXIT_HOLD); only the
    # direction head's BUY/SELL is blocked so no new direction can be opened.
    assert _allowed_dirs(dir_mask) == {FLAT}
    assert must_enter is False


def test_mask_gate_on_flat_forces_entry():
    dir_mask, must_enter = compute_action_mask(_PHASE, _GATE_ON, DEV, is_flat=True)
    assert _allowed_dirs(dir_mask) == {BUY, SELL}
    assert must_enter is True


def test_mask_gate_on_in_trade_full_freedom():
    dir_mask, must_enter = compute_action_mask(_PHASE, _GATE_ON, DEV, is_flat=False)
    assert _allowed_dirs(dir_mask) == {FLAT, BUY, SELL}
    assert must_enter is False


# ── 2. No anti-flicker cooldown anywhere ────────────────────────────────────
def test_min_hold_bars_not_in_cfg():
    """The anti-flicker setting must not exist; PPO can exit any bar."""
    from core.settings import CFG
    assert "MIN_HOLD_BARS" not in CFG, (
        "MIN_HOLD_BARS should not exist — PPO is allowed to exit any bar."
    )


def test_env_has_no_too_young_block():
    """Search the env source for the dead anti-flicker block."""
    src = open("core/env/environment.py").read()
    assert "too_young" not in src, (
        "Env still references too_young (anti-flicker cooldown). Must be removed."
    )
    assert "_min_hold_bars" not in src, (
        "Env still references _min_hold_bars. Must be removed."
    )


# ── 3. DQN bonus fires on HOLD bars when position direction matches DQN ────
def _make_test_wrapper(dist_cfg, fake_env, ckpt_path):
    mgr = DistPhaseManager(dist_cfg, start_phase=DistPhase.PRE_PHASE)
    teacher = DistDQNTeacher(
        checkpoint_path=ckpt_path,
        device="cpu",
        action_order=dist_cfg["dist_teacher"]["action_order"],
    )
    return DistPrePhaseWrapper(
        fake_env, teacher=teacher, dist_phase_manager=mgr,
        confidence_threshold=0.0,   # always confident in tests
        masking_enabled=True,
    ), teacher


def test_hold_bar_with_dqn_agreement_pays_bonus(
    fake_env, tiny_dqn_checkpoint, dist_config
):
    """PPO is long while DQN says BUY → bonus should fire on the HOLD bar."""
    env, teacher = _make_test_wrapper(dist_config, fake_env, tiny_dqn_checkpoint)
    # Force the DQN to want BUY decisively.
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[0] = 50.0   # canonical col 0 = BUY

    env.reset()
    # Step 1: open long (entry → entry-agree bonus fires).
    actions_buy = {
        "direction": torch.full((fake_env.B,), BUY, dtype=torch.long),
        "lot_raw":   torch.full((fake_env.B,), 0.5),
        "exit":      torch.zeros(fake_env.B, dtype=torch.long),
    }
    _, _, _, info_entry = env.step(actions_buy)
    # Step 2: hold the long (PPO picks FLAT direction — would normally pay no
    # bonus because it's not an entry — but the new hold-agreement rule says
    # an already-long position is "agreeing with BUY" so bonus should fire).
    actions_hold = {
        "direction": torch.zeros(fake_env.B, dtype=torch.long),   # FLAT (no flip)
        "lot_raw":   torch.full((fake_env.B,), 0.5),
        "exit":      torch.zeros(fake_env.B, dtype=torch.long),   # EXIT_HOLD
    }
    _, _, _, info_hold = env.step(actions_hold)
    # Hold-bar bonus must be > 0 (DQN says BUY, position is long → agreement).
    assert (info_hold["dist_bonus"] > 0).any(), (
        "hold-bar bonus did not fire when PPO held a long while DQN said BUY"
    )


def test_hold_bar_with_dqn_disagreement_pays_nothing(
    fake_env, tiny_dqn_checkpoint, dist_config
):
    """PPO is long but DQN says SELL → no bonus on the HOLD bar."""
    env, teacher = _make_test_wrapper(dist_config, fake_env, tiny_dqn_checkpoint)
    # Force DQN to want SELL.
    with torch.no_grad():
        teacher.model.layers[-1].weight.zero_()
        teacher.model.layers[-1].bias.zero_()
        teacher.model.layers[-1].bias[1] = 50.0   # canonical col 1 = SELL

    env.reset()
    # Open long despite DQN disagreement.
    actions_buy = {
        "direction": torch.full((fake_env.B,), BUY, dtype=torch.long),
        "lot_raw":   torch.full((fake_env.B,), 0.5),
        "exit":      torch.zeros(fake_env.B, dtype=torch.long),
    }
    env.step(actions_buy)
    # Hold the disagreeing position.
    actions_hold = {
        "direction": torch.zeros(fake_env.B, dtype=torch.long),
        "lot_raw":   torch.full((fake_env.B,), 0.5),
        "exit":      torch.zeros(fake_env.B, dtype=torch.long),
    }
    _, _, _, info_hold = env.step(actions_hold)
    assert torch.allclose(info_hold["dist_bonus"], torch.zeros_like(info_hold["dist_bonus"]))


# ── 4. Bigger dist weight in CFG ────────────────────────────────────────────
def test_initial_distillation_weight_bumped_to_0p6():
    from core.settings import CFG
    assert CFG["dist_teacher"]["initial_distillation_weight"] == 0.60, (
        "initial_distillation_weight should be 0.60 (bumped from 0.30 — "
        "stronger early DQN pull)."
    )
