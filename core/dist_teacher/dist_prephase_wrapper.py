# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
# Gym-style wrapper around the BatchedFTMOEnv. Adds two things:
#   1. Three DQN probability slots appended to every observation
#      (constant obs dimensionality across ALL dist phases — when the
#      teacher retires the slots freeze to the EMPIRICAL mean, not 0.333).
#   2. A direction-distillation BONUS reward that fires ONLY on trade
#      ENTRY steps (open or flip). Crucially NOT on every bar — that would
#      drown the PnL signal. Bonus is gated by:
#         - dist_phase_manager.get_distillation_weight() > 0
#         - DQN confidence >= configured threshold
#         - PPO direction == DQN top action
#         - DQN top action is NOT currently masked
#
# Base reward is NEVER modified. The bonus is additive and logged
# separately in ``info["dist_bonus"]``.
#
# REVERT: delete with the rest of core/dist_teacher/.
# ═══════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

# Direction codes — single source of truth (core/agent/action_space.py).
from core.agent.action_space import BUY, SELL, FLAT


# Canonical column order for the appended DQN slots — keep in sync with
# DistDQNTeacher.predict_probs_batch().
_DQN_BUY_COL, _DQN_SELL_COL, _DQN_HOLD_COL = 0, 1, 2
_CANONICAL_ACTIONS = ["BUY", "SELL", "HOLD"]


class DistPrePhaseWrapper:
    """[DIST] Wrapper that augments BatchedFTMOEnv with DQN direction signal.

    Designed for the existing repo's batched, tensor-native env: the wrapper
    is NOT a Gym `gym.Wrapper` subclass because the env is custom. It
    duck-types ``reset()``, ``step(actions)``, ``current_mask_and_force()``,
    and exposes ``state_dim``.

    REVERT: delete with the rest of core/dist_teacher/.
    """

    def __init__(
        self,
        env,
        teacher,
        dist_phase_manager,
        confidence_threshold: float = 0.55,
        masking_enabled: bool = True,
    ):
        self.env = env
        self.teacher = teacher
        self.dist_phase_manager = dist_phase_manager
        self.confidence_threshold = float(confidence_threshold)
        self.masking_enabled = bool(masking_enabled)

        self.device = getattr(env, "device", torch.device("cpu"))
        self.B = int(getattr(env, "B", 1))
        # Obs is constant-shape across all dist phases: base + 3 DQN slots.
        self.base_state_dim = int(env.state_dim)
        self.state_dim = self.base_state_dim + 3

        # Track previous-step position to detect "entry steps" cleanly.
        # +ve = long, -ve = short, 0 = flat. Matches BatchedFTMOEnv._position.
        self._prev_position = torch.zeros(self.B, device=self.device)

        # Retirement freeze value is materialized lazily when the manager
        # reports the teacher has retired.
        self._retirement_freeze: Optional[torch.Tensor] = None

        # Diagnostics counters reset each episode/day by callers.
        self.daily_dist_bonus = 0.0
        self.daily_entry_steps = 0
        self.daily_agreement_hits = 0
        self.daily_confidence_sum = 0.0
        self.daily_confidence_count = 0

    # ── pass-through plumbing ───────────────────────────────────────────
    def __getattr__(self, name):
        # Defer anything we don't override to the wrapped env.
        return getattr(self.env, name)

    def reset(self):
        base_state = self.env.reset()
        self._prev_position = self._current_position().clone()
        return self._augment_obs(base_state)

    # ── core step ───────────────────────────────────────────────────────
    def step(self, actions: Dict[str, torch.Tensor]):
        # Snapshot pre-step position to detect entries on THIS step.
        prev_pos = self._prev_position
        # Snapshot mask BEFORE step so we know what was/wasn't allowed.
        try:
            pre_mask, _must = self.env.current_mask_and_force()
        except Exception:
            pre_mask = None

        base_state, base_reward, done, info = self.env.step(actions)
        base_state = base_state if isinstance(
            base_state, torch.Tensor
        ) else torch.as_tensor(base_state, device=self.device)

        # Augment observation with DQN slots (or retirement freeze value).
        augmented_state = self._augment_obs(base_state)

        # Compute distillation bonus (entry-step only).
        bonus, dist_info = self._compute_bonus(
            base_state=base_state,
            actions=actions,
            prev_pos=prev_pos,
            pre_mask=pre_mask,
        )

        # Accumulate diagnostics (reset by training loop at day-end).
        if dist_info["entry_step_count"]:
            self.daily_entry_steps += int(dist_info["entry_step_count"])
            self.daily_agreement_hits += int(dist_info["agreement_count"])
            self.daily_confidence_sum += float(dist_info["confidence_sum"])
            self.daily_confidence_count += int(dist_info["confidence_count"])
        self.daily_dist_bonus += float(bonus.sum().item())

        total_reward = base_reward + bonus

        # Surface dist diagnostics in info (never mutating base keys).
        info = dict(info) if isinstance(info, dict) else {"_env_info": info}
        info["dist_bonus"] = bonus.detach()
        info["dist_weight"] = float(
            self.dist_phase_manager.get_distillation_weight()
        )
        info["dist_entry_step_mask"] = dist_info["entry_step_mask"]
        info["dqn_active"] = bool(self.dist_phase_manager.is_teacher_active())

        # Refresh prev-position snapshot for next call.
        self._prev_position = self._current_position().clone()

        return augmented_state, total_reward, done, info

    # ── helpers ─────────────────────────────────────────────────────────
    def _current_position(self) -> torch.Tensor:
        # BatchedFTMOEnv stores per-episode lots-signed in self._position.
        pos = getattr(self.env, "_position", None)
        if pos is None:
            return torch.zeros(self.B, device=self.device)
        return pos.detach()

    def _augment_obs(self, base_state: torch.Tensor) -> torch.Tensor:
        """Append 3 DQN slots — real probs when teacher active, frozen mean otherwise."""
        if base_state.dim() == 1:
            base_state = base_state.unsqueeze(0)

        if self.dist_phase_manager.is_teacher_active():
            probs = self.teacher.predict_probs_batch(base_state)
        else:
            if self._retirement_freeze is None:
                self._retirement_freeze = torch.as_tensor(
                    self.teacher.get_retirement_freeze_value(),
                    dtype=torch.float32,
                    device=self.device,
                )
            probs = self._retirement_freeze.unsqueeze(0).expand(
                base_state.shape[0], 3
            )
        return torch.cat([base_state, probs], dim=-1)

    def _is_entry_step(
        self, direction: torch.Tensor, prev_pos: torch.Tensor
    ) -> torch.Tensor:
        """True for batch rows where this step opens or flips a position.

        Logic:
          - was flat (prev_pos == 0) AND chose BUY or SELL → entry
          - was long  (prev_pos  > 0) AND chose SELL       → flip-entry
          - was short (prev_pos  < 0) AND chose BUY        → flip-entry
          - everything else (hold, exit only, stay flat)   → not entry
        """
        was_flat = prev_pos == 0
        was_long = prev_pos > 0
        was_short = prev_pos < 0
        open_new = was_flat & ((direction == BUY) | (direction == SELL))
        flip = (was_long & (direction == SELL)) | (was_short & (direction == BUY))
        return open_new | flip

    def _compute_bonus(
        self,
        base_state: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        prev_pos: torch.Tensor,
        pre_mask: Optional[torch.Tensor],
    ):
        """Return (bonus_tensor, dist_diagnostics_dict)."""
        B = base_state.shape[0]
        zero = torch.zeros(B, device=base_state.device, dtype=base_state.dtype)
        diag = {
            "entry_step_count": 0,
            "agreement_count": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "entry_step_mask": torch.zeros(B, dtype=torch.bool, device=base_state.device),
        }

        dist_weight = float(self.dist_phase_manager.get_distillation_weight())
        if dist_weight <= 0.0:
            return zero, diag
        if not self.dist_phase_manager.is_teacher_active():
            return zero, diag

        direction = actions["direction"].to(base_state.device).long()
        entry_mask = self._is_entry_step(direction, prev_pos.to(direction.device))
        diag["entry_step_mask"] = entry_mask
        if not bool(entry_mask.any()):
            return zero, diag
        diag["entry_step_count"] = int(entry_mask.sum().item())

        # DQN inference (one pass over the full batch — vectorized).
        probs = self.teacher.predict_probs_batch(base_state)  # (B, 3) [BUY,SELL,HOLD]
        top_col = probs.argmax(dim=-1)                        # (B,)
        top_conf = probs.gather(-1, top_col.unsqueeze(-1)).squeeze(-1)
        confident = top_conf >= self.confidence_threshold

        # Map DQN top column → DIRECTION_DIM code (BUY/SELL/HOLD→FLAT).
        # We only ever award bonus for BUY/SELL agreement (HOLD/FLAT is
        # not an entry by definition).
        dqn_dir = torch.full_like(direction, FLAT)
        dqn_dir = torch.where(top_col == _DQN_BUY_COL,
                              torch.full_like(direction, BUY), dqn_dir)
        dqn_dir = torch.where(top_col == _DQN_SELL_COL,
                              torch.full_like(direction, SELL), dqn_dir)
        agreement = (direction == dqn_dir) & (dqn_dir != FLAT)

        # Mask-aware: never reward agreement on a masked direction.
        if self.masking_enabled and pre_mask is not None:
            # pre_mask shape (B, DIRECTION_DIM) — 1.0 allowed, 0.0 masked.
            allowed = pre_mask.to(direction.device).gather(
                -1, dqn_dir.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            agreement = agreement & (allowed > 0.5)

        # Diagnostics: count confident DQN signals on entry steps.
        confident_entry = entry_mask & confident
        diag["agreement_count"] = int((entry_mask & agreement & confident).sum().item())
        diag["confidence_sum"] = float(
            torch.where(confident_entry, top_conf, torch.zeros_like(top_conf)).sum().item()
        )
        diag["confidence_count"] = int(confident_entry.sum().item())

        # Final bonus: weight * confidence, only on confident, agreeing, entry,
        # unmasked rows.
        award = entry_mask & agreement & confident
        bonus = torch.where(
            award,
            dist_weight * top_conf,
            torch.zeros_like(top_conf),
        ).to(base_state.dtype)
        return bonus, diag

    # ── lifecycle hooks for the training loop ──────────────────────────
    def reset_daily_diagnostics(self) -> Dict[str, Any]:
        """Return + reset accumulated dist diagnostics for this day."""
        snapshot = {
            "dist_bonus_sum": float(self.daily_dist_bonus),
            "entry_step_count": int(self.daily_entry_steps),
            "agreement_count": int(self.daily_agreement_hits),
            "avg_dqn_confidence": (
                self.daily_confidence_sum / max(1, self.daily_confidence_count)
            ),
        }
        self.daily_dist_bonus = 0.0
        self.daily_entry_steps = 0
        self.daily_agreement_hits = 0
        self.daily_confidence_sum = 0.0
        self.daily_confidence_count = 0
        return snapshot
