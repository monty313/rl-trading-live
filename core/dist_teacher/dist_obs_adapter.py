# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
# Slim observation adapter that aligns the current PPO base observation
# to the DQN teacher's input dimension. Built only when the checkpoint
# probe (Section 2 of spec) reports a CLOSE mismatch (≤20 features).
# A LARGE mismatch (>20) means the DQN was trained on a fundamentally
# different observation and cannot be safely bridged — STOP in that case.
#
# Strategy:
#   - Core price/indicator features sit at the FRONT of base_obs and are
#     the most stable across OBS_SCHEMA_VERSION bumps.
#   - The newer FTMO + session blocks are APPENDED at the END.
#   - We therefore slice the first dqn_input_dim features from base_obs.
# This is documented and tested. Logged loudly at startup.
# ═══════════════════════════════════════════════════════
from __future__ import annotations

from typing import Optional

import torch


class DistObsAdapter:
    """[DIST] Slice/pad base PPO obs to match DQN teacher input dim.

    This object is INERT when ``ppo_obs_dim == dqn_input_dim``: the
    constructor will refuse to build a degenerate adapter (use ``None``
    instead). When dimensions differ it slices the first ``dqn_input_dim``
    features of base_obs and feeds those to the DQN. The added DQN obs
    slots are still appended to the FULL ``base_obs`` by the wrapper —
    only the DQN's forward pass sees the sliced view.

    REVERT: delete this file with the rest of core/dist_teacher/.
    """

    def __init__(self, ppo_obs_dim: int, dqn_input_dim: int):
        if dqn_input_dim > ppo_obs_dim:
            raise ValueError(
                f"[DIST] DQN input ({dqn_input_dim}) larger than PPO base "
                f"obs ({ppo_obs_dim}) — cannot slice; manual review required."
            )
        if dqn_input_dim == ppo_obs_dim:
            raise ValueError(
                "[DIST] Adapter not needed when dims match. Pass "
                "obs_adapter=None to DistDQNTeacher instead."
            )
        self.ppo_obs_dim = int(ppo_obs_dim)
        self.dqn_input_dim = int(dqn_input_dim)
        print(
            f"[DIST] OBS ADAPTER ACTIVE: PPO_dim={self.ppo_obs_dim} → "
            f"DQN_dim={self.dqn_input_dim}"
        )
        print(
            f"[DIST] Slicing first {self.dqn_input_dim} features from base obs"
        )
        print(
            "[DIST] Assumption: stable base features precede newer FTMO/session blocks"
        )

    def adapt(self, base_obs_batch: torch.Tensor) -> torch.Tensor:
        """Slice the trailing feature dim down to ``dqn_input_dim``."""
        return base_obs_batch[..., : self.dqn_input_dim]

    def __repr__(self) -> str:
        return (
            f"DistObsAdapter(ppo={self.ppo_obs_dim}, dqn={self.dqn_input_dim})"
        )


def build_adapter_if_needed(
    ppo_obs_dim: int, dqn_input_dim: int
) -> Optional[DistObsAdapter]:
    """Convenience: return ``None`` when dims match, adapter otherwise."""
    if ppo_obs_dim == dqn_input_dim:
        return None
    return DistObsAdapter(ppo_obs_dim, dqn_input_dim)
