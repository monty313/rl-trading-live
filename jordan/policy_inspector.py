"""
jordan/policy_inspector.py
────────────────────────────────────────────────────────────────────────────
Loads a PPO policy checkpoint (READ-ONLY) and explains a decision: the chosen
direction + exit + lot, the policy's direction probabilities, and a simple
input-gradient feature importance. Used by the dashboard ("why this trade?").

Jordan never modifies the checkpoint or trades (HARD RULE 6).
"""
from __future__ import annotations

import torch

from core.agent.action_space import (DIRECTION_NAMES, EXIT_NAMES, map_lot)


class PolicyInspector:
    def __init__(self, agent=None):
        self.agent = agent

    def load_policy(self, path: str, agent_factory=None):
        """Load a PPO .pt into an agent (read-only). agent_factory() -> PPOAgent."""
        if self.agent is None and agent_factory is not None:
            self.agent = agent_factory()
        if self.agent is not None:
            self.agent.load(path, partial=True)
        return self.agent

    def inspect_action(self, obs_tensor: torch.Tensor, max_lot: float = 2.0) -> dict:
        """Return the PPO decision breakdown for one observation."""
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs = obs_tensor.clone().requires_grad_(True)
        dir_logits, exit_logits, lot_mean, value = self.agent.net(obs)
        dir_probs = torch.softmax(dir_logits, dim=-1).reshape(-1)
        exit_probs = torch.softmax(exit_logits, dim=-1).reshape(-1)
        direction = int(dir_probs.argmax().item())
        exit_act = int(exit_probs.argmax().item())
        lot_raw = float(torch.sigmoid(lot_mean.squeeze()).item())

        # input-gradient importance for the chosen direction logit
        self.agent.net.zero_grad(set_to_none=True)
        dir_logits[0, direction].backward()
        grads = obs.grad.detach().abs().reshape(-1)
        kk = min(3, grads.shape[0])
        gv, gi = torch.topk(grads, k=kk)
        top3 = [(f"feat_{int(i)}", float(v)) for v, i in zip(gv.tolist(), gi.tolist())]

        return {
            "direction": DIRECTION_NAMES.get(direction, "FLAT"),
            "lot": map_lot(lot_raw, max_lot),
            "exit": EXIT_NAMES.get(exit_act, "HOLD"),
            "value": float(value.reshape(-1)[0].item()),
            "direction_probs": {DIRECTION_NAMES[i]: round(float(p), 4)
                                for i, p in enumerate(dir_probs.tolist())},
            "top3_features": top3,
        }
