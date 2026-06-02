"""
jordan/policy_inspector.py
────────────────────────────────────────────────────────────────────────────
Loads a policy checkpoint (READ-ONLY) and explains a decision: decoded action,
top-5 Q-values, and a simple input-gradient feature importance. Used by the
dashboard when the user asks "why did we take that trade?".

Jordan never modifies the checkpoint or trades (HARD RULE 6).
"""
from __future__ import annotations

import torch

from core.agent.action_space import (decode, get_lot, get_sl_pips, get_tp_pips,
                                      DIRECTION_NAMES)


class PolicyInspector:
    def __init__(self, agent=None):
        self.agent = agent

    def load_policy(self, path: str, agent_factory=None):
        """Load a .pt into an agent (read-only). agent_factory() -> DQNAgent."""
        if self.agent is None and agent_factory is not None:
            self.agent = agent_factory()
        if self.agent is not None:
            self.agent.load(path, partial=True)
        return self.agent

    def inspect_action(self, obs_tensor: torch.Tensor, max_lot: float = 2.0) -> dict:
        """Return the decision breakdown for one observation."""
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs = obs_tensor.clone().requires_grad_(True)
        q = self.agent.policy_net(obs)
        best = int(q.argmax(dim=1).item())
        # top-5 Q-values
        topv, topi = torch.topk(q.reshape(-1), k=min(5, q.shape[-1]))
        q_top5 = [(int(i), float(v)) for v, i in zip(topv.tolist(), topi.tolist())]
        # input-gradient feature importance for the chosen action
        self.agent.policy_net.zero_grad(set_to_none=True)
        q[0, best].backward()
        grads = obs.grad.detach().abs().reshape(-1)
        kk = min(3, grads.shape[0])
        gv, gi = torch.topk(grads, k=kk)
        top3 = [(f"feat_{int(i)}", float(v)) for v, i in zip(gv.tolist(), gi.tolist())]
        direction, lot_idx, sl_idx, tp_idx = decode(best)
        return {
            "action_int": best,
            "direction": DIRECTION_NAMES[direction],
            "lot": get_lot(lot_idx, max_lot),
            "sl_pips": get_sl_pips(sl_idx),
            "tp_pips": get_tp_pips(tp_idx),
            "q_values_top5": q_top5,
            "top3_features": top3,
        }
