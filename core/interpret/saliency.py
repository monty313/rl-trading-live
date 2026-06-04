"""
core/interpret/saliency.py
────────────────────────────────────────────────────────────────────────────
FAST, ALWAYS-ON gradient saliency for the trained ActorCritic (PART 4).

Given a checkpoint (or a live PPOAgent) + a batch of observations, we take ONE
forward + ONE backward pass per head and read d(head_output)/d(obs). Averaging
|grad| across the batch gives a per-FEATURE importance ranking — "which inputs
does this head's decision move with". This is the cheap counterpart to SHAP:
no background set, no sampling, runs in <5s for ~10k obs on GPU (and on CPU).

The three heads of the ActorCritic (core/agent/ppo.py) are:
    direction -> dir_head logits   (FLAT/BUY/SELL)
    exit      -> exit_head logits  (HOLD/REDUCE/CLOSE)
    lot       -> lot_mean scalar   (continuous size)
For the categorical heads we differentiate the MAX-probability logit (the chosen
class) summed over the batch; for the lot head we differentiate the scalar mean.
This isolates "what drove the decision the policy actually makes".

Outputs: a dict of {head: {"importances": np.ndarray[F], "ranking": [(name, val)],
"feature_names": [...]}} AND (optionally) a horizontal bar-chart PNG per head in
the metrics dir. Feature names come from dashboard_utils.obs_feature_names so the
bars read in plain English (e.g. "cci30@t-0", "dd_budget_remaining").

NOTHING here is imported by the training loop — this is post-hoc tooling.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from core.interpret.dashboard_utils import obs_feature_names


# Heads we explain and how to reduce each head's output to a single scalar to
# backprop from. dir/exit are categorical (take the chosen-class logit); lot is
# already a scalar mean.
_HEADS = ("direction", "exit", "lot")


def _head_scalar(net_out, head: str) -> torch.Tensor:
    """Reduce the ActorCritic forward output to a scalar (summed over batch) for
    the requested head, so .backward() gives d(decision)/d(obs) for every sample.
    net_out is (dir_logits, exit_logits, lot_mean, value)."""
    dir_logits, exit_logits, lot_mean, _value = net_out
    if head == "direction":
        # chosen-class logit per row (the BUY/SELL/FLAT the policy would pick)
        return dir_logits.max(dim=-1).values.sum()
    if head == "exit":
        return exit_logits.max(dim=-1).values.sum()
    if head == "lot":
        return lot_mean.squeeze(-1).sum()
    raise ValueError(f"unknown head {head!r}")


def compute_saliency(net, obs: torch.Tensor,
                     feature_names: Optional[List[str]] = None,
                     heads: tuple = _HEADS,
                     top_k: int = 15) -> Dict[str, dict]:
    """Compute average-|grad| feature importance per head for a batch of obs.

    net   : an ActorCritic (or anything whose forward returns the 4-tuple). It is
            put in eval() mode; its parameters are NOT updated (we only need the
            input gradient).
    obs   : (N, state_dim) float tensor (any device). N up to ~10k runs <5s.
    feature_names : optional list of length state_dim; if omitted, generic
            'feat<j>' names are used (the caller usually passes
            obs_feature_names(...) so bars read in English).
    Returns {head: {"importances": np.ndarray[state_dim],
                    "ranking": [(name, importance), ...]  # top_k, descending,
                    "feature_names": [...]}}.
    """
    net.eval()
    device = next(net.parameters()).device
    x = obs.detach().to(device).float()
    state_dim = x.shape[-1]
    if feature_names is None:
        feature_names = [f"feat{j}" for j in range(state_dim)]
    results: Dict[str, dict] = {}
    for head in heads:
        # Fresh leaf each head so grads don't accumulate across heads.
        xin = x.clone().requires_grad_(True)
        out = net(xin)
        scalar = _head_scalar(out, head)
        grad = torch.autograd.grad(scalar, xin, retain_graph=False,
                                   create_graph=False)[0]          # (N, state_dim)
        importance = grad.abs().mean(dim=0).detach().cpu().numpy()  # (state_dim,)
        order = np.argsort(importance)[::-1]
        ranking = [(feature_names[i], float(importance[i]))
                   for i in order[:top_k]]
        results[head] = {
            "importances": importance,
            "ranking": ranking,
            "feature_names": list(feature_names),
        }
    return results


def saliency_from_checkpoint(checkpoint_path: str, obs: torch.Tensor,
                             cfg: dict, device: Optional[torch.device] = None,
                             indicator_columns: Optional[List[str]] = None,
                             top_k: int = 15) -> Dict[str, dict]:
    """Convenience entry point: load a PPO checkpoint into a fresh ActorCritic
    sized to `obs`, then compute saliency. Used by the Colab interpretability cell
    and the policy report. Returns the same dict shape as compute_saliency()."""
    from core.agent.ppo import PPOAgent
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = int(obs.shape[-1])
    cfg = dict(cfg)
    cfg["STATE_DIM"] = state_dim
    agent = PPOAgent(state_dim, cfg, device)
    agent.load(checkpoint_path, partial=True)
    lkbk = int(cfg.get("LOOKBACK", 20))
    # Infer the indicator-feature count from the obs width and the fixed appended
    # blocks (6 position + 7 ftmo + 7 session = 20) so names line up with the env.
    n_appended = 20
    n_ind = max(1, (state_dim - n_appended) // max(lkbk, 1))
    names = obs_feature_names(lkbk, n_ind, indicator_columns)
    if len(names) != state_dim:                # robustness: fall back to generic
        names = None
    return compute_saliency(agent.net, obs, feature_names=names, top_k=top_k)


def save_saliency_bars(results: Dict[str, dict], metrics_dir: str,
                       prefix: str = "saliency", top_k: int = 15) -> List[str]:
    """Save one horizontal bar-chart PNG per head to metrics_dir; return the paths.
    Degrades gracefully (returns []) if matplotlib is unavailable so the always-on
    path never crashes a headless/CI run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                            # pragma: no cover
        return []
    os.makedirs(metrics_dir, exist_ok=True)
    paths: List[str] = []
    for head, res in results.items():
        ranking = res["ranking"][:top_k]
        if not ranking:
            continue
        names = [n for n, _ in ranking][::-1]
        vals = [v for _, v in ranking][::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
        ax.barh(names, vals, color="#4C9BE8")
        ax.set_title(f"Gradient saliency — {head} head (avg |∂out/∂obs|)")
        ax.set_xlabel("mean |gradient|")
        fig.tight_layout()
        out = os.path.join(metrics_dir, f"{prefix}_{head}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        paths.append(out)
    return paths
