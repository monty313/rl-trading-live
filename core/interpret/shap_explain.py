"""
core/interpret/shap_explain.py
────────────────────────────────────────────────────────────────────────────
POST-HOC SHAP explainer for the trained ActorCritic (PART 2). NEVER imported by
the training loop — `shap` is an OPTIONAL dependency, import-guarded here, so a
training/CI environment without it keeps working. The interpretability Colab cell
calls this only when RUN_SHAP is on.

WHY GradientExplainer (not KernelExplainer): the policy is a small PyTorch MLP,
so the PyTorch-native GradientExplainer is FAST (expected-gradients over a
background set) where KernelExplainer would need thousands of model evaluations.

THREE HEAD WRAPPERS: shap explains a single tensor output, but the ActorCritic
returns 4 tensors. We wrap each head in a thin nn.Module that runs the real net
and exposes ONLY that head's output, matching the ActorCritic's actual attributes
(core/agent/ppo.py):
    DirectionHeadWrapper -> dir_head logits  (N, 3)
    ExitHeadWrapper      -> exit_head logits (N, 3)
    LotHeadWrapper       -> lot_mean         (N, 1)
One GradientExplainer is built per wrapper.

BACKGROUND + EXPLAIN sets: 200-500 background observations (sampled from a saved
rollout/env-steps, NOT the whole dataset) and <=500 explained observations, both
config-driven (SHAP_BACKGROUND_SAMPLES / SHAP_EXPLAIN_SAMPLES).

CACHE: SHAP values are saved as a .npz in the metrics dir keyed by the checkpoint
file's content hash, so a re-run with the same checkpoint skips the (slow)
recompute. Works on GPU and CPU.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from core.interpret.dashboard_utils import obs_feature_names
from core.agent.action_space import DIRECTION_NAMES, EXIT_NAMES, BUY, SELL, FLAT


# ── OPTIONAL DEPENDENCY GUARD ────────────────────────────────────────────────
def shap_available() -> bool:
    """True iff the optional `shap` package can be imported. The Colab cell + the
    tests use this to degrade gracefully with a clear message instead of crashing."""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PER-HEAD nn.Module WRAPPERS (single-output, for shap.GradientExplainer)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class DirectionHeadWrapper(nn.Module):
    """Expose ONLY the direction logits (N, DIRECTION_DIM) of the wrapped net."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        dir_logits, _exit, _lot, _v = self.net(x)
        return dir_logits


class ExitHeadWrapper(nn.Module):
    """Expose ONLY the exit logits (N, EXIT_DIM)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        _dir, exit_logits, _lot, _v = self.net(x)
        return exit_logits


class LotHeadWrapper(nn.Module):
    """Expose ONLY the continuous lot mean (N, 1)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        _dir, _exit, lot_mean, _v = self.net(x)
        return lot_mean


_WRAPPERS = {
    "direction": DirectionHeadWrapper,
    "exit": ExitHeadWrapper,
    "lot": LotHeadWrapper,
}


def file_hash(path: str) -> str:
    """Short md5 of a checkpoint file's bytes — the SHAP cache key. A different
    checkpoint => a different key => a recompute; the same checkpoint hits cache."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_path(metrics_dir: str, ckpt_hash: str) -> str:
    return os.path.join(metrics_dir, f"shap_values_{ckpt_hash}.npz")


def explain_heads(net, background: torch.Tensor, explain: torch.Tensor,
                  feature_names: Optional[List[str]] = None,
                  heads: tuple = ("direction", "exit", "lot")) -> Dict[str, dict]:
    """Run shap.GradientExplainer for each head and return per-head SHAP values +
    mean-|SHAP| feature importance.

    background : (B, state_dim) reference set (200-500 obs).
    explain    : (M, state_dim) obs to explain (<=500).
    Returns {head: {"shap_values": np.ndarray, "importances": np.ndarray[F],
                    "feature_names": [...]}}. Raises RuntimeError if shap missing
    (callers guard with shap_available())."""
    if not shap_available():
        raise RuntimeError("shap not installed — guard with shap_available()")
    import shap

    net.eval()
    device = next(net.parameters()).device
    bg = background.detach().to(device).float()
    ex = explain.detach().to(device).float()
    state_dim = ex.shape[-1]
    if feature_names is None:
        feature_names = [f"feat{j}" for j in range(state_dim)]

    out: Dict[str, dict] = {}
    for head in heads:
        wrapper = _WRAPPERS[head](net).to(device)
        explainer = shap.GradientExplainer(wrapper, bg)
        sv = explainer.shap_values(ex)
        # shap returns a list (one array per output class) for multi-output heads,
        # or a single array for the 1-output lot head. Stack to a common shape and
        # average |value| over samples (and classes) for a per-feature importance.
        if isinstance(sv, list):
            arr = np.stack([np.asarray(s) for s in sv], axis=0)   # (C, M, F)
            importance = np.abs(arr).mean(axis=(0, 1))            # (F,)
        else:
            arr = np.asarray(sv)
            importance = np.abs(arr).reshape(-1, state_dim).mean(axis=0)
        out[head] = {
            "shap_values": arr,
            "importances": importance,
            "feature_names": list(feature_names),
        }
    return out


def explain_single_decision(head_result: dict, head: str, row: int = 0,
                            top_k: int = 6) -> Dict[str, object]:
    """Turn one explained observation's SHAP values into a RANKED, human-readable
    set of feature contributions toward each class/output (PART 2 single-decision
    explain). For direction: contributions toward BUY/SELL/FLAT; for lot: toward
    a bigger/smaller size. Returns {"summary": str, "contributions": {...}}."""
    sv = head_result["shap_values"]
    names = head_result["feature_names"]
    contributions: Dict[str, list] = {}
    if head == "direction" and sv.ndim == 3:
        for cls in range(sv.shape[0]):
            vals = sv[cls, row]
            order = np.argsort(np.abs(vals))[::-1][:top_k]
            label = DIRECTION_NAMES.get(cls, str(cls))
            contributions[label] = [(names[i], float(vals[i])) for i in order]
    elif head == "exit" and sv.ndim == 3:
        for cls in range(sv.shape[0]):
            vals = sv[cls, row]
            order = np.argsort(np.abs(vals))[::-1][:top_k]
            label = EXIT_NAMES.get(cls, str(cls))
            contributions[label] = [(names[i], float(vals[i])) for i in order]
    else:                                          # lot (scalar) or single-output
        vals = np.asarray(sv).reshape(-1, len(names))[row]
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        contributions["lot_size"] = [(names[i], float(vals[i])) for i in order]

    # Build a one-line plain-English summary from the strongest contributions.
    bits: List[str] = []
    for label, feats in contributions.items():
        if not feats:
            continue
        fn, fv = feats[0]
        arrow = "+" if fv >= 0 else "-"
        bits.append(f"{fn} {arrow}{abs(fv):.3f} toward {label}")
    summary = "; ".join(bits) if bits else "(no contributions)"
    return {"summary": summary, "contributions": contributions}


def run_shap(checkpoint_path: str, background: torch.Tensor, explain: torch.Tensor,
             cfg: dict, metrics_dir: str, device: Optional[torch.device] = None,
             indicator_columns: Optional[List[str]] = None,
             use_cache: bool = True) -> Dict[str, dict]:
    """Top-level SHAP entry point used by the Colab cell. Loads the checkpoint into
    a fresh ActorCritic, checks the .npz cache keyed by the checkpoint hash, runs
    explain_heads() on a miss, caches the importances, and returns the per-head
    result dict (importances + feature names; cached path skips the slow recompute).
    Caller MUST guard with shap_available()."""
    from core.agent.ppo import PPOAgent
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = int(explain.shape[-1])
    cfg = dict(cfg)
    cfg["STATE_DIM"] = state_dim
    lkbk = int(cfg.get("LOOKBACK", 20))
    n_ind = max(1, (state_dim - 20) // max(lkbk, 1))
    names = obs_feature_names(lkbk, n_ind, indicator_columns)
    if len(names) != state_dim:
        names = None

    os.makedirs(metrics_dir, exist_ok=True)
    ckpt_hash = file_hash(checkpoint_path)
    cache = _cache_path(metrics_dir, ckpt_hash)
    if use_cache and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        fnames = list(data["feature_names"]) if "feature_names" in data else names
        return {h: {"importances": data[f"{h}_imp"], "feature_names": fnames,
                    "cached": True}
                for h in ("direction", "exit", "lot") if f"{h}_imp" in data}

    agent = PPOAgent(state_dim, cfg, device)
    agent.load(checkpoint_path, partial=True)
    result = explain_heads(agent.net, background, explain, feature_names=names)
    # Cache the importances + names (not the full SHAP tensors — keep it small).
    save_kw = {f"{h}_imp": result[h]["importances"] for h in result}
    save_kw["feature_names"] = np.array(result[next(iter(result))]["feature_names"],
                                        dtype=object)
    np.savez(cache, **save_kw)
    for h in result:
        result[h]["cached"] = False
    return result


def save_summary_plots(net, background: torch.Tensor, explain: torch.Tensor,
                       metrics_dir: str, feature_names: Optional[List[str]] = None,
                       prefix: str = "shap") -> List[str]:
    """Save beeswarm/bar summary PNGs per head (PART 2 batch summary plots). Best
    effort — returns the paths written; skips silently if shap/matplotlib missing."""
    if not shap_available():
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except Exception:                                            # pragma: no cover
        return []
    net.eval()
    device = next(net.parameters()).device
    bg = background.detach().to(device).float()
    ex = explain.detach().to(device).float()
    state_dim = ex.shape[-1]
    if feature_names is None:
        feature_names = [f"feat{j}" for j in range(state_dim)]
    os.makedirs(metrics_dir, exist_ok=True)
    paths: List[str] = []
    ex_np = ex.detach().cpu().numpy()
    for head in ("direction", "exit", "lot"):
        wrapper = _WRAPPERS[head](net).to(device)
        sv = shap.GradientExplainer(wrapper, bg).shap_values(ex)
        sv_plot = sv[0] if isinstance(sv, list) else sv
        try:
            plt.figure()
            shap.summary_plot(sv_plot, ex_np, feature_names=feature_names,
                              show=False, plot_type="bar", max_display=15)
            out = os.path.join(metrics_dir, f"{prefix}_{head}_bar.png")
            plt.tight_layout()
            plt.savefig(out, dpi=110)
            plt.close()
            paths.append(out)
        except Exception:                                        # pragma: no cover
            plt.close()
    return paths
