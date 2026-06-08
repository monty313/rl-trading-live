"""
core/agent/dqn_warm_start.py
──────────────────────────────────────────────────────────────────────────────
Best-effort warm-start: copy the DQN checkpoint's shared trunk weights into a
fresh PPO actor-critic. The DQN's action head (7-way) does NOT transfer — PPO
has its own (direction × lot × exit) heads. We only transfer the layers whose
shapes are byte-identical.

Requires MULTI_TF_OBS=True so PPO's state_dim == DQN's input_dim == 2166.

Transfer policy:
  - Walk the DQN state_dict in order, find every Linear layer's (W, b).
  - Walk PPO's trunk Linear layers in order.
  - Copy DQN trunk[i] -> PPO trunk[i] if and only if shapes match exactly.
  - Heads (direction/exit/lot_mean/value_head) are NEVER copied.

This mirrors the predecessor's load_partial() transfer pattern. Logged loudly
so the user sees exactly what transferred.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# Reuse the same state-dict locator the dist teacher uses — it knows the
# common checkpoint layouts.
def _find_dqn_state_dict(ckpt) -> dict:
    """Locate the policy weights inside a loaded checkpoint object."""
    if not isinstance(ckpt, dict):
        if hasattr(ckpt, "state_dict"):
            return ckpt.state_dict()
        raise RuntimeError("[WARM-START] Cannot extract state_dict from checkpoint object")
    for key in ("policy_state_dict", "model_state_dict", "state_dict",
                "online_net", "q_net", "policy", "net"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    raise RuntimeError("[WARM-START] Could not locate DQN policy weights")


def _ordered_linear_pairs(state_dict: dict) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Return [(layer_key_prefix, weight, bias)] in checkpoint order, Linear only."""
    pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    seen_prefixes = set()
    for k, v in state_dict.items():
        if not k.endswith(".weight") or not hasattr(v, "shape") or v.dim() != 2:
            continue
        prefix = k.rsplit(".", 1)[0]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        bias_key = f"{prefix}.bias"
        bias = state_dict.get(bias_key)
        pairs.append((prefix, v, bias))
    return pairs


def warm_start_ppo_from_dqn(
    ppo_net: nn.Module,
    dqn_checkpoint_path: str,
    device,
) -> Dict[str, str]:
    """Copy DQN trunk weights into a PPO ActorCritic. Returns transfer report.

    Args:
        ppo_net: instance of core.agent.ppo.ActorCritic with state_dim == 2166.
        dqn_checkpoint_path: filesystem path to the .pt file.
        device: torch device (only used for the loaded checkpoint placement).

    Returns:
        {"transferred": "...", "skipped": "...", "ppo_layer_count": "N"}.

    Raises:
        FileNotFoundError if the checkpoint is missing.
        AssertionError if PPO's first Linear input != DQN's first Linear input
        (this means you forgot to set MULTI_TF_OBS=True and the warm-start
        would be silently wrong — fail loudly).
    """
    if not os.path.exists(dqn_checkpoint_path):
        raise FileNotFoundError(
            f"[WARM-START] DQN checkpoint not found at {dqn_checkpoint_path}"
        )
    print(f"[WARM-START] Loading DQN checkpoint: {os.path.basename(dqn_checkpoint_path)}")
    ckpt = torch.load(dqn_checkpoint_path, map_location="cpu", weights_only=False)
    sd = _find_dqn_state_dict(ckpt)
    dqn_pairs = _ordered_linear_pairs(sd)
    if not dqn_pairs:
        raise RuntimeError("[WARM-START] No Linear layers found in DQN checkpoint")

    # Walk PPO trunk in module order.
    ppo_trunk_linears: List[Tuple[str, nn.Linear]] = []
    if hasattr(ppo_net, "trunk"):
        for name, m in ppo_net.trunk.named_modules():
            if isinstance(m, nn.Linear):
                ppo_trunk_linears.append((f"trunk.{name}", m))

    if not ppo_trunk_linears:
        raise RuntimeError("[WARM-START] PPO ActorCritic has no trunk Linear layers")

    transferred: List[str] = []
    skipped: List[str] = []

    # Sanity: PPO's first Linear input must be DQN's input dim OR DQN's input
    # dim + N_DIST_SLOTS (the distillation wrapper appends 3 DQN-prob slots to
    # every obs). Anything else means the user forgot MULTI_TF_OBS=True and
    # would get a silently-wrong warm-start.
    N_DIST_SLOTS = 3   # see core/dist_teacher/dist_prephase_wrapper.py
    ppo_in = ppo_trunk_linears[0][1].in_features
    dqn_in = int(dqn_pairs[0][1].shape[1])
    if ppo_in == dqn_in:
        dist_pad = 0
    elif ppo_in == dqn_in + N_DIST_SLOTS:
        dist_pad = N_DIST_SLOTS
        print(
            f"[WARM-START] Detected dist wrapper: PPO obs has {N_DIST_SLOTS} extra "
            f"slots ({dqn_in} DQN-era + {N_DIST_SLOTS} DQN-probability slots). "
            f"DQN weights will be copied into the first {dqn_in} input columns; "
            f"the {N_DIST_SLOTS} dist slots will be zero-initialized so they "
            f"contribute nothing at step 0 (PPO learns them from scratch)."
        )
    else:
        raise AssertionError(
            f"[WARM-START] PPO state_dim ({ppo_in}) != DQN input_dim ({dqn_in}) "
            f"and != {dqn_in}+{N_DIST_SLOTS}. Set MULTI_TF_OBS=True so PPO observes "
            f"the same 2166-dim schema as the DQN, then retry."
        )

    # Pair by ordinal position. Stop at the first shape mismatch — DQN's last
    # layer is the 7-way action head; PPO's last trunk layer is hidden→hidden.
    # They will not match. That's the correct stopping point.
    for i, (ppo_name, ppo_linear) in enumerate(ppo_trunk_linears):
        if i >= len(dqn_pairs):
            skipped.append(f"{ppo_name} (no DQN counterpart)")
            continue
        dqn_prefix, W, b = dqn_pairs[i]
        ppo_W = ppo_linear.weight.data
        ppo_b = ppo_linear.bias.data if ppo_linear.bias is not None else None
        # Special case: first PPO layer may be wider by exactly N_DIST_SLOTS so
        # PPO can also consume the 3 DQN-prob features appended by the dist
        # wrapper. Copy DQN weights into the first dqn_in columns, zero-init
        # the trailing dist slots.
        if (
            i == 0
            and dist_pad > 0
            and W.shape[0] == ppo_W.shape[0]
            and ppo_W.shape[1] == W.shape[1] + dist_pad
        ):
            with torch.no_grad():
                W_dev = W.to(ppo_W.device, dtype=ppo_W.dtype)
                ppo_W[:, : W.shape[1]].copy_(W_dev)
                ppo_W[:, W.shape[1] :].zero_()    # dist slots: neutral start
                if b is not None and ppo_b is not None and b.shape == ppo_b.shape:
                    ppo_b.copy_(b.to(ppo_b.device, dtype=ppo_b.dtype))
            transferred.append(
                f"{ppo_name} <- {dqn_prefix} shape={tuple(W.shape)} "
                f"(+{dist_pad} dist slots zero-initialized)"
            )
            continue
        if W.shape != ppo_W.shape:
            skipped.append(
                f"{ppo_name} <- {dqn_prefix} (shape mismatch: DQN {tuple(W.shape)} vs PPO {tuple(ppo_W.shape)})"
            )
            continue
        with torch.no_grad():
            ppo_W.copy_(W.to(ppo_W.device, dtype=ppo_W.dtype))
            if b is not None and ppo_b is not None and b.shape == ppo_b.shape:
                ppo_b.copy_(b.to(ppo_b.device, dtype=ppo_b.dtype))
        transferred.append(f"{ppo_name} <- {dqn_prefix} shape={tuple(W.shape)}")

    # Heads are intentionally NOT transferred — they have different output dims
    # (DQN: 7-way Q; PPO: direction/exit/lot/value). Re-initialized at PPO build.
    print("[WARM-START] Transferred trunk layers:")
    for line in transferred:
        print(f"             ✓ {line}")
    if skipped:
        print("[WARM-START] Skipped (heads or shape mismatch — re-initialized):")
        for line in skipped:
            print(f"             · {line}")
    print(f"[WARM-START] Done. {len(transferred)} layer(s) transferred, "
          f"{len(skipped)} skipped/reinit.")
    return {
        "transferred": "; ".join(transferred),
        "skipped":     "; ".join(skipped),
        "ppo_layer_count": str(len(ppo_trunk_linears)),
    }
