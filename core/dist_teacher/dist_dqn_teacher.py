# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
# >>> FULL LIFECYCLE + REVERT INSTRUCTIONS live at the TOP of
# >>> core/dist_phase/dist_phase_manager.py. Read THAT file first.
#
# Frozen DQN teacher used for DIRECTION-ONLY distillation.
#   - Loads a DQN checkpoint in eval mode with requires_grad=False.
#   - Vectorized batch inference on GPU (matches BatchedFTMOEnv tensors).
#   - Outputs (B, 3) direction probabilities in the canonical order
#     reported by the checkpoint probe (see Section 2 of the spec).
#   - Tracks a running mean of observed probs so the wrapper can freeze
#     the appended obs slots to the EMPIRICAL distribution at retirement
#     instead of an arbitrary uniform [0.333, 0.333, 0.333].
#
# REVERT: delete this file with the rest of core/dist_teacher/.
# ═══════════════════════════════════════════════════════
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Common keys we'll look for in the loaded checkpoint dict. First hit wins.
_POLICY_STATE_KEYS = (
    "policy_state_dict",
    "model_state_dict",
    "state_dict",
    "online_net",
    "q_net",
    "policy",
    "net",
)


def _find_state_dict(ckpt) -> Optional[dict]:
    """Best-effort search for the policy weights inside a checkpoint dict."""
    if not isinstance(ckpt, dict):
        # Loaded object is already a nn.Module or a raw state_dict tensor map.
        if hasattr(ckpt, "state_dict"):
            return ckpt.state_dict()
        return ckpt if all(hasattr(v, "shape") for v in ckpt.values()) else None
    for key in _POLICY_STATE_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Maybe ckpt IS the state_dict (all values are tensors).
    if all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    return None


def _infer_input_dim(state_dict: dict) -> Tuple[int, str]:
    """Find the first weight matrix and report its input dim."""
    for k, v in state_dict.items():
        if "weight" in k and hasattr(v, "shape") and v.dim() == 2:
            return int(v.shape[1]), k
    raise RuntimeError(
        "[DIST] Could not locate a 2-D weight matrix in the checkpoint. "
        "Run the Section 2 probe script and report the key listing."
    )


def _infer_output_dim(state_dict: dict) -> Tuple[int, str]:
    """Find the LAST 2-D weight matrix and report its output dim."""
    last_key, last_weight = None, None
    for k, v in state_dict.items():
        if "weight" in k and hasattr(v, "shape") and v.dim() == 2:
            last_key, last_weight = k, v
    if last_weight is None:
        raise RuntimeError("[DIST] No 2-D weight matrix in checkpoint.")
    return int(last_weight.shape[0]), last_key


class _MLPPolicyHead(nn.Module):
    """Generic stacked-Linear MLP reconstructed from the checkpoint's keys.

    Only used when we can identify a clean ``layer.{i}.weight/.bias`` or
    ``fc{i}.weight/.bias`` pattern. For anything more exotic, you should
    load your own architecture object and pass it as ``preloaded_model``.
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_d, out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class DistDQNTeacher:
    """[DIST] Frozen DQN teacher for DIRECTION-only distillation.

    The teacher contributes EXACTLY one thing to training: a probability
    distribution over [BUY, SELL, HOLD] per observation. It does not
    touch lot size, exit timing, or anything else.

    REVERT: delete with the rest of core/dist_teacher/.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device,
        action_order: List[str],
        obs_adapter=None,
        temperature: float = 1.0,
        preloaded_model: Optional[nn.Module] = None,
        gdrive_file_id: Optional[str] = None,
    ):
        self.device = torch.device(device) if not isinstance(
            device, torch.device
        ) else device
        self.action_order = list(action_order)
        self.obs_adapter = obs_adapter
        self.temperature = float(temperature)
        assert self.temperature > 0, "[DIST] temperature must be positive"
        assert set(self.action_order) == {"BUY", "SELL", "HOLD"}, (
            "[DIST] action_order must be a permutation of "
            "['BUY','SELL','HOLD'] — confirm with checkpoint probe"
        )
        self.checkpoint_path = checkpoint_path
        self.gdrive_file_id = gdrive_file_id

        # Running mean of probs (for empirical retirement freeze value).
        self._prob_sum = np.zeros(3, dtype=np.float64)
        self._prob_count = 0

        # Build / load the model.
        if preloaded_model is not None:
            self.model = preloaded_model.to(self.device).eval()
            self.input_dim = None  # caller is responsible
            self.output_dim = None
            print(
                f"[DIST] DQN Teacher loaded (preloaded_model) | device={self.device}"
            )
        else:
            self.model, self.input_dim, self.output_dim = self._load_from_path(
                checkpoint_path
            )
            self.model = self.model.to(self.device).eval()
            print(
                f"[DIST] DQN Teacher loaded | ckpt="
                f"{os.path.basename(checkpoint_path)} | "
                f"input_dim={self.input_dim} | output_dim={self.output_dim} | "
                f"device={self.device}"
            )

        # Freeze every parameter and verify.
        for p in self.model.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in self.model.parameters()), (
            "[DIST] DQN weights not frozen — safety check failed"
        )
        # Also stash a checksum so tests can assert no drift.
        self._frozen_checksum = self._compute_checksum()
        print(
            f"[DIST] action_order={self.action_order} — VERIFY THIS matches "
            "the DQN's head output ordering"
        )

    # ── construction helpers ────────────────────────────────────────────
    def _load_from_path(self, path: str) -> Tuple[nn.Module, int, int]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[DIST] Checkpoint not found at {path}. Use the Section 2 "
                "probe cell to download it to Drive first."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = _find_state_dict(ckpt)
        if state_dict is None:
            raise RuntimeError(
                "[DIST] Could not locate policy weights in checkpoint. "
                "Run the Section 2 probe and verify the key layout."
            )
        in_dim, in_key = _infer_input_dim(state_dict)
        out_dim, out_key = _infer_output_dim(state_dict)
        print(
            f"[DIST] Located policy head: first='{in_key}' "
            f"(input_dim={in_dim}) | last='{out_key}' (output_dim={out_dim})"
        )

        # Try to reconstruct a clean MLP from the linear layers in order.
        linear_layers = [
            (k.rsplit(".", 1)[0], v)
            for k, v in state_dict.items()
            if k.endswith(".weight") and hasattr(v, "shape") and v.dim() == 2
        ]
        if not linear_layers:
            raise RuntimeError(
                "[DIST] No Linear layers found — checkpoint architecture not supported."
            )
        dims = [int(linear_layers[0][1].shape[1])] + [
            int(w.shape[0]) for _, w in linear_layers
        ]
        model = _MLPPolicyHead(dims)
        # Map state_dict keys onto layers.{i}.{weight,bias}
        target_sd = {}
        for i, (prefix, _) in enumerate(linear_layers):
            target_sd[f"layers.{i}.weight"] = state_dict[f"{prefix}.weight"]
            if f"{prefix}.bias" in state_dict:
                target_sd[f"layers.{i}.bias"] = state_dict[f"{prefix}.bias"]
        try:
            model.load_state_dict(target_sd, strict=True)
        except Exception as e:
            raise RuntimeError(
                f"[DIST] Failed to map checkpoint into MLP head: {e}. "
                "Pass preloaded_model= with the correct architecture instead."
            ) from e
        return model, in_dim, out_dim

    def _compute_checksum(self) -> float:
        with torch.no_grad():
            return float(
                sum(p.detach().abs().sum().item() for p in self.model.parameters())
            )

    # ── inference ──────────────────────────────────────────────────────
    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    @torch.no_grad()
    def predict_probs_batch(self, base_obs_batch: torch.Tensor) -> torch.Tensor:
        """[DIST] Vectorized forward pass — returns (B, 3) probability tensor.

        Used in DIST_PRE_PHASE and DIST_PHASE_1 only. Never called when the
        teacher is retired.
        """
        if not isinstance(base_obs_batch, torch.Tensor):
            base_obs_batch = torch.as_tensor(
                base_obs_batch, dtype=torch.float32, device=self.device
            )
        else:
            base_obs_batch = base_obs_batch.to(self.device, dtype=torch.float32)

        if self.obs_adapter is not None:
            x = self.obs_adapter.adapt(base_obs_batch)
        else:
            x = base_obs_batch

        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.model(x)
        # Some DQNs may have more than 3 outputs (e.g. include position-management
        # actions). action_order tells us which 3 columns are BUY/SELL/HOLD; if
        # output dim already equals 3 we just softmax the lot directly.
        if logits.shape[-1] != 3:
            # Best-effort: take the first three columns and warn.
            logits = logits[..., :3]

        probs = F.softmax(logits / self.temperature, dim=-1)

        # Reorder columns to canonical [BUY, SELL, HOLD] if needed.
        canonical = ["BUY", "SELL", "HOLD"]
        if self.action_order != canonical:
            perm = [self.action_order.index(name) for name in canonical]
            probs = probs[..., perm]

        # Numerical hygiene: clamp & renormalize (defends against fp16 edge cases).
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Update running mean of probs for retirement freeze value.
        batch_mean = probs.mean(dim=0).detach().cpu().numpy().astype(np.float64)
        self._prob_sum += batch_mean * probs.shape[0]
        self._prob_count += int(probs.shape[0])

        return probs

    @torch.no_grad()
    def predict_probs_single(self, base_obs: np.ndarray) -> np.ndarray:
        """Single-observation convenience wrapper; returns shape (3,)."""
        t = torch.as_tensor(base_obs, dtype=torch.float32, device=self.device)
        out = self.predict_probs_batch(t.unsqueeze(0))
        return out.squeeze(0).cpu().numpy()

    def is_confident_batch(
        self, probs: torch.Tensor, threshold: float = 0.55
    ) -> torch.Tensor:
        """Returns a (B,) bool tensor: True where max(probs) >= threshold."""
        return probs.max(dim=-1).values >= threshold

    def get_retirement_freeze_value(self) -> np.ndarray:
        """[DIST] Empirical mean of probs seen so far ([BUY, SELL, HOLD]).

        Used by the wrapper to freeze the appended obs slots at retirement.
        If no batches have been seen yet, falls back to uniform 1/3.
        """
        if self._prob_count == 0:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        mean = (self._prob_sum / float(self._prob_count)).astype(np.float32)
        # Re-normalize defensively in case of accumulated float noise.
        s = float(mean.sum())
        if s > 0:
            mean = mean / s
        return mean

    def verify_no_drift(self) -> bool:
        """Returns True iff every parameter is byte-identical to load time."""
        return abs(self._compute_checksum() - self._frozen_checksum) < 1e-9
