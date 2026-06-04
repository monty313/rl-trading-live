"""
core/hardware.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 11 — hardware-detection module. The SINGLE source of truth for
"what am I running on" so the notebook GPU cell, auto_tune_batch, the profiler
and the Optuna budget all agree.

Classifies the accelerator into a coarse TIER (A100/H100, L4/A10/3090, T4/V100,
small-GPU, or CPU) by reading the same GPU_TIERS table core.settings uses, and
reports the device name, VRAM, CUDA availability and the recommended
(BATCH_SIZE_ENV, ROLLOUT_STEPS) for that card. Pure read-only probing — never
allocates on the device, so it is safe to call before any model is built.
"""
from __future__ import annotations

from typing import Dict

import torch

from core.settings import GPU_TIERS, GPU_UTIL_TARGET


def detect_hardware() -> Dict:
    """Return a dict describing the active compute device.

    Keys: cuda(bool), name(str), vram_gb(float), tier(str),
          batch_size_env(int), rollout_steps(int), util_target(float).
    On CPU, vram_gb is 0.0 and the tier is "CPU" with the small smoke-test sizes."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        for min_vram, batch, rollout, label in GPU_TIERS:
            if vram_gb >= min_vram:
                return {"cuda": True, "name": name, "vram_gb": float(vram_gb),
                        "tier": label, "batch_size_env": int(batch),
                        "rollout_steps": int(rollout),
                        "util_target": float(GPU_UTIL_TARGET)}
    # CPU fallback (dev/CI): the small, fast smoke sizes auto_tune_batch uses.
    return {"cuda": False, "name": "CPU", "vram_gb": 0.0, "tier": "CPU",
            "batch_size_env": 4, "rollout_steps": 64,
            "util_target": float(GPU_UTIL_TARGET)}


def describe_hardware(hw: Dict = None) -> str:
    """One-line human-readable summary (used by the notebook GPU cell)."""
    hw = hw or detect_hardware()
    if hw["cuda"]:
        return (f"GPU: {hw['name']} | VRAM {hw['vram_gb']:.1f}GB | tier "
                f"{hw['tier']} | BATCH_SIZE_ENV={hw['batch_size_env']} "
                f"ROLLOUT_STEPS={hw['rollout_steps']}")
    return ("CPU (no CUDA) | tier CPU | BATCH_SIZE_ENV="
            f"{hw['batch_size_env']} ROLLOUT_STEPS={hw['rollout_steps']}")
