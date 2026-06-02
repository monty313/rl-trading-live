"""
core/settings.py
────────────────────────────────────────────────────────────────────────────
Central CFG dict — ported from gpu_rl_trading/config/settings.py (REPO1, the
authoritative source) with the A100 upgrades approved in the extra-note:

    BATCH_SIZE_RL : 256  -> 2048   (A100 80GB headroom)
    TRAIN_EVERY   : 4    -> 2      (halve GPU idle gap; not 1 — DQN needs decorrelation)
    MEMORY_SIZE   : 100k -> 500k   (larger replay; A100 VRAM supports it)
    NUM_ACTIONS   : 7    -> 756    (imported from action_space.py, never hardcoded)

Paths are NOT hardcoded here for Drive — train.py / live_runner receive them as
CLI args or read them from config YAML (HARD RULE 11). The defaults below are
local/dev fallbacks only.

Device is auto-detected: CUDA (A100/T4) when available, else CPU. The same code
runs identically in Colab (GPU) and in CI/dev (CPU) — only speed differs.
"""
from __future__ import annotations

import torch

from core.agent.action_space import NUM_ACTIONS


def get_device() -> torch.device:
    """Return CUDA device if available (Colab A100/T4), else CPU (dev/CI)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_tune_batch(cfg: dict, device: torch.device) -> dict:
    """
    Auto-scale batch sizes to the detected GPU (extra-note fallback policy).

    A100 (>30GB VRAM)  -> full sizes (BATCH_SIZE_RL=2048, BATCH_SIZE_ENV=64)
    Smaller GPU / T4   -> BATCH_SIZE_RL=512, BATCH_SIZE_ENV=32
    CPU (dev/CI)       -> tiny sizes so smoke tests run fast
    """
    cfg = dict(cfg)
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb > 30:          # A100-class
            cfg["BATCH_SIZE_RL"] = 2048
            cfg["BATCH_SIZE_ENV"] = 64
        else:                     # T4 / smaller
            cfg["BATCH_SIZE_RL"] = 512
            cfg["BATCH_SIZE_ENV"] = 32
    else:                         # CPU dev/CI
        cfg["BATCH_SIZE_RL"] = 64
        cfg["BATCH_SIZE_ENV"] = 4
    return cfg


# ── The master CFG dict ──────────────────────────────────────────────────────
CFG = {
    # Data (paths supplied via CLI / YAML at runtime — never hardcoded to Drive)
    "DATA_CSV_EURUSD": None,
    "SYMBOL":          "EURUSD",

    # Episode
    "EPISODE_BARS":    43_200,    # ~30 trading days of 1m bars
    "BATCH_SIZE_ENV":  64,        # parallel episodes on GPU (auto-tuned per device)
    "LOOKBACK":        20,        # bars of history in state
    "TF_FACTORS":      [1, 15, 60, 1440],   # resample factors from 1m

    # FTMO / risk (mirrors config/trading_policy.yaml; YAML wins at runtime)
    "DAILY_TARGET_PCT":   0.025,  # PASS target = start * 1.025
    "DAILY_MAX_DD_PCT":   0.010,  # 1% trailing DD
    "INITIAL_EQUITY":     100_000.0,
    "MAX_TRADES_PER_DAY": 800,

    # Agent / training
    "STATE_DIM":       None,      # filled at runtime from env.state_dim
    "NUM_ACTIONS":     NUM_ACTIONS,   # 756 — imported, never hardcoded
    "HIDDEN":          256,
    "LR":              5e-4,
    "GAMMA":           0.95,
    "EPSILON_START":   0.9,
    "EPSILON_MIN":     0.05,
    "EPSILON_DECAY_EPISODES": 500,
    "BATCH_SIZE_RL":   2048,      # A100 upgrade (auto-tuned per device)
    "MEMORY_SIZE":     500_000,   # A100 upgrade
    "TRAIN_EVERY":     2,         # A100 upgrade (was 4)
    "SYNC_EVERY":      200,       # steps between target-net sync

    # Transfer learning (old small action space -> new 756 space)
    "TRANSFER_EPSILON":        0.3,   # exploration after transfer
    "TRANSFER_EPISODES":       200,   # episodes to hold elevated epsilon

    # torch.compile / AMP toggles (auto-disabled on CPU)
    "USE_TORCH_COMPILE": True,
    "USE_AMP":           True,

    # Curriculum
    "PHASE":                  0,
    "MAX_EPISODES_PER_PHASE": 500,
    "CHECKPOINT_EVERY":       10,
    "EVAL_EVERY":             50,

    # ── Potential-based reward shaping (Φ) ───────────────────────────────────
    # Φ = (pass_rate × avg_ret_normalised) / (1 + λ × avg_dd_normalised)
    "SHAPE_ALPHA":   0.006,
    "SHAPE_CLIP":    0.006,
    "SHAPE_LAMBDA":  5.0,
    "SHAPE_WARMUP":  150,
    "WEEKLY_BONUS":  0.02,    # weekly_consistency_bonus when 7-day pass rate improves
    "PASS_NO_BREACH_BONUS": 0.01,   # bonus when target hit AND no DD breach
}
