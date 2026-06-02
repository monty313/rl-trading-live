"""
core/settings.py
────────────────────────────────────────────────────────────────────────────
Central CFG dict. Agent is PURE PPO (DQN deprecated -> legacy/; see
DESIGN_DECISIONS.md #1). A100 tuning scales the parallel env batch and the
on-policy rollout length:

    BATCH_SIZE_ENV : parallel episodes on GPU (4 CPU / 32 T4 / 64 A100)
    ROLLOUT_STEPS  : on-policy steps per PPO update (tuned per device)

Paths are NOT hardcoded here for Drive — train.py / live_runner receive them as
CLI args or read them from config YAML (HARD RULE 11). The defaults below are
local/dev fallbacks only.

Device is auto-detected: CUDA (A100/T4) when available, else CPU. The same code
runs identically in Colab (GPU) and in CI/dev (CPU) — only speed differs.
"""
from __future__ import annotations

import torch



def get_device() -> torch.device:
    """Return CUDA device if available (Colab A100/T4), else CPU (dev/CI)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_tune_batch(cfg: dict, device: torch.device) -> dict:
    """
    Auto-scale batch sizes to the detected GPU (extra-note fallback policy).

    A100 (>30GB VRAM)  -> BATCH_SIZE_ENV=64,  ROLLOUT_STEPS=2048
    Smaller GPU / T4   -> BATCH_SIZE_ENV=32,  ROLLOUT_STEPS=1024
    CPU (dev/CI)       -> BATCH_SIZE_ENV=4,   ROLLOUT_STEPS=64  (fast smoke tests)
    """
    cfg = dict(cfg)
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb > 30:          # A100-class
            cfg["BATCH_SIZE_ENV"] = 64
            cfg["ROLLOUT_STEPS"] = 2048
        else:                     # T4 / smaller
            cfg["BATCH_SIZE_ENV"] = 32
            cfg["ROLLOUT_STEPS"] = 1024
    else:                         # CPU dev/CI
        cfg["BATCH_SIZE_ENV"] = 4
        cfg["ROLLOUT_STEPS"] = 64
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

    # Agent / training — PURE PPO (DQN deprecated; see DESIGN_DECISIONS.md #1)
    "STATE_DIM":       None,      # filled at runtime from env.state_dim
    "HIDDEN":          256,
    "LR":              3e-4,
    "GAMMA":           0.95,
    "ROLLOUT_STEPS":   2048,      # on-policy steps collected before each PPO update
    "PPO": {                      # PPO hyperparameters (see core/agent/ppo.py)
        "learning_rate": 3e-4,
        "gamma":         0.95,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        "ent_coef":      0.01,
        "vf_coef":       0.5,
        "n_epochs":      4,
        "max_grad_norm": 0.5,
    },

    # torch.compile / AMP toggles (auto-disabled on CPU).
    # compile uses mode="default" — NOT "reduce-overhead" which uses CUDA Graphs
    # and overwrites rollout buffer tensors, crashing torch.stack() in update().
    "USE_TORCH_COMPILE": True,
    "USE_AMP":           True,

    # Curriculum
    "PHASE":                  0,
    "MAX_EPISODES_PER_PHASE": 500,
    "CHECKPOINT_EVERY":       10,
    "EVAL_EVERY":             50,

    # Per-day reward weights
    "REWARD": {
        "pass_day_bonus":    2.0,
        "ok_day_bonus":      0.5,
        "fail_day_penalty":  -2.0,
        "no_trade_penalty":  -1.0,   # gate was active all day but agent never traded
        "streak_scale":      0.1,
        "low_dd_threshold":  0.005,
        "low_dd_bonus":      0.3,
    },

    # ── Potential-based reward shaping (Φ) ───────────────────────────────────
    # Φ = (pass_rate × avg_ret_normalised) / (1 + λ × avg_dd_normalised)
    "SHAPE_ALPHA":   0.006,
    "SHAPE_CLIP":    0.006,
    "SHAPE_LAMBDA":  5.0,
    "SHAPE_WARMUP":  150,
    "WEEKLY_BONUS":  0.02,    # weekly_consistency_bonus when 7-day pass rate improves
    "PASS_NO_BREACH_BONUS": 0.01,   # bonus when target hit AND no DD breach
}
