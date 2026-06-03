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

    A100 (>30GB VRAM)  -> BATCH_SIZE_ENV=64,  ROLLOUT_STEPS=4096
    Smaller GPU / T4   -> BATCH_SIZE_ENV=32,  ROLLOUT_STEPS=2048
    CPU (dev/CI)       -> BATCH_SIZE_ENV=4,   ROLLOUT_STEPS=64  (fast smoke tests)

    ROLLOUT_STEPS raised on GPU (2048->4096 A100) for more samples per PPO update
    and better A100 utilization (learning_loop_fix.md FIX 4.3). The PPO update is
    full-batch over time*env, so minibatch/epoch math stays coherent (n_epochs is
    independent of rollout length). Respects a caller-supplied ROLLOUT_STEPS if it
    was explicitly set non-None in the incoming cfg.
    """
    cfg = dict(cfg)
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb > 30:          # A100-class
            cfg["BATCH_SIZE_ENV"] = 64
            cfg["ROLLOUT_STEPS"] = 4096
        else:                     # T4 / smaller
            cfg["BATCH_SIZE_ENV"] = 32
            cfg["ROLLOUT_STEPS"] = 2048
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
    # ── ACCOUNT SIZE (learning_loop_fix.md FIX 3) ────────────────────────────
    # Default $10,000 so numbers are comprehensible. Override via the CLI flag
    # --account-size or by setting CFG["ACCOUNT_SIZE"] (10000/25000/50000/100000).
    # The env reads ACCOUNT_SIZE first, then falls back to INITIAL_EQUITY. FTMO
    # +2.5%/1% rules are percentage-based off start-of-day equity, and reward is
    # normalized, so everything scales automatically with account size.
    "ACCOUNT_SIZE":       10_000.0,
    "INITIAL_EQUITY":     10_000.0,
    "ACCOUNT_SIZE_CHOICES": [10_000.0, 25_000.0, 50_000.0, 100_000.0],
    "RANDOMIZE_ACCOUNT_SIZE": False,   # FUTURE HOOK — disabled (see env.reset)
    "MAX_TRADES_PER_DAY": 800,
    "LEVERAGE":           100,    # 1:100 FTMO leverage. Affects margin only —
                                  # PnL per lot is always price_move * lots * 10
                                  # (EURUSD: 100000 units * 0.0001 pip = $10/pip/lot)

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
        # ent_coef raised 0.01 -> 0.02 to keep exploration alive early: the old
        # ~$0 do-nothing policy was a collapse symptom (learning_loop_fix.md FIX
        # 1.5). lot_log_std is also floored in ppo.py so the sizing head can't
        # collapse to a deterministic 0-variance lot.
        "ent_coef":      0.02,
        "vf_coef":       0.5,
        "n_epochs":      4,
        "max_grad_norm": 0.5,
        "lot_log_std_init": -0.5,   # initial log-std for the continuous lot head
        "lot_log_std_min":  -2.0,   # floor so exploration on lot size never dies
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

    # ── Console / heartbeat (learning_loop_fix.md FIX 2) ─────────────────────
    # Heartbeat is WALL-CLOCK time-based (default 300s = 5 min), one-liner.
    "HEARTBEAT_SECS":         300,
    "BARS_PER_DAY":           1440,

    # ── Feature/indicator cache (learning_loop_fix.md FIX 4) ─────────────────
    # On first build the TF indicators + feature matrix are cached to disk keyed
    # by CSV path+mtime+feature-config hash; restart loads in seconds. Dir is
    # configurable (default next to the checkpoint/data Drive dir). None = auto.
    "FEATURE_CACHE_DIR":      None,
    "USE_FEATURE_CACHE":      True,

    # ── Reward weights (ALL normalized / percent units — account-size invariant)
    # Dense per-step shaping + sparse terminal day bonus share one O(1) scale so
    # the gradient points toward "pass FTMO" long before the first pass happens.
    # See core/env/environment.py step() REWARD block (learning_loop_fix.md FIX 1).
    "REWARD": {
        # terminal per-day bonus (applied once at day close) — BINARY PASS/FAIL
        # (ftmo_rules_fix.md RULE 2): no OK tier, no separate no-trade tier. A
        # zero-trade day is simply a FAIL and takes fail_day_penalty.
        "pass_day_bonus":    2.0,    # day reached day_start + fixed daily increment
        "fail_day_penalty":  -2.0,   # everything else (incl. zero-trade / DD breach)
        "streak_scale":      0.1,    # + per consecutive passing day
        "low_dd_threshold":  0.005,
        "low_dd_bonus":      0.3,
        # dense per-step shaping (percent-of-day-start units)
        "step_pnl_scale":        1.0,     # weight on Δequity/day_start_eq each bar
        "target_progress_scale": 0.5,     # extra pull while below+toward target
        "dd_proximity_scale":    0.02,    # quadratic penalty as DD nears the 1% cap
        "overtrade_penalty":     0.0005,  # small cost per new trade (curbs churn)
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
