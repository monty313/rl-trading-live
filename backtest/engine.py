"""
backtest/engine.py
────────────────────────────────────────────────────────────────────────────
Backtest engine. Uses the SAME indicators.py + intrabar_fills.py + env + guard
as training, so backtest results match training/live (no divergence).

PARITY ASSERTION (HARD RULE 10): at startup, compute md5 of
core/env/indicators.py and core/env/intrabar_fills.py and compare to the
"parity_hashes" stored in the manifest. On mismatch -> ParityError, BEFORE any
trading occurs. If the manifest has no stored hashes yet, the current hashes are
recorded (first run establishes the baseline).

run_backtest(...) -> {
    daily_returns, pass_fail, phi, total_pass_days, total_fail_days, max_drawdown_pct
}
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import torch

from core.settings import CFG, get_device, auto_tune_batch
from core.pipeline import build_pipeline, load_ohlcv_csv
from core.reward.shaper import EpisodeRewardShaper

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ParityError(Exception):
    """Raised when indicators/fills md5 differs from the trained baseline."""


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def current_parity_hashes() -> dict:
    """md5 of the parity-critical source files (indicators + fills)."""
    return {
        "indicators.py": _md5(os.path.join(_REPO_ROOT, "core", "env", "indicators.py")),
        "intrabar_fills.py": _md5(os.path.join(_REPO_ROOT, "core", "env", "intrabar_fills.py")),
    }


def assert_parity(manifest_path: Optional[str]):
    """
    Compare current md5s to those in the manifest. Establish the baseline on
    first run; raise ParityError on any mismatch thereafter.
    """
    cur = current_parity_hashes()
    if not manifest_path or not os.path.exists(manifest_path):
        return cur   # nothing to compare against yet
    from training.checkpoint_manager import CheckpointManager
    mgr = CheckpointManager(os.path.dirname(manifest_path), manifest_path)
    stored = mgr.get_parity_hashes()
    if not stored:
        mgr.set_parity_hashes(cur)   # first run establishes baseline
        return cur
    for name, h in cur.items():
        if stored.get(name) and stored[name] != h:
            raise ParityError(
                f"{name} changed since last training run (md5 {stored[name]} -> "
                f"{h}). Backtest results may not match training. Re-train or revert "
                f"{name} to match.")
    return cur


def run_backtest(csv: Optional[str], cfg: dict = None, device: torch.device = None,
                 manifest_path: Optional[str] = None, n_days: int = 10,
                 features=None) -> dict:
    """Run a deterministic backtest and return the 6-key result dict."""
    device = device or get_device()
    cfg = auto_tune_batch(dict(cfg or CFG), device)
    if features is not None:
        cfg["FEATURES"] = features
    elif csv:
        cfg["DATA_CSV_EURUSD"] = csv

    assert_parity(manifest_path)   # HARD RULE 10 — before any trading

    env, agent, sizer, guard, gate = build_pipeline(
        cfg, device, phase={"entry_conditions": {"buy": "any", "sell": "any"}})

    bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
    env.reset()
    daily_returns, pass_fail = [], []
    peak = float(env._equity[0].item())
    max_dd = 0.0
    day_start = float(env._equity[0].item())
    steps = n_days * bars_per_day
    # PASS uses the FIXED daily increment off INITIAL equity (ftmo_rules_fix.md
    # RULE 1): daily_target = day_start + daily_increment, NOT a percent of the
    # day's opening balance. Binary PASS/FAIL only.
    daily_increment = float(env.daily_increment)

    for step in range(steps):
        mask = env.current_direction_mask()
        out = agent.select_actions(env._get_state(), mask=mask)
        _s, _r, done, info = env.step(out)
        eq = float(info["equity"][0].item())
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / (peak + 1e-9))
        if (step + 1) % bars_per_day == 0:
            ret = (eq - day_start) / (day_start + 1e-9)
            daily_returns.append(ret)
            passed = eq >= day_start + daily_increment
            pass_fail.append("PASS" if passed else "FAIL")
            day_start = eq
        if done.all():
            break

    if not daily_returns:
        eq = float(env._equity[0].item())
        daily_returns = [(eq - day_start) / (day_start + 1e-9)]
        pass_fail = ["PASS" if eq >= day_start + daily_increment else "FAIL"]

    total_pass = pass_fail.count("PASS")
    total_fail = pass_fail.count("FAIL")
    pass_rate = total_pass / max(len(pass_fail), 1)
    shaper = EpisodeRewardShaper(cfg)
    phi = shaper._phi(pass_rate, float(np.mean(daily_returns)), max_dd)

    return {
        "daily_returns": daily_returns,
        "pass_fail": pass_fail,
        "phi": float(phi),
        "total_pass_days": total_pass,
        "total_fail_days": total_fail,
        "max_drawdown_pct": float(max_dd * 100.0),
    }
