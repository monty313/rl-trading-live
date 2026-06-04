"""
core/interpret/action_logger.py
────────────────────────────────────────────────────────────────────────────
LIGHTWEIGHT action-distribution logger (PART 3) — NO SHAP, negligible overhead.

At a configurable interval (LOG_ACTION_DIST_EVERY rollout steps) the training
loop appends ONE row to {metrics_dir}/action_distributions.csv capturing the
current batch's mean action-prob distribution + the market state it was taken in,
so a human can later correlate "what the policy does" with "where the day is":

    bar_index, cest_time, equity, streak, dd_budget_remaining,
    dir_BUY, dir_SELL, dir_FLAT, exit_HOLD, exit_REDUCE, exit_CLOSE,
    lot_mean, lot_std

direction (BUY/SELL/FLAT) and exit (HOLD/REDUCE/CLOSE) probabilities each sum to
~1.0 by construction (softmax means over the batch). This is pure I/O over numbers
the loop already has — no extra forward passes, no gradients — so it is safe on
the hot path and fully toggleable via CFG["LOG_ACTION_DIST"].

The probability computation is factored into action_distribution() so it can be
unit-tested without a CSV / training loop.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.agent.action_space import (DIRECTION_NAMES, EXIT_NAMES, BUY, SELL,
                                      FLAT, EXIT_HOLD, EXIT_REDUCE, EXIT_CLOSE)

CSV_HEADER = [
    "bar_index", "cest_time", "equity", "streak", "dd_budget_remaining",
    "dir_BUY", "dir_SELL", "dir_FLAT",
    "exit_HOLD", "exit_REDUCE", "exit_CLOSE",
    "lot_mean", "lot_std",
]


def action_distribution(dir_logits: torch.Tensor, exit_logits: torch.Tensor,
                        lot_raw: torch.Tensor) -> Dict[str, float]:
    """Reduce a batch of head outputs to mean action probabilities + lot stats.

    dir_logits  (B, 3) direction logits, exit_logits (B, 3) exit logits, lot_raw
    (B,) the sigmoid-squashed [0,1] lot fraction (or the mapped lot — either works
    for mean/std). Returns a dict with dir_*/exit_* probabilities (each trio sums
    to ~1.0) and lot_mean/lot_std. Pure + testable."""
    dir_p = F.softmax(dir_logits, dim=-1).mean(dim=0).detach().cpu().numpy()
    exit_p = F.softmax(exit_logits, dim=-1).mean(dim=0).detach().cpu().numpy()
    lot = lot_raw.detach().cpu().numpy().reshape(-1)
    return {
        "dir_BUY": float(dir_p[BUY]),
        "dir_SELL": float(dir_p[SELL]),
        "dir_FLAT": float(dir_p[FLAT]),
        "exit_HOLD": float(exit_p[EXIT_HOLD]),
        "exit_REDUCE": float(exit_p[EXIT_REDUCE]),
        "exit_CLOSE": float(exit_p[EXIT_CLOSE]),
        "lot_mean": float(np.mean(lot)) if lot.size else 0.0,
        "lot_std": float(np.std(lot)) if lot.size else 0.0,
    }


def append_row(csv_path: str, dist: Dict[str, float], bar_index: int,
               cest_time: str, equity: float, streak: int,
               dd_budget_remaining: float) -> None:
    """Append one action-distribution row (creating the file + header on first
    write). os.makedirs the parent so a fresh metrics dir works. Crash-safe: any
    I/O failure is swallowed so logging NEVER takes down the training loop."""
    try:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_HEADER)
            w.writerow([
                bar_index, cest_time, f"{equity:.2f}", streak,
                f"{dd_budget_remaining:.4f}",
                f"{dist['dir_BUY']:.4f}", f"{dist['dir_SELL']:.4f}",
                f"{dist['dir_FLAT']:.4f}",
                f"{dist['exit_HOLD']:.4f}", f"{dist['exit_REDUCE']:.4f}",
                f"{dist['exit_CLOSE']:.4f}",
                f"{dist['lot_mean']:.4f}", f"{dist['lot_std']:.4f}",
            ])
    except Exception:                                            # pragma: no cover
        pass


def format_shift(first: Dict[str, float], latest: Dict[str, float]) -> str:
    """One-line human summary of how the direction mix shifted from the first
    logged distribution to the latest (the optional per-episode print, PART 3)."""
    def pct(d):
        return (f"BUY {d['dir_BUY']*100:.0f}/SELL {d['dir_SELL']*100:.0f}/"
                f"FLAT {d['dir_FLAT']*100:.0f}")
    return f"{pct(first)} -> {pct(latest)}"
