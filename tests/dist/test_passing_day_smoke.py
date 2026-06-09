"""
Permanent test: confirm the architecture can produce a PASS day even with
an UNTRAINED PPO. This catches regressions where a config change makes the
$250 daily target structurally unreachable (e.g. lot windows too narrow,
gate logic broken, exit head locked, etc.).

We don't assert ≥1 PASS day on every individual seed — random exploration
on a small dataset is noisy and some seeds genuinely happen to lose. What
we DO assert is the aggregate signal:

  - across 4 seeds × 2 episodes × ~4 days = ~32 day attempts,
  - at least one day must end ≥ $250 in PnL,
  - the max single-day PnL across all runs must exceed +$250.

A regression that breaks the architecture (e.g. clamps lots to 0.01 again,
or pins the mask so PPO can never enter, or freezes the gradient updates)
will drop these numbers to zero.

Runs in ~80s on CPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "mini_train_smoke.py"
SEEDS = (11, 33, 42, 55)


def _run_smoke(seed: int):
    """Run the mini-train smoke once and return (pass_days_total, max_pnl)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--episodes", "2", "--bars", "6000",
         "--seed", str(seed), "--quiet"],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=180,
    )
    out = result.stdout
    pass_days = 0
    max_pnl = float("-inf")
    for line in out.splitlines():
        if "TOTAL PASS DAYS:" in line:
            pass_days = int(line.rsplit(":", 1)[1].strip())
        if "max $" in line:
            # e.g.  "min $-121.98 / median $-65.82 / max $+479.02"
            tail = line.rsplit("max $", 1)[1].strip()
            max_pnl = float(tail.replace("+", "").replace(",", ""))
    return pass_days, max_pnl, out


@pytest.mark.slow
def test_architecture_can_produce_pass_day_across_seeds():
    """Aggregate across 4 seeds — total PASS days ≥ 1 AND a single day exceeded
    +$250. This proves $250/day is reachable with the current config + masks
    + lot windows + reward shape, even without any training."""
    total_passes = 0
    best_pnl = float("-inf")
    per_seed = []
    for s in SEEDS:
        passes, max_pnl, _out = _run_smoke(s)
        per_seed.append((s, passes, max_pnl))
        total_passes += passes
        best_pnl = max(best_pnl, max_pnl)
    msg = (
        "Architecture regression: untrained PPO cannot produce a PASS day "
        f"across {len(SEEDS)} seeds. Per-seed (passes, max_pnl): {per_seed}. "
        f"Total passes={total_passes}, best PnL=${best_pnl:+.2f}."
    )
    assert total_passes >= 1, msg
    assert best_pnl > 250.0, msg
