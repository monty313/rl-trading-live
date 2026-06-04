"""
scripts/profile_training.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 11 — HONEST training profiler CLI. Builds the pipeline on synthetic
data (or a CSV) and profiles ~N steps, printing the same report the notebook
profiling cell shows (via core.profiling). Reports an honest GPU- vs CPU-bound
verdict instead of a misleading single utilization number.

    python scripts/profile_training.py [--csv PATH] [--steps 200]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402
from core.hardware import detect_hardware, describe_hardware  # noqa: E402
from core.profiling import profile_episodes, profile_report  # noqa: E402
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest training profiler")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = get_device()
    print(describe_hardware(detect_hardware()), flush=True)
    cfg = auto_tune_batch(dict(CFG), device)
    cfg.update({"EPISODE_BARS": 240, "BARS_PER_DAY": 60,
                "USE_AMP": False, "USE_TORCH_COMPILE": False})
    if args.csv:
        cfg["DATA_CSV_EURUSD"] = args.csv
    else:
        cfg["FEATURES"] = make_synthetic_ohlcv_array(n=600)
    env, agent, *_ = build_pipeline(cfg, device,
        phase={"name": "profile", "entry_conditions": {"buy": "any", "sell": "any"}})
    rep = profile_episodes(agent, env, device, n_steps=args.steps)
    print(profile_report(rep), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
