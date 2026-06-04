"""
core/interpret/results_writer.py
────────────────────────────────────────────────────────────────────────────
TRAINING RESULTS WRITER (PART 1).

When training ends (normally OR on a graceful interrupt), we match the run to its
PARAMS SNAPSHOT — written by the Colab "💾 Save Snapshot" cell — by the snapshot's
`params_hash`, and append a RESULTS block to that snapshot file. This closes the
loop so the Compare panel can show, per saved config, how the run actually did.

MATCHING: snapshots live in CFG["SNAPSHOT_DIR"] as
`params_snapshot_*_<label>.json`, each with snapshot_meta.params_hash. The train
loop computes the SAME md5[:8] over the run's effective config (params_hash()) and
finds the snapshot(s) with that hash; if several match, the MOST RECENT (by
snapshot_meta.timestamp) wins. If none match (e.g. trained without saving a
snapshot first) we write nothing and log a one-liner — never crash.

NO OVERWRITE: results are appended under a list with an incrementing run_index, so
re-training the same snapshot adds {run_index: 1, ...} next to {run_index: 0, ...}
rather than clobbering the earlier run.

CRASH-SAFE: write_results() takes whatever metrics dict it is given — on a
KeyboardInterrupt the caller passes the PARTIAL metrics accrued so far (with
"interrupted": True), and we persist those. The write itself is wrapped so a
failure to write results never masks the real training error.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.interpret.dashboard_utils import params_hash


RESULT_FIELDS = [
    "pass_rate", "best_phi", "episodes_trained", "final_equity",
    "best_streak", "dd_efficiency_avg", "timestamp_completed", "phase_reached",
]


def find_snapshot_by_hash(snapshot_dir: str, target_hash: str) -> Optional[str]:
    """Return the path of the snapshot JSON whose snapshot_meta.params_hash matches
    target_hash; if multiple match, the most recent by snapshot_meta.timestamp.
    None if no snapshot matches or the dir is missing."""
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None
    candidates: List[tuple] = []
    for path in glob.glob(os.path.join(snapshot_dir, "params_snapshot_*.json")):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            continue
        meta = blob.get("snapshot_meta", {})
        if meta.get("params_hash") == target_hash:
            candidates.append((meta.get("timestamp", ""), path))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0])      # ascending timestamp
    return candidates[-1][1]                    # most recent


def write_results(snapshot_path: str, metrics: Dict[str, object]) -> Dict[str, object]:
    """Append a results block to the snapshot JSON's "results" list (creating it),
    stamping an incrementing run_index and timestamp_completed. Returns the block
    written. Crash-safe: never raises on a write/read error (returns {} instead)."""
    try:
        with open(snapshot_path) as f:
            blob = json.load(f)
    except Exception:
        return {}
    results: List[dict] = blob.get("results", [])
    if not isinstance(results, list):
        results = []
    block = {k: metrics.get(k) for k in RESULT_FIELDS}
    block["run_index"] = len(results)
    block["timestamp_completed"] = (metrics.get("timestamp_completed")
                                    or datetime.now(timezone.utc).isoformat())
    block["interrupted"] = bool(metrics.get("interrupted", False))
    results.append(block)
    blob["results"] = results
    try:
        with open(snapshot_path, "w") as f:
            json.dump(blob, f, indent=2, default=str)
    except Exception:                                            # pragma: no cover
        return {}
    return block


def record_training_results(cfg: dict, run_params: Dict[str, object],
                            metrics: Dict[str, object],
                            snapshot_dir: Optional[str] = None) -> Optional[str]:
    """Top-level call from train.py at end-of-training / interrupt. Computes the
    run's params_hash from `run_params`, finds the matching snapshot in
    snapshot_dir (CFG["SNAPSHOT_DIR"] if not given), and appends `metrics` as a
    results block. Returns the snapshot path written, or None if no snapshot
    matched (logs a one-liner either way). NEVER raises."""
    try:
        snapshot_dir = snapshot_dir or cfg.get("SNAPSHOT_DIR")
        target = params_hash(run_params)
        path = find_snapshot_by_hash(snapshot_dir, target)
        if path is None:
            print(f"[results] no params snapshot matched hash {target} in "
                  f"{snapshot_dir!r} — results not written "
                  f"(save a snapshot from the dashboard to enable this).",
                  flush=True)
            return None
        block = write_results(path, metrics)
        print(f"[results] ✅ wrote run_index={block.get('run_index')} results to "
              f"{os.path.basename(path)} (hash {target}): "
              f"pass_rate={block.get('pass_rate')}, "
              f"best_phi={block.get('best_phi')}, "
              f"episodes={block.get('episodes_trained')}"
              f"{' [interrupted]' if block.get('interrupted') else ''}.",
              flush=True)
        return path
    except Exception as exc:                                     # pragma: no cover
        print(f"[results] could not write training results: {exc}", flush=True)
        return None
