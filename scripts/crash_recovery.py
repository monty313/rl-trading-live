"""
scripts/crash_recovery.py
────────────────────────────────────────────────────────────────────────────
Run in Colab CELL 8 when training crashes. Verifies Drive checkpoints, marks
corrupt ones, finds the best non-corrupt resume, and prints the exact command
to restart training. No hardcoded paths (HARD RULE 11).

    python scripts/crash_recovery.py --checkpoint-dir DIR --manifest PATH
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.checkpoint_manager import CheckpointManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    mgr = CheckpointManager(args.checkpoint_dir, args.manifest)
    mgr.load_manifest()
    report = mgr.verify_all()
    for name in report["ok"]:
        print(f"OK: {name}")
    for name in report["corrupt"]:
        print(f"CORRUPT: {name}")
    best = mgr.find_best_resume()
    print(f"\n{len(report['ok'])} checkpoints OK, {len(report['corrupt'])} corrupt, "
          f"best resume: {best}")
    if best:
        print(f"Run Colab CELL 6 — training will resume from: {best}")
    else:
        print("No valid checkpoint — training will start fresh on CELL 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
