#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
# Run this BEFORE wiring DistPrePhaseWrapper into training. It loads the
# DQN checkpoint, identifies the policy head, prints its input/output dims,
# compares them to the current PPO base obs dim, and tells you whether you
# need a DistObsAdapter.
#
# Colab usage:
#     %run scripts/dist_checkpoint_probe.py
# CLI usage:
#     python scripts/dist_checkpoint_probe.py --ckpt /path/to/eurusd_gpu_ph0_ep0120.pt
# ═══════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


DEFAULT_CKPT = "/content/drive/MyDrive/checkpoints/eurusd_gpu_ph0_ep0120.pt"
GDRIVE_FILE_ID = "1s1sC0OFBnbEicgEnkhAzHcw4qiJt1Kvc"


def _download_if_needed(path: str) -> str:
    if os.path.exists(path):
        return path
    print(f"[DIST] Checkpoint not at {path} — attempting gdown fallback")
    try:
        import gdown  # type: ignore

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            output=path,
            quiet=False,
        )
        return path
    except Exception as e:
        raise SystemExit(
            f"[DIST] Could not download checkpoint: {e}\n"
            f"      Place it manually at: {path}"
        )


def probe(ckpt_path: str) -> dict:
    print("[DIST] Loading checkpoint for inspection...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(
        f"\n[DIST] Top-level keys: "
        f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )
    if isinstance(ckpt, dict):
        for key, val in ckpt.items():
            if isinstance(val, dict):
                print(f"\n  [{key}] sub-keys: {list(val.keys())[:10]}")
                for subkey, tensor in val.items():
                    if hasattr(tensor, "shape"):
                        print(f"    {subkey}: {tuple(tensor.shape)}")
            elif hasattr(val, "shape"):
                print(f"  {key}: {tuple(val.shape)}")
            else:
                print(f"  {key}: {type(val).__name__}")

    # Locate policy weights.
    from core.dist_teacher.dist_dqn_teacher import (
        _find_state_dict,
        _infer_input_dim,
        _infer_output_dim,
    )

    state_dict = _find_state_dict(ckpt)
    if state_dict is None:
        raise SystemExit(
            "[DIST] Could not locate policy state_dict — manual inspection required."
        )
    in_dim, in_key = _infer_input_dim(state_dict)
    out_dim, out_key = _infer_output_dim(state_dict)
    print(f"\n[DIST] DQN POLICY INPUT DIM:  {in_dim}  (from '{in_key}')")
    print(f"[DIST] DQN POLICY OUTPUT DIM: {out_dim} (from '{out_key}')")

    # Compare against current PPO base obs dim.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core.settings import CFG, get_device, auto_tune_batch  # type: ignore
        from core.env.environment import BatchedFTMOEnv  # type: ignore

        device = get_device()
        cfg = auto_tune_batch(dict(CFG), device)
        cfg["BATCH_SIZE_ENV"] = 1
        cfg["EPISODE_BARS"] = 1024
        # If real data isn't wired up, just compute state_dim symbolically.
        # state_dim = lkbk * F + N_POSITION_FEATS + N_FTMO_FEATS + N_SESSION_FEATS
        from core.env.environment import (
            N_POSITION_FEATS,
            N_FTMO_FEATS,
            N_SESSION_FEATS,
        )

        lkbk = int(cfg.get("LOOKBACK", 20))
        # F (per-bar feature count) depends on the feature pipeline; report
        # the symbolic formula since we likely don't have data loaded here.
        print(
            f"\n[DIST] PPO base obs formula: lkbk({lkbk}) * F + "
            f"{N_POSITION_FEATS} + {N_FTMO_FEATS} + {N_SESSION_FEATS}"
        )
        print(
            "[DIST] To compute the EXACT dim, build the env with real data "
            "and read env.state_dim."
        )
    except Exception as e:
        print(f"[DIST] Could not import env for symbolic check: {e}")

    return {
        "dqn_input_dim": in_dim,
        "dqn_output_dim": out_dim,
        "first_weight_key": in_key,
        "last_weight_key": out_key,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = parser.parse_args()
    ckpt = _download_if_needed(args.ckpt)
    result = probe(ckpt)
    print("\n[DIST] Probe complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
