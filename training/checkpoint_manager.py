"""
training/checkpoint_manager.py
────────────────────────────────────────────────────────────────────────────
Checkpoint manifest + rolling-deletion + corruption check + crash recovery.

Paths are passed in (HARD RULE 11) — never hardcoded to Drive here.

PROTECTED checkpoints (NEVER deleted, HARD RULE 8):
    latest.pt, best_eval.pt, live_trading.pt, transfer_start.pt

ROLLING DELETION (HARD RULE 8 + extra-note free-space policy):
    Keep at most N phase-named checkpoints per phase (N auto-tuned by free disk:
    >20GB->5, 10-20GB->3, <10GB->2). On exceeding N, delete the LOWEST-Φ
    phase file that is not protected.

MANIFEST BOOTSTRAP (extra-note): if manifest.json is absent, build it by
scanning existing .pt files (phi=0.0 placeholder), protecting the largest file.

CORRUPTION CHECK: before save/load, torch.load(map_location="cpu") in try/except;
on failure mark corrupt=True and delete.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

PROTECTED = {"latest.pt", "best_eval.pt", "live_trading.pt", "transfer_start.pt"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, manifest_path: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest_path = Path(manifest_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"checkpoints": {}, "parity_hashes": {}}

    # ── free-space auto-tune ─────────────────────────────────────────────────
    def max_per_phase(self) -> int:
        try:
            free_gb = shutil.disk_usage(self.checkpoint_dir).free / 1e9
        except Exception:
            free_gb = 100.0
        if free_gb > 20:
            return 5
        if free_gb >= 10:
            return 3
        return 2

    # ── manifest IO ──────────────────────────────────────────────────────────
    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    self.manifest = json.load(f)
            except Exception:
                print("[ckpt] manifest unreadable — rebuilding", flush=True)
                self.bootstrap()
        else:
            self.bootstrap()
        self.manifest.setdefault("checkpoints", {})
        self.manifest.setdefault("parity_hashes", {})
        return self.manifest

    def save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def bootstrap(self):
        """Build a manifest from existing .pt files; protect the largest."""
        pts = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints = {}
        largest = max(pts, key=lambda p: p.stat().st_size, default=None)
        for p in pts:
            checkpoints[p.name] = {
                "phase": "unknown", "episode": 0, "phi": 0.0, "pass_rate": 0.0,
                "timestamp": _now_iso(), "size_bytes": p.stat().st_size,
                "corrupt": False,
                "protected": (p.name in PROTECTED) or (largest is not None and p.name == largest.name),
            }
        self.manifest = {"checkpoints": checkpoints, "parity_hashes": {}}
        self.save_manifest()
        print(f"[ckpt] MANIFEST CREATED from {len(pts)} existing checkpoints",
              flush=True)

    # ── parity hashes (HARD RULE 10) ─────────────────────────────────────────
    def set_parity_hashes(self, hashes: dict):
        self.load_manifest()
        self.manifest["parity_hashes"] = hashes
        self.save_manifest()

    def get_parity_hashes(self) -> dict:
        self.load_manifest()
        return self.manifest.get("parity_hashes", {})

    # ── corruption check ─────────────────────────────────────────────────────
    def _is_loadable(self, path: Path) -> bool:
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception:
            return False

    def verify_all(self) -> dict:
        """Try to load every checkpoint; mark corrupt ones and delete them."""
        self.load_manifest()
        report = {"ok": [], "corrupt": []}
        for name in list(self.manifest["checkpoints"].keys()):
            path = self.checkpoint_dir / name
            if not path.exists():
                continue
            if self._is_loadable(path):
                self.manifest["checkpoints"][name]["corrupt"] = False
                report["ok"].append(name)
            else:
                self.manifest["checkpoints"][name]["corrupt"] = True
                report["corrupt"].append(name)
                if not self.manifest["checkpoints"][name].get("protected"):
                    try:
                        path.unlink()
                    except Exception:
                        pass
        self.save_manifest()
        return report

    # ── save + rolling deletion ──────────────────────────────────────────────
    def save(self, agent, phase, episode: int, phi: float, pass_rate: float,
             name: Optional[str] = None) -> str:
        """Save a checkpoint, register it in the manifest, prune worst-by-Φ."""
        self.load_manifest()
        fname = name or f"{str(phase).lower()}_ep{episode:04d}.pt"
        path = self.checkpoint_dir / fname
        agent.save(str(path), extra={"phase": phase, "episode": episode,
                                     "phi": phi, "pass_rate": pass_rate})
        self.manifest["checkpoints"][fname] = {
            "phase": phase, "episode": episode, "phi": float(phi),
            "pass_rate": float(pass_rate), "timestamp": _now_iso(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "corrupt": False, "protected": fname in PROTECTED,
        }
        # also refresh latest.pt (protected pointer to most recent)
        if fname not in PROTECTED:
            latest = self.checkpoint_dir / "latest.pt"
            agent.save(str(latest), extra={"phase": phase, "episode": episode,
                                           "phi": phi, "pass_rate": pass_rate})
            self.manifest["checkpoints"]["latest.pt"] = {
                "phase": phase, "episode": episode, "phi": float(phi),
                "pass_rate": float(pass_rate), "timestamp": _now_iso(),
                "size_bytes": latest.stat().st_size if latest.exists() else 0,
                "corrupt": False, "protected": True,
            }
        self._prune_phase(phase)
        self.save_manifest()
        return str(path)

    def _prune_phase(self, phase):
        """Keep at most max_per_phase non-protected files for this phase."""
        keep = self.max_per_phase()
        cks = self.manifest["checkpoints"]
        phase_files = [
            (n, m) for n, m in cks.items()
            if m.get("phase") == phase and n not in PROTECTED and not m.get("protected")
        ]
        if len(phase_files) <= keep:
            return
        # sort by phi ascending; delete the lowest-Φ ones beyond `keep`
        phase_files.sort(key=lambda kv: kv[1].get("phi", 0.0))
        to_delete = phase_files[:len(phase_files) - keep]
        for name, meta in to_delete:
            path = self.checkpoint_dir / name
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            print(f"[ckpt] DELETED {name} Φ={meta.get('phi')} "
                  f"ep={meta.get('episode')} — worst for phase {phase}", flush=True)
            del cks[name]

    # ── resume selection ─────────────────────────────────────────────────────
    def find_best_resume(self, phase=None) -> Optional[Path]:
        """Return path of the highest-Φ non-corrupt PPO checkpoint (prefer `phase`).
        Skips checkpoints whose phase is 'unknown' — these are legacy DQN files
        that cannot be loaded by PPOAgent and would cause a CUDAGraphs crash."""
        self.load_manifest()
        cks = self.manifest["checkpoints"]
        candidates = [
            (n, m) for n, m in cks.items()
            if not m.get("corrupt")
            and m.get("phase", "unknown") != "unknown"   # skip legacy DQN
            and (self.checkpoint_dir / n).exists()
        ]
        if not candidates:
            return None
        if phase is not None:
            phase_cands = [(n, m) for n, m in candidates if m.get("phase") == phase]
            if phase_cands:
                candidates = phase_cands
        best = max(candidates, key=lambda kv: kv[1].get("phi", -1e9))
        return self.checkpoint_dir / best[0]
