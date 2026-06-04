"""
tests/unit/test_reliability_s7.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 7 — training reliability. Covers:
  • checkpoints written ATOMICALLY (temp + os.replace; no half-written file,
    no leftover temp), and a save/load roundtrip restores weights;
  • a CORRUPTED checkpoint is detected and the resume path falls back cleanly;
  • a SCHEMA-MISMATCH checkpoint loads with a loud input-layer reinit (not silent);
  • CONTINUOUS metrics: _append_metrics appends one JSON line per episode;
  • the heartbeat file is written;
  • phases.yaml loads & is ordered;
  • SEED reproducibility: set_global_seed makes two agents' rollouts identical;
  • worst-checkpoint pruning can NEVER delete the only good checkpoint.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from core.agent.ppo import PPOAgent, OBS_SCHEMA_VERSION
from core.seeding import set_global_seed
from training.checkpoint_manager import CheckpointManager, PROTECTED
from core.settings import CFG

DEV = torch.device("cpu")


def _agent(state_dim=16):
    c = dict(CFG); c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False})
    return PPOAgent(state_dim, c, DEV)


# ════════════════════════════════════════════════════════════════════════════
# Atomic checkpoint write + roundtrip.
# ════════════════════════════════════════════════════════════════════════════
def test_save_is_atomic_no_temp_left_and_roundtrips(tmp_path):
    a = _agent()
    path = tmp_path / "ckpt.pt"
    a.save(str(path), extra={"phase": "ph0", "episode": 7})
    assert path.exists(), "atomic save did not produce the final file"
    # No leftover temp files in the directory.
    leftover = glob.glob(str(tmp_path / ".ckpt_*"))
    assert leftover == [], f"atomic save leaked temp files: {leftover}"
    # Roundtrip: a fresh agent loads it and matches the saved weights.
    b = _agent()
    # perturb b so we can prove load actually changed it
    with torch.no_grad():
        for p in b.net.parameters():
            p.add_(1.0)
    b.load(str(path))
    for (ka, va), (kb, vb) in zip(a.net.state_dict().items(),
                                  b.net.state_dict().items()):
        assert torch.allclose(va, vb), f"weights differ after load at {ka}"


def test_save_overwrites_in_place_atomically(tmp_path):
    """A second save to the SAME path must replace it atomically (the file must
    always remain loadable; no partial/corrupt intermediate)."""
    a = _agent()
    path = tmp_path / "latest.pt"
    a.save(str(path), extra={"episode": 1})
    first = torch.load(path, map_location="cpu", weights_only=False)
    a.save(str(path), extra={"episode": 2})
    second = torch.load(path, map_location="cpu", weights_only=False)
    assert first["episode"] == 1 and second["episode"] == 2
    assert glob.glob(str(tmp_path / ".ckpt_*")) == []


# ════════════════════════════════════════════════════════════════════════════
# Corrupted checkpoint detected; resume falls back.
# ════════════════════════════════════════════════════════════════════════════
def test_corrupted_checkpoint_detected_and_resume_falls_back(tmp_path):
    d = tmp_path / "gpu"; d.mkdir()
    a = _agent()
    mgr = CheckpointManager(str(d), str(d / "manifest.json"))
    mgr.save(a, "ph0", 10, phi=0.5, pass_rate=0.5, name="good.pt")
    mgr.save(a, "ph0", 20, phi=0.6, pass_rate=0.5)   # also writes latest.pt
    # Corrupt the latest.pt on disk.
    (d / "latest.pt").write_text("garbage not a torch checkpoint")
    report = mgr.verify_all()
    assert "latest.pt" in report["corrupt"]
    # Resume must still find a good (non-corrupt) checkpoint.
    best = mgr.find_best_resume()
    assert best is not None and best.exists()
    # And it must be loadable.
    a2 = _agent()
    a2.load(str(best))


# ════════════════════════════════════════════════════════════════════════════
# Schema-mismatch handled loudly (input-layer reinit), not a silent mis-load.
# ════════════════════════════════════════════════════════════════════════════
def test_schema_mismatch_reinits_input_layer(tmp_path, capsys):
    a = _agent(state_dim=16)
    path = tmp_path / "ck.pt"
    a.save(str(path))
    # Load into an agent with a DIFFERENT input dim -> schema mismatch path.
    b = _agent(state_dim=20)
    b.load(str(path))
    out = capsys.readouterr().out
    assert "OBSERVATION-SCHEMA MISMATCH" in out, "schema mismatch was not announced"
    # The input layer must match b's (new) width, not the checkpoint's.
    assert b.net.state_dict()["trunk.0.weight"].shape[1] == 20


# ════════════════════════════════════════════════════════════════════════════
# Continuous metrics + heartbeat files.
# ════════════════════════════════════════════════════════════════════════════
def test_append_metrics_writes_one_line_per_episode(tmp_path):
    from training.train import _append_metrics
    for ep in range(1, 4):
        _append_metrics(str(tmp_path), {"episode": ep, "phi": 0.1 * ep})
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    assert [r["episode"] for r in rows] == [1, 2, 3]
    assert all("timestamp" in r for r in rows)


def test_heartbeat_file_written(tmp_path):
    from training.train import _write_heartbeat
    _write_heartbeat(str(tmp_path), episode=5, phase="ph0")
    hb = json.loads((tmp_path / "heartbeat_training.txt").read_text())
    assert hb["episode"] == 5 and hb["phase"] == "ph0"


# ════════════════════════════════════════════════════════════════════════════
# phases.yaml loads and is ordered.
# ════════════════════════════════════════════════════════════════════════════
def test_phases_load_and_are_ordered():
    from training.train import _load_phases
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phases = _load_phases(repo_root)
    assert len(phases) > 0
    orders = [p.get("order", 0) for p in phases]
    assert orders == sorted(orders), "phases not returned in ascending order"


# ════════════════════════════════════════════════════════════════════════════
# CLI overrides incl --seed are parsed.
# ════════════════════════════════════════════════════════════════════════════
def test_cli_has_seed_and_core_overrides():
    import argparse, training.train as T
    # Reconstruct the parser the same way main() does by introspecting argparse
    # is brittle; instead assert the seed wiring exists by calling set_global_seed
    # and that train.py imports it lazily. We at least confirm the flag is declared.
    src = open(T.__file__).read()
    for flag in ("--seed", "--account-size", "--target-pct", "--max-dd-pct",
                 "--resume", "--force-fresh", "--start-phase"):
        assert flag in src, f"CLI flag {flag} missing from train.py"


# ════════════════════════════════════════════════════════════════════════════
# Seed reproducibility.
# ════════════════════════════════════════════════════════════════════════════
def test_seed_makes_rollout_sampling_reproducible():
    """Two agents seeded identically must emit identical sampled actions for the
    same observation (weight init + sampling RNG both reproducible)."""
    # Fixed observation built OUTSIDE the seeded region so it is byte-identical
    # for both runs and does not consume the shared RNG between seed and sample.
    state = torch.randn(8, 16, generator=torch.Generator().manual_seed(99))
    outs = []
    for _ in range(2):
        set_global_seed(1234)
        a = _agent(state_dim=16)          # weight init consumes RNG identically
        out = a.select_actions(state)     # sampling consumes RNG identically
        outs.append((out["direction"].clone(), out["exit"].clone(),
                     out["lot_raw"].clone()))
    assert torch.equal(outs[0][0], outs[1][0]), "direction sampling not reproducible"
    assert torch.equal(outs[0][1], outs[1][1]), "exit sampling not reproducible"
    assert torch.allclose(outs[0][2], outs[1][2]), "lot sampling not reproducible"


def test_different_seeds_differ():
    s = torch.randn(8, 16, generator=torch.Generator().manual_seed(99))
    set_global_seed(1)
    a = _agent(state_dim=16)
    o1 = a.select_actions(s)["lot_raw"].clone()
    set_global_seed(2)
    b = _agent(state_dim=16)
    o2 = b.select_actions(s)["lot_raw"].clone()
    assert not torch.allclose(o1, o2), "different seeds produced identical sampling"


# ════════════════════════════════════════════════════════════════════════════
# Pruning can never delete the only good checkpoint.
# ════════════════════════════════════════════════════════════════════════════
def test_prune_never_deletes_only_good_checkpoint(tmp_path, monkeypatch):
    d = tmp_path / "gpu"; d.mkdir()
    a = _agent()
    mgr = CheckpointManager(str(d), str(d / "manifest.json"))
    # Force keep=1 and a tiny phase so pruning is aggressive.
    monkeypatch.setattr(mgr, "max_per_phase", lambda: 1)
    mgr.save(a, "ph0", 10, phi=0.5, pass_rate=0.5, name="only_a.pt")
    # After saving one more, pruning could try to remove the lower-Φ one, but a
    # good file must always survive somewhere (best_eval/latest/highest-Φ).
    mgr.save(a, "ph0", 20, phi=0.9, pass_rate=0.5, name="only_b.pt")
    good = [n for n, m in mgr.manifest["checkpoints"].items()
            if not m.get("corrupt") and (d / n).exists()]
    assert len(good) >= 1, "pruning deleted the last good checkpoint"
