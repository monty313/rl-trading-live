"""Unit tests for CheckpointManager: rolling deletion, bootstrap, resume, transfer."""
import json
import torch
from training.checkpoint_manager import CheckpointManager, PROTECTED
from core.agent.ppo import PPOAgent
from core.settings import CFG

DEV = torch.device("cpu")


def _agent():
    c = dict(CFG); c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False})
    return PPOAgent(16, c, DEV)


def test_rolling_deletion_keeps_five_lowest_phi_removed(tmp_path, monkeypatch):
    mgr = CheckpointManager(str(tmp_path / "gpu"), str(tmp_path / "gpu" / "manifest.json"))
    monkeypatch.setattr(mgr, "max_per_phase", lambda: 5)
    agent = _agent()
    phis = [0.1, 0.5, 0.2, 0.9, 0.3, 0.05]   # 6 saves; lowest (0.05) must be deleted
    for i, phi in enumerate(phis):
        mgr.save(agent, "ph0", episode=(i + 1) * 10, phi=phi, pass_rate=0.5)
    cks = mgr.manifest["checkpoints"]
    phase_files = [n for n, m in cks.items() if m.get("phase") == "ph0" and n not in PROTECTED]
    assert len(phase_files) == 5
    remaining_phis = sorted(m["phi"] for n, m in cks.items()
                            if m.get("phase") == "ph0" and n not in PROTECTED)
    assert 0.05 not in remaining_phis   # lowest phi deleted


def test_protected_never_deleted(tmp_path):
    mgr = CheckpointManager(str(tmp_path / "gpu"), str(tmp_path / "gpu" / "manifest.json"))
    agent = _agent()
    mgr.save(agent, "ph0", episode=10, phi=0.9, pass_rate=0.6, name="best_eval.pt")
    for i in range(8):
        mgr.save(agent, "ph0", episode=100 + i, phi=0.01 * i, pass_rate=0.1)
    assert (mgr.checkpoint_dir / "best_eval.pt").exists()


def test_bootstrap_from_existing(tmp_path):
    d = tmp_path / "gpu"; d.mkdir()
    agent = _agent()
    # create two raw .pt files, no manifest
    agent.save(str(d / "old_a.pt"))
    agent.save(str(d / "old_b.pt"))
    mgr = CheckpointManager(str(d), str(d / "manifest.json"))
    mgr.bootstrap()
    assert (d / "manifest.json").exists()
    assert "old_a.pt" in mgr.manifest["checkpoints"]


def test_find_best_resume(tmp_path):
    mgr = CheckpointManager(str(tmp_path / "gpu"), str(tmp_path / "gpu" / "manifest.json"))
    agent = _agent()
    mgr.save(agent, "ph0", 10, phi=0.2, pass_rate=0.5)
    mgr.save(agent, "ph0", 20, phi=0.8, pass_rate=0.7)
    best = mgr.find_best_resume()
    assert best is not None and best.exists()


def test_verify_all_detects_corrupt(tmp_path):
    d = tmp_path / "gpu"; d.mkdir()
    agent = _agent()
    mgr = CheckpointManager(str(d), str(d / "manifest.json"))
    mgr.save(agent, "ph0", 10, phi=0.5, pass_rate=0.5, name="good.pt")
    bad = d / "bad.pt"; bad.write_text("not a torch file")
    mgr.manifest["checkpoints"]["bad.pt"] = {"phase": "ph0", "episode": 0,
        "phi": 0.0, "pass_rate": 0.0, "timestamp": "x", "size_bytes": 5,
        "corrupt": False, "protected": False}
    mgr.save_manifest()
    report = mgr.verify_all()
    assert "bad.pt" in report["corrupt"]
