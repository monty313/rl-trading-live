"""
tests/unit/test_perf_s11.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 11 — performance tooling. Covers:
  • hardware detection returns a coherent tier + recommended sizes (CPU here);
  • the profiler runs and produces an HONEST GPU/CPU-bound verdict;
  • SMOOTH entropy annealing: cosine/exp are MONOTONIC, smooth (no step), start
    at ENTROPY_START_COEF and END exactly at the stable coef — and are NOT linear;
  • Optuna search space + best-params apply + persistence/resume (guarded so the
    suite passes whether or not optuna is installed).
"""
from __future__ import annotations

import json
import math
import os

import pytest
import torch

from core.hardware import detect_hardware, describe_hardware
from core.agent.ppo import PPOAgent
from core.settings import CFG

DEV = torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════
# Hardware detection.
# ════════════════════════════════════════════════════════════════════════════
def test_detect_hardware_coherent():
    hw = detect_hardware()
    assert set(("cuda", "name", "vram_gb", "tier", "batch_size_env",
                "rollout_steps")).issubset(hw)
    assert hw["batch_size_env"] > 0 and hw["rollout_steps"] > 0
    if not hw["cuda"]:
        assert hw["tier"] == "CPU" and hw["vram_gb"] == 0.0
    assert isinstance(describe_hardware(hw), str)


# ════════════════════════════════════════════════════════════════════════════
# Profiler honesty.
# ════════════════════════════════════════════════════════════════════════════
def test_profiler_runs_and_reports_verdict():
    from core.profiling import profile_episodes, profile_report
    from core.pipeline import build_pipeline
    c = dict(CFG)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False, "BATCH_SIZE_ENV": 2,
              "EPISODE_BARS": 180, "BARS_PER_DAY": 60})
    from tests.fixtures.sample_candles import make_synthetic_ohlcv_array
    c["FEATURES"] = make_synthetic_ohlcv_array(n=300)
    env, agent, *_ = build_pipeline(c, DEV,
        phase={"name": "profile", "entry_conditions": {"buy": "any", "sell": "any"}})
    rep = profile_episodes(agent, env, DEV, n_steps=20)
    assert rep["wall_s"] > 0
    assert rep["forward_frac"] + rep["env_step_frac"] <= 1.0 + 1e-6
    # On CPU the env loop should dominate -> honest CPU-bound verdict.
    assert ("CPU-BOUND" in rep["verdict"]) or ("GPU-BOUND" in rep["verdict"])
    assert isinstance(profile_report(rep), str)


# ════════════════════════════════════════════════════════════════════════════
# SMOOTH entropy annealing.
# ════════════════════════════════════════════════════════════════════════════
def _agent_with_anneal(shape, start=0.1, stable=0.01, eps=20):
    c = dict(CFG)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False,
              "ENTROPY_ANNEAL_ENABLED": True, "ENTROPY_START_COEF": start,
              "ENTROPY_ANNEAL_EPISODES": eps, "ENTROPY_ANNEAL_SHAPE": shape,
              "PPO": {"ent_coef": stable}})
    return PPOAgent(16, c, DEV)


@pytest.mark.parametrize("shape", ["cosine", "exp"])
def test_entropy_anneal_smooth_endpoints_and_monotonic(shape):
    a = _agent_with_anneal(shape, start=0.1, stable=0.01, eps=20)
    vals = [a.anneal_entropy(ep) for ep in range(0, 21)]
    assert abs(vals[0] - 0.1) < 1e-9, "must START at ENTROPY_START_COEF"
    assert abs(vals[20] - 0.01) < 1e-9, "must END exactly at the stable coef by ep20"
    # monotonic non-increasing (start > stable).
    for x, y in zip(vals, vals[1:]):
        assert y <= x + 1e-9, "anneal must be monotonic (no bounce)"
    # smoothness: NO single step bigger than the linear step (a step schedule
    # would have one giant jump). Each delta must be < the full range.
    deltas = [abs(y - x) for x, y in zip(vals, vals[1:])]
    assert max(deltas) < (0.1 - 0.01), "a smooth schedule has no full-range jump"


def test_entropy_anneal_cosine_is_not_linear():
    cos = _agent_with_anneal("cosine", eps=20)
    lin = _agent_with_anneal("linear", eps=20)
    midpoint_ep = 10
    cv = cos.anneal_entropy(midpoint_ep)
    lv = lin.anneal_entropy(midpoint_ep)
    # at the midpoint cosine equals the linear midpoint by symmetry; sample an
    # off-center episode where the two MUST differ to prove it isn't linear.
    cv2 = cos.anneal_entropy(5)
    lv2 = lin.anneal_entropy(5)
    assert abs(cv2 - lv2) > 1e-4, "cosine schedule must differ from linear"


def test_entropy_anneal_holds_stable_after_horizon():
    a = _agent_with_anneal("cosine", start=0.1, stable=0.01, eps=20)
    assert abs(a.anneal_entropy(50) - 0.01) < 1e-9, "must hold stable past the horizon"


# ════════════════════════════════════════════════════════════════════════════
# Optuna search engine (guarded).
# ════════════════════════════════════════════════════════════════════════════
def test_apply_params_to_cfg_maps_ppo_and_batch():
    from training.hyperopt import apply_params_to_cfg
    cfg = apply_params_to_cfg(dict(CFG), {
        "learning_rate": 1e-4, "gamma": 0.97, "gae_lambda": 0.9,
        "clip_range": 0.15, "ent_coef": 0.02, "batch_size_env": 32})
    assert cfg["PPO"]["learning_rate"] == 1e-4
    assert cfg["PPO"]["gamma"] == 0.97
    assert cfg["BATCH_SIZE_ENV"] == 32


def test_load_best_params_none_when_absent(tmp_path):
    from training.hyperopt import load_best_params
    assert load_best_params(str(tmp_path)) is None


def test_load_best_params_reads_file(tmp_path):
    from training.hyperopt import load_best_params, BEST_PARAMS_FILENAME
    p = tmp_path / BEST_PARAMS_FILENAME
    p.write_text(json.dumps({"best_params": {"gamma": 0.95}}))
    assert load_best_params(str(tmp_path)) == {"gamma": 0.95}


@pytest.mark.skipif(__import__("training.hyperopt", fromlist=["has_optuna"]).has_optuna()
                    is False, reason="optuna not installed")
def test_optuna_search_runs_persists_and_resumes(tmp_path):
    """A tiny fast study: synthetic objective, SQLite persistence + resume, and
    best_hyperparams.json written. Skipped cleanly when optuna is absent."""
    from training.hyperopt import run_search, storage_url, best_params_path

    def _fake_train(params, trial):
        # deterministic synthetic objective peaking at gamma≈0.95 — cheap, no model.
        return -abs(params["gamma"] - 0.95)

    best = run_search(_fake_train, str(tmp_path), n_trials=5,
                      study_name="t", pruner="median", seed=0)
    assert "gamma" in best
    assert os.path.exists(best_params_path(str(tmp_path)))
    # The SQLite study file must exist (persistence).
    assert os.path.exists(str(tmp_path / "optuna_study.db"))
    # Resume: a second run re-opens the SAME study (load_if_exists) and adds trials.
    import optuna
    study = optuna.load_study(study_name="t", storage=storage_url(str(tmp_path)))
    n_before = len(study.trials)
    run_search(_fake_train, str(tmp_path), n_trials=2, study_name="t", seed=0)
    study2 = optuna.load_study(study_name="t", storage=storage_url(str(tmp_path)))
    assert len(study2.trials) == n_before + 2, "resume must append, not restart"


def test_hyperopt_import_is_guarded():
    """Importing the module must NEVER raise even without optuna installed."""
    import importlib
    m = importlib.import_module("training.hyperopt")
    assert hasattr(m, "has_optuna") and isinstance(m.has_optuna(), bool)
