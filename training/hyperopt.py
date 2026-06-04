"""
training/hyperopt.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 11 — Optuna hyperparameter search (GUARDED import; Run-All-safe).

Searches the PPO knobs that matter most — learning rate, gamma, GAE lambda, clip
range, entropy coef, batch size — with a pruner (MedianPruner default, or
Hyperband) so unpromising trials die early and the whole sweep fits a T4/A100
session (~2h). The study PERSISTS to a SQLite file (default under the checkpoint
dir, which lives on Drive in Colab) so a crashed/timed-out sweep RESUMES by
re-opening the same study name + storage. The best params are written to
best_hyperparams.json, which training.train auto-loads on the next run.

GUARDED: `import optuna` is wrapped so importing this module never breaks a
Run-All when optuna is not installed — has_optuna() reports availability and
run_search() raises a clear, actionable error only if actually invoked without it.

The objective is injectable (train_fn) so the same search engine is unit-tested
with a fast synthetic objective and used for real by passing the training run.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    _HAVE_OPTUNA = True
except Exception:                                  # pragma: no cover - env without optuna
    optuna = None
    _HAVE_OPTUNA = False


BEST_PARAMS_FILENAME = "best_hyperparams.json"


def has_optuna() -> bool:
    return _HAVE_OPTUNA


# ── search space (single source of truth) ────────────────────────────────────
def suggest_params(trial) -> Dict:
    """The PPO search space. Edit HERE only — both the real sweep and tests use it."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.80, 0.99),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-3, 1e-1, log=True),
        "batch_size_env": trial.suggest_categorical("batch_size_env", [16, 32, 48, 64]),
    }


def _make_pruner(kind: str):
    kind = (kind or "median").lower()
    if kind == "hyperband":
        return HyperbandPruner()
    return MedianPruner(n_warmup_steps=2)


def storage_url(checkpoint_dir: str) -> str:
    """SQLite URL under the checkpoint dir (= Drive in Colab) so it persists."""
    path = os.path.join(checkpoint_dir, "optuna_study.db")
    return f"sqlite:///{path}"


def best_params_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, BEST_PARAMS_FILENAME)


def run_search(train_fn: Callable[[Dict, "optuna.Trial"], float],
               checkpoint_dir: str, n_trials: int = 20,
               study_name: str = "ppo_search", pruner: str = "median",
               timeout: Optional[float] = None, seed: int = 0) -> Dict:
    """Run/resume an Optuna study.

    train_fn(params, trial) -> objective value to MAXIMIZE (e.g. holdout pass rate).
    It should call trial.report(value, step) + check trial.should_prune() for the
    pruner to act. The study persists to SQLite under checkpoint_dir and is
    reopened (load_if_exists=True) so a re-run RESUMES rather than restarting.
    Writes best_hyperparams.json and returns the best params dict."""
    if not _HAVE_OPTUNA:
        raise RuntimeError(
            "optuna is not installed — `pip install optuna` to run the sweep. "
            "(Import is guarded so the rest of the pipeline is unaffected.)")
    os.makedirs(checkpoint_dir, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url(checkpoint_dir),
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=_make_pruner(pruner),
        load_if_exists=True,                       # RESUME a prior study
    )

    def _objective(trial):
        params = suggest_params(trial)
        return train_fn(params, trial)

    study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    best = dict(study.best_params)
    with open(best_params_path(checkpoint_dir), "w") as f:
        json.dump({"best_params": best, "best_value": study.best_value,
                   "n_trials": len(study.trials), "study_name": study_name},
                  f, indent=2)
    return best


def load_best_params(checkpoint_dir: str) -> Optional[Dict]:
    """Auto-load best_hyperparams.json if present (train.py calls this). Returns
    the best_params dict or None when no sweep has been run."""
    p = best_params_path(checkpoint_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("best_params")
    except Exception:
        return None


def apply_params_to_cfg(cfg: dict, params: Dict) -> dict:
    """Merge searched params onto a cfg: PPO knobs go under cfg['PPO'];
    batch_size_env maps to the top-level BATCH_SIZE_ENV the env reads."""
    cfg = dict(cfg)
    ppo = dict(cfg.get("PPO", {}) or {})
    for k in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef"):
        if k in params:
            ppo[k] = params[k]
    cfg["PPO"] = ppo
    if "batch_size_env" in params:
        cfg["BATCH_SIZE_ENV"] = int(params["batch_size_env"])
    return cfg
