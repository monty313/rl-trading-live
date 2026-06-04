"""
core/interpret/dashboard_utils.py
────────────────────────────────────────────────────────────────────────────
PURE-PYTHON helpers behind the Colab dashboard cells (Tasks A/B/C). Factored out
of the notebook so every non-widget piece of logic is UNIT-TESTABLE without a
running Jupyter kernel or ipywidgets installed. The notebook cells import from
here and only own the thin widget wiring + display.

WHY THIS EXISTS (PART 7 of the spec): the dashboard's real logic — assembling
PARAMS from widget values, type-coercing/clamping a checkpoint cfg onto the
widget specs, sanitizing a snapshot filename, hashing PARAMS, and diffing PARAMS
against the hardcoded defaults — must be tested in the main suite. Keeping it in
a normal importable module (not buried in a .ipynb) is the only way to do that.

SOURCE OF TRUTH = THE CODE. Every default value + every widget min/max/step/
options below is mirrored from core/settings.py (CFG) and core/reward/shaper.py
at HEAD. The widget SPEC table is also the single place that drives:
  • which keys appear in the dashboard (Task A),
  • the type-safe coercion + clamp used when loading a checkpoint cfg (Task B),
  • the per-key default used for the snapshot diff (Task C).
If settings.py changes a default, change it HERE too (one place) and the
notebook + tests follow automatically.

────────────────────────────────────────────────────────────────────────────
WIDGET SPEC FORMAT
────────────────────────────────────────────────────────────────────────────
_WIDGET_SPECS maps a FLAT settings key -> a spec dict:
    {"kind": "float"|"floatlog"|"int"|"dropdown"|"checkbox",
     "default": <value>, "min": <n>, "max": <n>, "step": <n>,
     "options": [...], "group": "<section>", "label": "<human label>",
     "reward": <bool — True if it lives under CFG["REWARD"]>}

Reward keys carry reward=True so build_params() can nest them under PARAMS["REWARD"]
exactly as the train cell / shaper expect, while the flat _WIDGET_MAP in the
notebook can still address them by their bare key (matching how a checkpoint cfg's
"REWARD" sub-dict is unpacked in Task B).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION GROUPS (the six dashboard panels, Task A)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Ordered so the notebook renders panels top-to-bottom in this order. The emoji
# + title match the spec's six grouped section panels exactly.
SECTION_GROUPS: List[Tuple[str, str]] = [
    ("ftmo",     "🎯 FTMO Settings"),
    ("reward",   "🏆 5-Tier Reward Tuning"),
    ("streak",   "🔥 Streak System"),
    ("risk",     "🛡️ Risk Management"),
    ("training", "🚀 Training Config"),
    ("gpu",      "⚡ GPU Settings"),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WIDGET SPECS — mirrored 1:1 from core/settings.py + core/reward/shaper.py ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# NOTE: defaults below are COPIED from CFG / CFG["REWARD"] at commit 2166ec8. The
# tests assert these stay in sync with the live settings module so drift is caught.
_WIDGET_SPECS: Dict[str, Dict[str, Any]] = {
    # ── 1. 🎯 FTMO SETTINGS ────────────────────────────────────────────────
    "DAILY_TARGET_PCT": {"kind": "float", "default": 0.025, "min": 0.005,
                         "max": 0.10, "step": 0.001, "group": "ftmo",
                         "label": "Daily target (fraction of initial equity)"},
    "DAILY_MAX_DD_PCT": {"kind": "float", "default": 0.010, "min": 0.002,
                         "max": 0.05, "step": 0.001, "group": "ftmo",
                         "label": "Daily max trailing DD (fraction)"},
    "ACCOUNT_SIZE":     {"kind": "dropdown", "default": 10_000.0,
                         "options": [10_000.0, 25_000.0, 50_000.0, 100_000.0],
                         "group": "ftmo", "label": "Account size ($)"},
    "RANDOMIZE_FTMO_INPUTS": {"kind": "checkbox", "default": False,
                              "group": "ftmo",
                              "label": "Randomize FTMO target/DD per episode"},
    "BEAST_MODE":       {"kind": "checkbox", "default": False, "group": "ftmo",
                         "label": "Beast mode (lift lot-curriculum narrowing)"},

    # ── 2. 🏆 5-TIER REWARD TUNING (all CFG["REWARD"] keys -> reward=True) ──
    "pass_day_bonus":   {"kind": "float", "default": 2.0, "min": 0.0, "max": 10.0,
                         "step": 0.1, "group": "reward", "reward": True,
                         "label": "PASS day bonus (>=100% of target)"},
    "fail_day_penalty": {"kind": "float", "default": -2.0, "min": -10.0, "max": 0.0,
                         "step": 0.1, "group": "reward", "reward": True,
                         "label": "FAIL day penalty (<50% of target)"},
    "ok_partial_lo":    {"kind": "float", "default": 0.25, "min": 0.0, "max": 1.0,
                         "step": 0.01, "group": "reward", "reward": True,
                         "label": "OK partial credit at 50% of target"},
    "ok_partial_hi":    {"kind": "float", "default": 0.95, "min": 0.0, "max": 1.0,
                         "step": 0.01, "group": "reward", "reward": True,
                         "label": "OK partial credit just under 100%"},
    "exceed_scale":     {"kind": "float", "default": 1.0, "min": 0.0, "max": 5.0,
                         "step": 0.1, "group": "reward", "reward": True,
                         "label": "EXCEED progressive bonus scale (no cap)"},
    "survival_bonus":   {"kind": "float", "default": 1.5, "min": 0.0, "max": 5.0,
                         "step": 0.1, "group": "reward", "reward": True,
                         "label": "SURVIVAL bonus (traded all day, no breach)"},
    "red_day_scale":    {"kind": "float", "default": 1.0, "min": 0.0, "max": 5.0,
                         "step": 0.1, "group": "reward", "reward": True,
                         "label": "Red-day linear penalty scale"},
    "dd_efficiency_weight": {"kind": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                             "step": 0.01, "group": "reward", "reward": True,
                             "label": "DD-efficiency multiplier weight"},

    # ── 3. 🔥 STREAK SYSTEM ────────────────────────────────────────────────
    "streak_curve_a":   {"kind": "float", "default": 0.616998, "min": 0.0, "max": 5.0,
                         "step": 0.001, "group": "streak", "reward": True,
                         "label": "Streak curve amplitude a"},
    "streak_curve_b":   {"kind": "float", "default": 0.221749, "min": 0.0, "max": 2.0,
                         "step": 0.001, "group": "streak", "reward": True,
                         "label": "Streak curve growth rate b"},
    "streak_base":      {"kind": "float", "default": 0.5, "min": 0.0, "max": 3.0,
                         "step": 0.05, "group": "streak", "reward": True,
                         "label": "Day-1 flat streak base"},
    "negative_streak_mult": {"kind": "float", "default": 1.5, "min": 1.0, "max": 3.0,
                             "step": 0.1, "group": "streak", "reward": True,
                             "label": "Negative-streak mirror multiplier"},
    "mulligan_count":   {"kind": "int", "default": 1, "min": 0, "max": 5, "step": 1,
                         "group": "streak", "reward": True,
                         "label": "Mulligan count (free fails per streak)"},
    "recovery_bonus":   {"kind": "float", "default": 3.0, "min": 0.0, "max": 10.0,
                         "step": 0.1, "group": "streak", "reward": True,
                         "label": "Recovery bonus (PASS breaks a fail streak)"},
    "momentum_bonus":   {"kind": "float", "default": 0.2, "min": 0.0, "max": 2.0,
                         "step": 0.05, "group": "streak", "reward": True,
                         "label": "Momentum bonus (day after a pass)"},
    "PASS_NO_BREACH_BONUS": {"kind": "float", "default": 0.01, "min": 0.0, "max": 1.0,
                             "step": 0.01, "group": "streak",
                             "label": "Pass-rate / no-breach bonus (PASS_RATE_THRESHOLD)"},
    "PHASE_ADVANCE_STREAK": {"kind": "int", "default": 10, "min": 1, "max": 60,
                             "step": 1, "group": "streak",
                             "label": "Phase-advance streak (consecutive passes)"},

    # ── 4. 🛡️ RISK MANAGEMENT ──────────────────────────────────────────────
    "MAX_LOT":          {"kind": "float", "default": 2.0, "min": 0.1, "max": 5.0,
                         "step": 0.05, "group": "risk",
                         "label": "MAX_LOT (head ceiling; curriculum clamps within)"},
    "BARS_PER_DAY":     {"kind": "int", "default": 1440, "min": 60, "max": 1440,
                         "step": 60, "group": "risk", "label": "Bars per trading day"},
    "EPISODE_BARS":     {"kind": "int", "default": 43_200, "min": 1440, "max": 200_000,
                         "step": 1440, "group": "risk", "label": "Episode length (bars)"},
    "LOOKBACK":         {"kind": "int", "default": 20, "min": 5, "max": 100, "step": 1,
                         "group": "risk", "label": "Lookback window (bars of history)"},
    "SPEED_BONUS_MINUTES": {"kind": "int", "default": 3, "min": 1, "max": 60, "step": 1,
                            "group": "risk", "label": "Speed-bonus window (minutes)"},
    "speed_bonus":      {"kind": "float", "default": 0.3, "min": 0.0, "max": 2.0,
                         "step": 0.05, "group": "risk", "reward": True,
                         "label": "Speed-bonus magnitude"},
    "intraday_progress_scale": {"kind": "float", "default": 0.5, "min": 0.0, "max": 2.0,
                                "step": 0.05, "group": "risk", "reward": True,
                                "label": "Intra-day progress pull scale"},
    "cross_day_giveback_scale": {"kind": "float", "default": 0.5, "min": 0.0, "max": 2.0,
                                 "step": 0.05, "group": "risk", "reward": True,
                                 "label": "Cross-day give-back penalty scale"},

    # ── 5. 🚀 TRAINING CONFIG ──────────────────────────────────────────────
    "LR":               {"kind": "floatlog", "default": 3e-4, "min": 1e-6, "max": 1e-2,
                         "step": 0.1, "group": "training",
                         "label": "Learning rate (log scale)"},
    "BATCH_SIZE_ENV":   {"kind": "dropdown", "default": 64,
                         "options": [4, 16, 32, 48, 64, 128],
                         "group": "training", "label": "Parallel envs (auto-tuned by GPU)"},
    "ROLLOUT_STEPS":    {"kind": "int", "default": 2048, "min": 64, "max": 8192,
                         "step": 64, "group": "training",
                         "label": "Rollout steps per PPO update"},
    "PPO_EPOCHS":       {"kind": "int", "default": 4, "min": 1, "max": 20, "step": 1,
                         "group": "training", "label": "PPO epochs per update"},
    "GAMMA":            {"kind": "float", "default": 0.95, "min": 0.80, "max": 0.999,
                         "step": 0.001, "group": "training", "label": "Discount γ"},
    "GAE_LAMBDA":       {"kind": "float", "default": 0.95, "min": 0.80, "max": 0.999,
                         "step": 0.001, "group": "training", "label": "GAE λ"},
    "CLIP_EPS":         {"kind": "float", "default": 0.2, "min": 0.05, "max": 0.5,
                         "step": 0.01, "group": "training", "label": "PPO clip ε"},
    "ENTROPY_START_COEF": {"kind": "floatlog", "default": 0.10, "min": 1e-3, "max": 1.0,
                           "step": 0.1, "group": "training",
                           "label": "Entropy start coef (annealed, log scale)"},
    "ENTROPY_ANNEAL_EPISODES": {"kind": "int", "default": 20, "min": 1, "max": 200,
                                "step": 1, "group": "training",
                                "label": "Entropy anneal episodes"},
    "MAX_EPISODES_PER_PHASE": {"kind": "int", "default": 500, "min": 1, "max": 10_000,
                               "step": 10, "group": "training",
                               "label": "Max episodes per phase (N_EPISODES)"},
    "LOT_CURRICULUM_ENABLED": {"kind": "checkbox", "default": True, "group": "training",
                               "label": "Curriculum (narrow→wide lot clamp)"},
    "START_PHASE":      {"kind": "dropdown", "default": 0,
                         "options": [0, 1, 2, 3, 4, 5, 6, 7],
                         "group": "training",
                         "label": "Start strategy phase (index into phases.yaml)"},

    # ── 6. ⚡ GPU SETTINGS ─────────────────────────────────────────────────
    "AUTO_TUNE_GPU":    {"kind": "checkbox", "default": True, "group": "gpu",
                         "label": "Auto-tune batch/rollout to GPU tier"},
    "USE_AMP":          {"kind": "checkbox", "default": True, "group": "gpu",
                         "label": "Mixed precision (AMP, CUDA only)"},
    # DEFAULT OFF — mirrors CFG["USE_TORCH_COMPILE"]=False (no_holdups_default.md):
    # training is CPU-bound so compile's steady-state win is marginal while its
    # ~10-15 min first-step warmup is the biggest startup hold-up. Off = instant
    # start; CHECK this to claim the steady-state speedup (then warmup applies).
    "USE_TORCH_COMPILE": {"kind": "checkbox", "default": False, "group": "gpu",
                          "label": "torch.compile model (CUDA only) — OFF by "
                                   "default for instant start (training is "
                                   "CPU-bound). Check for steady-state speedup; "
                                   "first step then warms up ~10-15 min on A100"},
    "COMPILE_WATCHDOG_ENABLED": {"kind": "checkbox", "default": True, "group": "gpu",
                                 "label": "Compile watchdog (prints 'still "
                                          "compiling…' during torch.compile warmup)"},
    "HEARTBEAT_SECS":   {"kind": "int", "default": 300, "min": 10, "max": 1800,
                         "step": 10, "group": "gpu",
                         "label": "Heartbeat interval (s) — wall-clock liveness "
                                  "one-liner in the rollout loop"},
    "GPU_UTIL_TARGET":  {"kind": "float", "default": 0.80, "min": 0.5, "max": 1.0,
                         "step": 0.05, "group": "gpu",
                         "label": "Target GPU utilization fraction"},
}


def widget_specs() -> Dict[str, Dict[str, Any]]:
    """Return the (copied) widget-spec table so callers cannot mutate the module
    global by accident. The notebook builds one widget per entry."""
    return {k: dict(v) for k, v in _WIDGET_SPECS.items()}


def default_params() -> Dict[str, Any]:
    """The full DEFAULTS dict (flat key -> default value) drawn straight from the
    widget specs. Used as the Task-C diff baseline (_DEFAULTS) and by tests to
    assert the dashboard mirrors settings.py."""
    return {k: spec["default"] for k, spec in _WIDGET_SPECS.items()}


def reward_keys() -> List[str]:
    """Flat keys that belong under PARAMS['REWARD'] (reward=True in the spec)."""
    return [k for k, s in _WIDGET_SPECS.items() if s.get("reward")]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  build_params() — assemble the nested PARAMS dict from a flat values map   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def build_params(values: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the PARAMS dict the train cell consumes from a FLAT {key: value}
    map (the notebook passes {key: widget.value}). Reward keys are nested under
    PARAMS['REWARD'] exactly as the shaper / env expect; everything else stays a
    top-level key. Missing keys fall back to the spec default so the result is
    always complete (Run-All safe even before any widget is touched).

    The returned dict is the SINGLE contract the train cell maps onto CLI flags
    (see notebook Cell 7 / params_to_cli)."""
    out: Dict[str, Any] = {}
    reward: Dict[str, Any] = {}
    for key, spec in _WIDGET_SPECS.items():
        val = values.get(key, spec["default"])
        if spec.get("reward"):
            reward[key] = val
        else:
            out[key] = val
    out["REWARD"] = reward
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Type-safe coercion + clamp (Task B checkpoint-cfg apply)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def coerce_and_clamp(key: str, raw: Any) -> Tuple[Any, bool, bool]:
    """Coerce + clamp a raw cfg value to the widget's type/range for `key`.

    Returns (value, clamped, skipped):
      • float / floatlog -> float, clamped to [min, max]   (clamped=True if moved).
      • int              -> int,   clamped to [min, max]   (clamped=True if moved).
      • dropdown         -> kept only if in options, else SKIPPED (skipped=True).
      • checkbox         -> bool() of the value.
    Unknown keys are skipped. The notebook uses `clamped` to add a ⚠️ note and
    `skipped` to leave the widget untouched + warn (Task B coercion table)."""
    spec = _WIDGET_SPECS.get(key)
    if spec is None:
        return (None, False, True)
    kind = spec["kind"]
    if kind in ("float", "floatlog"):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return (None, False, True)
        lo, hi = float(spec["min"]), float(spec["max"])
        cl = min(max(v, lo), hi)
        return (cl, cl != v, False)
    if kind == "int":
        try:
            v = int(round(float(raw)))
        except (TypeError, ValueError):
            return (None, False, True)
        lo, hi = int(spec["min"]), int(spec["max"])
        cl = min(max(v, lo), hi)
        return (cl, cl != v, False)
    if kind == "dropdown":
        opts = spec["options"]
        # Coerce numeric-looking values to the option dtype so 64 == 64.0 matches.
        for o in opts:
            try:
                if type(o)(raw) == o or raw == o:
                    return (o, False, False)
            except (TypeError, ValueError):
                continue
        return (None, False, True)        # not an allowed option -> skip + warn
    if kind == "checkbox":
        return (bool(raw), False, False)
    return (None, False, True)


def apply_checkpoint_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Pure core of Task B's _apply_checkpoint_cfg: take a checkpoint cfg dict
    (flat keys, plus an optional 'REWARD' sub-dict whose keys are treated as
    top-level), coerce/clamp each value against the widget specs, and return a
    structured result the notebook turns into widget .value sets + a diff table:

        {"applied": {key: value, ...},     # coerced values to write to widgets
         "clamped": [key, ...],            # keys whose value was clamped (⚠️)
         "skipped": [(key, reason), ...]}  # keys not applied (unknown / bad opt)

    No widgets or display here — the notebook owns those. This makes the whole
    coercion path unit-testable (PART 7)."""
    flat: Dict[str, Any] = {}
    for k, v in (cfg_dict or {}).items():
        if k == "REWARD" and isinstance(v, dict):
            flat.update(v)                  # unpack nested reward onto top level
        else:
            flat[k] = v
    applied: Dict[str, Any] = {}
    clamped: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for k, v in flat.items():
        if k not in _WIDGET_SPECS:
            skipped.append((k, "no matching widget"))
            continue
        val, was_clamped, was_skipped = coerce_and_clamp(k, v)
        if was_skipped:
            skipped.append((k, "value not coercible / not an allowed option"))
            continue
        applied[k] = val
        if was_clamped:
            clamped.append(k)
    return {"applied": applied, "clamped": clamped, "skipped": skipped}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Snapshot helpers (Task C)                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def sanitize_label(label: str) -> str:
    """Sanitize a free-text run label for use in a filename (Task C): spaces ->
    underscores, strip anything that is not alnum/underscore/dash, collapse repeats,
    and fall back to 'unnamed' when the result is empty (e.g. an all-special label)."""
    s = (label or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    s = re.sub(r"_+", "_", s).strip("_-")
    return s or "unnamed"


def params_hash(params: Dict[str, Any]) -> str:
    """md5(json.dumps(params, sort_keys=True))[:8] — the stable content hash used
    to (a) name a snapshot, (b) detect duplicate snapshots, and (c) MATCH a
    snapshot to its training results (PART 1). Deterministic across runs because
    of sort_keys=True."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:8]


def diff_from_defaults(params: Dict[str, Any],
                       defaults: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    """Return {key: {"default": d, "saved": s}} for every FLAT key whose value
    differs from the defaults, else {} (Task C diff_from_defaults). Reward keys
    are compared at their flat name (unpacked from PARAMS['REWARD']). `defaults`
    falls back to the module's hardcoded defaults so the snapshot cell can supply
    its own independent _DEFAULTS copy."""
    defaults = defaults if defaults is not None else default_params()
    flat = dict(params)
    rew = flat.pop("REWARD", {}) or {}
    flat.update(rew)
    out: Dict[str, Dict[str, Any]] = {}
    for key, dval in defaults.items():
        if key not in flat:
            continue
        sval = flat[key]
        if sval != dval:
            out[key] = {"default": dval, "saved": sval}
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMS -> training CLI flags (the train-cell contract)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The existing train cell builds argv from a flat {flag: value} dict (store_true
# flags appear only when True; valued flags only when not None). The dashboard
# PARAMS dict is RICHER (it holds reward + GPU + training knobs too), but only a
# SUBSET maps to training/train.py CLI flags today — the rest are written into the
# snapshot for provenance + consumed by core/settings.py at import. This function
# extracts EXACTLY the flags train.py accepts, preserving the original contract.
_PARAM_TO_FLAG = {
    "ACCOUNT_SIZE":          "account-size",
    "DAILY_TARGET_PCT":      "target-pct",
    "DAILY_MAX_DD_PCT":      "max-dd-pct",
    "PHASE_ADVANCE_STREAK":  "phase-advance-streak",
    "START_PHASE":           "start-phase",
    "RANDOMIZE_FTMO_INPUTS": "randomize-ftmo",          # store_true
}
_STORE_TRUE_FLAGS = {"randomize-ftmo", "randomize-ftmo-account"}


def params_to_cli(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map the dashboard PARAMS dict onto the FLAT {flag: value} dict the existing
    train cell already turns into argv (keeping that contract intact). Only keys
    train.py actually accepts are emitted; account-size/target/DD default to None
    when at their settings.py default so the CLI omits them (train uses the cfg
    default), matching the notebook's existing 'override only when changed' logic."""
    out: Dict[str, Any] = {}
    defs = default_params()
    for pkey, flag in _PARAM_TO_FLAG.items():
        val = params.get(pkey, defs.get(pkey))
        if flag in _STORE_TRUE_FLAGS:
            out[flag] = bool(val)
            continue
        # Emit the value only when it DIFFERS from the settings default, so an
        # untouched dashboard reproduces the existing "trained defaults" launch.
        if val != defs.get(pkey):
            out[flag] = val
        else:
            out[flag] = None
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVATION FEATURE NAMES (shared by the interpret modules)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The ordered names of the v1/v2/v3 APPENDED feature blocks (the ones with
# human-meaningful labels). The leading lkbk*F window features are the normalized
# indicator window; we name them indicator@lag so saliency/SHAP still rank them.
_POSITION_FEAT_NAMES = [
    "position", "unrealised_pnl", "equity_change", "target_gap",
    "dd_headroom", "daily_return",
]
_FTMO_FEAT_NAMES = [
    "ftmo_target_pct", "ftmo_max_dd_pct", "day_difficulty",
    "progress_to_target", "dd_headroom_frac", "fraction_day_remaining",
    "account_size_log",
]
_SESSION_FEAT_NAMES = [
    "cest_time_of_day", "session_progress_to_target", "remaining_time_in_day",
    "session_code", "dd_budget_remaining", "signed_streak", "commission_cost",
]


def obs_feature_names(lookback: int, n_indicator_features: int,
                      indicator_columns: List[str] = None) -> List[str]:
    """Build the FULL ordered list of obs feature names matching env._get_state():
        [<lkbk*F window names>, <6 position>, <7 ftmo>, <7 session>]
    so saliency / SHAP can attach a real name to every input dimension. The window
    block is named '<indicator>@t-<lag>' (lag 0 == current bar). When the indicator
    column list is unavailable, falls back to 'feat<j>@t-<lag>'. Total length ==
    lookback*n_indicator_features + 20."""
    names: List[str] = []
    cols = indicator_columns or [f"feat{j}" for j in range(n_indicator_features)]
    # Window is built oldest->newest (offsets lkbk-1 .. 0), reshaped row-major as
    # (lkbk, F) -> flattened, so name[i*F + j] == cols[j] @ lag (lkbk-1-i).
    for i in range(lookback):
        lag = lookback - 1 - i
        for j in range(n_indicator_features):
            col = cols[j] if j < len(cols) else f"feat{j}"
            names.append(f"{col}@t-{lag}")
    names += list(_POSITION_FEAT_NAMES)
    names += list(_FTMO_FEAT_NAMES)
    names += list(_SESSION_FEAT_NAMES)
    return names


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LIVE SUBPROCESS STREAMING (Colab "frozen launch" fix)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# WHY THIS EXISTS — read before "simplifying" it back to subprocess.run():
#
#   The RUN-TRAINING notebook cell (CELL 7b) launches training as a CHILD process
#   so it can consume the dashboard PARAMS dict. The naive `subprocess.run(argv)`
#   has a nasty failure mode IN COLAB specifically:
#
#     • CPython block-buffers stdout when it is NOT a TTY. Under Colab the child's
#       stdout is a PIPE/redirected stream, not a terminal, so the child fills an
#       internal ~8 KB buffer before ANY bytes are flushed to the notebook.
#     • Training's startup (CSV load, feature build, torch.compile warmup) emits
#       only a few hundred bytes before it blocks for many minutes, so the buffer
#       never fills and the cell shows ONLY the parent's "Launching: …" line.
#     • Result: 20+ minutes of apparent FREEZE (GPU RAM idle) even though training
#       is running fine — indistinguishable, on screen, from a real crash.
#
#   The robust cure is belt-and-suspenders:
#     (a) UNBUFFER THE CHILD: run it with `python -u` AND PYTHONUNBUFFERED=1 in its
#         env, so every print reaches the pipe immediately (covers the C-level and
#         the Python-level buffering paths; either alone can be defeated).
#     (b) STREAM THE PIPE: read the child's merged stdout/stderr line-by-line and
#         re-emit each line through the parent immediately (flush=True), so the
#         notebook shows [train] startup lines within SECONDS of launch.
#
#   `build_train_argv()` enforces (a) by injecting "-u"; `stream_subprocess()`
#   enforces both (a, via env) and (b, via Popen + line iteration). They are unit
#   tested in tests/unit/test_notebook_s10.py so a future edit can't silently
#   regress the streaming behaviour.

def build_train_argv(python_exe: str, module: str = "training.train") -> List[str]:
    """Return the leading argv for launching a training module UNBUFFERED.

    Always inserts the ``-u`` flag (force-unbuffered stdio in the child) right
    after the interpreter, so the very first [train] prints reach the notebook
    immediately instead of sitting in a block buffer. The caller appends the
    PATH/RESUME/dashboard flags after this prefix."""
    return [python_exe, "-u", "-m", module]


def unbuffered_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of ``base`` (defaults to os.environ) with PYTHONUNBUFFERED=1.

    Second half of the belt-and-suspenders unbuffering: even if a child somehow
    re-enables buffering, the env var forces line/stream flushing. Returned as a
    fresh dict so callers never mutate os.environ in place."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stream_subprocess(
    argv: Sequence[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    echo: Callable[[str], None] = None,
) -> int:
    """Launch ``argv`` and FORWARD its output line-by-line as it arrives.

    Merges the child's stderr into stdout (stderr=STDOUT) and iterates the pipe
    so each line is re-emitted the instant the child flushes it — giving live
    output in Colab where ``subprocess.run`` would buffer for minutes. Forces
    PYTHONUNBUFFERED=1 in the child env (via :func:`unbuffered_env`) as the second
    unbuffering guard. Returns the child's exit code (the caller decides how to
    surface a nonzero result).

    ``echo`` defaults to ``print(line, end="", flush=True)`` — kept injectable so
    the unit test can capture forwarded lines and assert they arrive incrementally
    rather than all at once at the end."""
    if echo is None:
        def echo(line: str) -> None:           # default: forward to the cell live
            print(line, end="", flush=True)

    child_env = unbuffered_env(env)
    # bufsize=1 + text=True => line-buffered text pipe on the PARENT side; combined
    # with the child's -u/PYTHONUNBUFFERED this yields true line-at-a-time relay.
    proc = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    # Iterating proc.stdout yields each line as the child flushes it. We keep the
    # trailing newline (echo uses end="") so output is byte-faithful to the child.
    assert proc.stdout is not None
    for line in proc.stdout:
        echo(line)
    proc.stdout.close()
    return proc.wait()
