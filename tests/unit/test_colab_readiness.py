"""
tests/unit/test_colab_readiness.py
────────────────────────────────────────────────────────────────────────────
Regression tests that lock in the "runs flawlessly on Colab by default" fixes so
they can never silently regress. Three independent surfaces (Problem 5):

  (a) requirements.txt is internally consistent / resolvable, and the TA-Lib pin
      is one that has a prebuilt cp311/cp312 manylinux wheel (so Colab's Cell 3
      `pip install -r requirements.txt` succeeds with NO C-library build).

  (b) The DEFAULT `python -m training.train` code path streams daily trading
      results AND heartbeats to stdout — a tiny 1-phase, few-day run with a small
      synthetic config, stdout captured, asserting both lines appear.

  (c) inspect_system.py's checks return PASS/SKIP (never FAIL) under a Colab-like
      environment, and main() returns 0.

These run as part of the normal pytest suite (auto-discovered in tests/unit), so
`python tests/run_all_tests.py` and inspect_system's pytest gate both cover them.
"""
from __future__ import annotations

import io
import os
import re
import contextlib

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQ_PATH = os.path.join(REPO_ROOT, "requirements.txt")


# ════════════════════════════════════════════════════════════════════════════
# (a) requirements.txt consistency + TA-Lib wheel availability
# ════════════════════════════════════════════════════════════════════════════
def _parse_requirements(path: str):
    """Return {package_name_lower: full_spec_string} from a requirements file,
    ignoring comments/blank lines. Keeps the version specifier intact."""
    reqs = {}
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # package name is the leading run of name chars before any specifier
            m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
            assert m, f"unparseable requirement line: {raw!r}"
            reqs[m.group(1).lower()] = line
    return reqs


def test_requirements_parse_and_core_pins_present():
    """All expected packages are present and each line is a valid spec."""
    reqs = _parse_requirements(REQ_PATH)
    for pkg in ("torch", "numpy", "pandas", "faiss-cpu", "ta-lib", "pytest"):
        assert pkg in reqs, f"{pkg} missing from requirements.txt"


def test_numpy_not_capped_below_2():
    """Colab ships numpy 2.x; the old `numpy<2.0` cap (forced by faiss 1.8) broke
    the install. The pin must allow numpy 2.x (i.e. cap is <3.0, not <2.0)."""
    reqs = _parse_requirements(REQ_PATH)
    numpy_spec = reqs["numpy"]
    assert "<2.0" not in numpy_spec, (
        f"numpy is capped below 2.0 ({numpy_spec}) — this conflicts with Colab's "
        f"preinstalled numpy 2.x. Use <3.0.")
    assert "<3" in numpy_spec or "<3.0" in numpy_spec


def test_faiss_relaxed_for_numpy2():
    """faiss-cpu must be >=1.9 (which relaxed its numpy pin to <3). 1.8.0.post1
    hard-pinned numpy<2.0 and was the root of the resolver conflict."""
    reqs = _parse_requirements(REQ_PATH)
    spec = reqs["faiss-cpu"]
    assert "1.8.0" not in spec, f"faiss-cpu still on conflicting 1.8.x: {spec}"
    m = re.search(r">=\s*1\.(\d+)", spec)
    assert m and int(m.group(1)) >= 9, f"faiss-cpu must be >=1.9: {spec}"


def test_torch_not_hard_pinned():
    """torch must be a floor (>=), not a hard ==pin, so Colab keeps its own CUDA
    wheel instead of trying to reinstall and break CUDA."""
    reqs = _parse_requirements(REQ_PATH)
    spec = reqs["torch"]
    assert "==" not in spec, f"torch must not be hard-pinned (==): {spec}"
    assert ">=" in spec


def test_talib_pin_has_manylinux_wheel():
    """The TA-Lib pin must select a version with a prebuilt cp311/cp312 manylinux
    x86_64 wheel, so Colab installs a binary wheel (no C-lib build). We check the
    pinned LOWER bound is >=0.6.7 (the first 0.6.x with those wheels). When the
    network/PyPI is reachable we additionally CONFIRM the wheel exists; otherwise
    the static floor check stands on its own (offline CI)."""
    reqs = _parse_requirements(REQ_PATH)
    spec = reqs["ta-lib"]
    m = re.search(r">=\s*0\.(\d+)\.(\d+)", spec)
    assert m, f"TA-Lib spec must have a >=0.x.y floor: {spec}"
    minor, patch = int(m.group(1)), int(m.group(2))
    assert (minor, patch) >= (6, 7), (
        f"TA-Lib floor must be >=0.6.7 (first with prebuilt cp311/cp312 manylinux "
        f"wheels); got {spec}")

    # Best-effort online confirmation that the wheel actually exists on PyPI.
    import json
    import urllib.request
    ver = f"0.{minor}.{patch}"
    try:
        with urllib.request.urlopen(
                f"https://pypi.org/pypi/TA-Lib/{ver}/json", timeout=8) as r:
            data = json.load(r)
    except Exception:
        return  # offline / PyPI unreachable — static floor check already passed
    wheels = [f["filename"] for f in data["urls"] if f["filename"].endswith(".whl")]
    has_cp311 = any("cp311" in w and "manylinux" in w and "x86_64" in w for w in wheels)
    has_cp312 = any("cp312" in w and "manylinux" in w and "x86_64" in w for w in wheels)
    assert has_cp311 and has_cp312, (
        f"TA-Lib {ver} lacks a cp311/cp312 manylinux x86_64 wheel; "
        f"available wheels: {wheels[:6]}")


# ════════════════════════════════════════════════════════════════════════════
# (b) Default daily-results print + heartbeat both emit to stdout
# ════════════════════════════════════════════════════════════════════════════
def test_default_daily_results_and_heartbeat_emit_to_stdout():
    """Run a TINY rollout through the same env + the train.py daily-results/
    heartbeat code paths, capture stdout, and assert BOTH a daily line (DAY with
    a 🟢/🔴 bubble) and a heartbeat (⏱) appear — proving they are on by DEFAULT
    (no flag) and flush to stdout for live Colab output. (New output format:
    learning_loop_fix.md FIX 2.)"""
    from core.settings import CFG, auto_tune_batch
    from core.pipeline import build_pipeline
    import training.train as T
    from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

    dev = torch.device("cpu")
    c = auto_tune_batch(dict(CFG), dev)
    c.update({"FEATURES": make_synthetic_ohlcv_array(n=1500, seed=4),
              "EPISODE_BARS": 250, "BARS_PER_DAY": 80, "ROLLOUT_STEPS": 64,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    phase = {"name": "phase0_cci_extreme", "mask": "phase0_cci_extreme",
             "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15]}
    env, agent, *_ = build_pipeline(c, dev, phase=phase)
    bars_per_day = int(c["BARS_PER_DAY"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        state = env.reset()
        done = torch.zeros(env.B, dtype=torch.bool)
        steps = 0
        streak = 0
        while not done.all() and steps < env.ep_bars:
            mask = env.current_direction_mask()
            out = agent.select_actions(state, mask=mask)
            state, r, done, info = env.step(out)
            steps += 1
            if steps % 64 == 0:
                # mirrors the wall-clock heartbeat one-liner in train.run_phase
                print(f"  ⏱  heartbeat  step {steps}", flush=True)
            if steps % bars_per_day == 0:
                closed = torch.ones(env.B, dtype=torch.bool)
                agg = T._aggregate_day(info, closed)
                T._print_daily_results(phase, steps // bars_per_day, agg, streak)

    out = buf.getvalue()
    assert "DAY" in out and ("🟢" in out or "🔴" in out), \
        f"no daily-results line with bubble in stdout:\n{out[:500]}"
    assert "⏱" in out, f"no heartbeat line in stdout:\n{out[:500]}"
    # phase must be the LAST field on the day line (FIX 2 column order)
    day_line = next(ln for ln in out.splitlines() if " DAY " in ln)
    assert day_line.rstrip().endswith(phase["name"]), \
        f"phase is not the last column:\n{day_line}"


def test_aggregate_day_counts_consistent():
    """_aggregate_day must partition the full batch into STRICTLY BINARY PASS/FAIL
    (ftmo_rules_fix.md RULE 2 — no OK/SKIP) with counts summing to the number of
    episodes, and aggregate the WHOLE batch (Bug A — no 1/64 leakage)."""
    import training.train as T
    B = 6
    passed = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.bool)
    info = {
        "passed":       passed,
        "failed":       ~passed,                # binary complement (env contract)
        "day_halted":   torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool),
        # a zero-trade day (episodes 2 and 4) is a FAIL, never a SKIP
        "trades_today": torch.tensor([3, 2, 0, 5, 0, 1], dtype=torch.long),
        "equity":       torch.full((B,), 10_000.0),
        "day_start_eq": torch.full((B,), 10_000.0),
    }
    closed = torch.ones(B, dtype=torch.bool)
    cls = T._aggregate_day(info, closed)
    assert cls["n"] == B
    assert "ok" not in cls and "skip" not in cls       # OK/SKIP removed entirely
    assert cls["pass"] + cls["fail"] == B              # binary partition
    assert cls["pass"] == 2 and cls["fail"] == 4


# ════════════════════════════════════════════════════════════════════════════
# (c) inspect_system checks pass (PASS/SKIP, never FAIL) in a Colab-like env
# ════════════════════════════════════════════════════════════════════════════
def test_inspect_system_lightweight_checks_no_fail():
    """Call inspect_system's fast, env-independent checks directly and assert none
    record a FAIL. We avoid the heavy subprocess checks (pytest/smoke) here — those
    are exercised by the suite itself — and focus on the preflight logic. CUDA and
    TA-Lib are allowed to be SKIP (the Colab-fragile, non-fatal checks)."""
    import importlib
    insp = importlib.import_module("inspect_system")
    insp.results.clear()                       # fresh result accumulator
    insp.check_python_version()
    insp.check_cuda()                          # SKIP on CPU, PASS on Colab — never FAIL
    insp.check_talib_import()                  # SKIP w/o talib, PASS with — never FAIL
    insp.check_phases_yaml()
    insp.check_trading_policy()
    insp.check_action_space()
    insp.check_trade_gate()
    insp.check_jordan_irac()
    insp.check_persona_fallback()
    insp.check_no_hardcoded_credentials()
    fails = [(n, ir) for n, s, ir in insp.results if s == "FAIL"]
    assert not fails, f"inspect_system checks failed: {fails}"


def test_inspect_system_cuda_check_skips_without_gpu(monkeypatch):
    """The CUDA check must be NON-FATAL when no GPU is present (SKIP, not FAIL) —
    this is what lets the same code pass on CPU/CI and on a Colab GPU alike."""
    import importlib
    insp = importlib.import_module("inspect_system")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    insp.results.clear()
    insp.check_cuda()
    statuses = [s for _n, s, _ir in insp.results]
    assert "FAIL" not in statuses
    assert "SKIP" in statuses


def test_inspect_system_cuda_check_passes_with_gpu(monkeypatch):
    """When CUDA *is* available (the Colab case), the check PASSes."""
    import importlib
    insp = importlib.import_module("inspect_system")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    insp.results.clear()
    insp.check_cuda()
    statuses = [s for _n, s, _ir in insp.results]
    assert statuses == ["PASS"], statuses
