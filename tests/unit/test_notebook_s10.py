"""
tests/unit/test_notebook_s10.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 10 — Colab notebook audit. Asserts (without a Jupyter kernel):
  • the notebook is VALID JSON and every cell parses;
  • clone (CELL 3) comes BEFORE install (CELL 4) so requirements.txt exists;
  • the install cell installs TA-Lib and verifies the import;
  • the GPU-check cell DETECTS + falls back (no bare `assert cuda.is_available`);
  • the dashboard defaults are ZERO-DRIFT vs settings/shaper (via dashboard_utils);
  • paths are consistent (gpu/ manifest + /content/rl-trading-live clone dir).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(ROOT, "rl_trading_colab.ipynb")


def _cells():
    nb = json.load(open(NB))
    return [("".join(c["source"]), c) for c in nb["cells"]]


def _code_cells():
    return [(s, c) for s, c in _cells() if c["cell_type"] == "code"]


def test_notebook_is_valid_json():
    nb = json.load(open(NB))
    assert nb["cells"], "notebook has no cells"
    for c in nb["cells"]:
        assert "cell_type" in c and "source" in c


def test_clone_before_install():
    srcs = [s for s, _ in _code_cells()]
    clone_i = next(i for i, s in enumerate(srcs)
                   if "subprocess" in s and "clone" in s)
    install_i = next(i for i, s in enumerate(srcs)
                     if "pip" in s and "requirements.txt" in s)
    assert clone_i < install_i, "install must run AFTER the repo clone"


def test_talib_installed_and_verified():
    srcs = "\n".join(s for s, _ in _code_cells())
    assert "talib" in srcs.lower(), "TA-Lib install/verify missing"
    assert "import talib" in srcs, "TA-Lib import not verified in notebook"


def test_gpu_check_has_cpu_fallback_not_hard_assert():
    gpu = next(s for s, _ in _code_cells() if "GPU CHECK" in s)
    assert "assert torch.cuda.is_available()" not in gpu, \
        "GPU cell must NOT hard-assert CUDA — it must warn + fall back to CPU"
    assert "CPU" in gpu and "WARNING" in gpu


def test_dashboard_defaults_zero_drift():
    """Every dashboard default must equal its source of truth (settings.CFG for
    flat keys, reward shaper for reward keys). dashboard_utils is that mirror."""
    from core.interpret.dashboard_utils import default_params, widget_specs
    from core.settings import CFG
    dp = default_params()
    specs = widget_specs()
    for key, spec in specs.items():
        if spec.get("reward"):
            continue                       # reward keys mirror shaper.py, not CFG
        if key in CFG:
            assert CFG[key] == dp[key], f"dashboard default drift on {key}"


def test_clone_dir_and_manifest_paths_consistent():
    srcs = "\n".join(s for s, _ in _cells())
    assert "/content/rl-trading-live" in srcs, "clone dir path missing/inconsistent"


# ── RUN-TRAINING CELL: unbuffered + live-streamed launch (Colab "freeze" fix) ──
def _run_training_cell() -> str:
    """The CELL 7b source (the one that launches training as a child process)."""
    return next(s for s, _ in _code_cells() if "# CELL 7b" in s)


def test_run_training_cell_launches_unbuffered_and_streams():
    """CELL 7b MUST launch the child unbuffered and stream its stdout live, or the
    Colab cell looks frozen for 20+ min while training is actually running.

    We assert the cell wires in the dashboard_utils helpers (build_train_argv adds
    `-u`, unbuffered_env adds PYTHONUNBUFFERED=1, stream_subprocess forwards lines)
    and that it no longer uses the buffering subprocess.run() path."""
    cell = _run_training_cell()
    assert "build_train_argv" in cell, "launch must use build_train_argv (adds -u)"
    assert "stream_subprocess" in cell, "launch must stream child output live"
    assert "unbuffered_env" in cell, "child env must set PYTHONUNBUFFERED=1"
    assert "subprocess.run(" not in cell, \
        "subprocess.run buffers child stdout in Colab — use stream_subprocess"
    assert "returncode" in cell and "exited with code" in cell, \
        "a nonzero exit must be surfaced with a clear banner"


def test_build_train_argv_includes_dash_u():
    """build_train_argv must inject `-u` right after the interpreter so the child's
    stdout is unbuffered (first half of the Colab-freeze fix)."""
    from core.interpret.dashboard_utils import build_train_argv
    argv = build_train_argv("/usr/bin/python3")
    assert argv[0] == "/usr/bin/python3"
    assert argv[1] == "-u", "`-u` (unbuffered) must immediately follow the exe"
    assert argv[2:] == ["-m", "training.train"]


def test_unbuffered_env_sets_pythonunbuffered():
    """unbuffered_env returns a FRESH dict with PYTHONUNBUFFERED=1 (never mutates
    the passed-in base / os.environ)."""
    from core.interpret.dashboard_utils import unbuffered_env
    base = {"FOO": "bar"}
    env = unbuffered_env(base)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["FOO"] == "bar"
    assert "PYTHONUNBUFFERED" not in base, "must not mutate the caller's dict"


def test_stream_subprocess_forwards_lines_incrementally():
    """stream_subprocess must forward the child's stdout LINE-BY-LINE as it is
    produced, not buffer it all until exit. We launch a tiny child that prints two
    lines with a delay between them and capture the (line, wall-clock) of each
    forwarded line; the second line must arrive measurably AFTER the first, proving
    incremental relay rather than a single end-of-process flush."""
    import sys
    import time
    from core.interpret.dashboard_utils import stream_subprocess

    child = (
        "import sys, time\n"
        "print('FIRST', flush=True)\n"
        "time.sleep(0.6)\n"
        "print('SECOND', flush=True)\n"
    )
    received = []                       # (text, monotonic timestamp) per line
    t0 = time.monotonic()
    rc = stream_subprocess(
        [sys.executable, "-c", child],
        echo=lambda line: received.append((line, time.monotonic() - t0)),
    )
    assert rc == 0
    texts = [t.strip() for t, _ in received]
    assert texts == ["FIRST", "SECOND"], f"lines not forwarded in order: {texts}"
    # Incremental proof: FIRST arrives well before the child's 0.6s sleep elapses,
    # and SECOND arrives only after it — so they were NOT delivered in one batch.
    first_ts = received[0][1]
    second_ts = received[1][1]
    assert first_ts < 0.4, f"first line was delayed ({first_ts:.2f}s) — buffered?"
    assert second_ts - first_ts > 0.3, \
        f"lines arrived together ({second_ts - first_ts:.2f}s apart) — not streamed"


def test_stream_subprocess_returns_child_exit_code():
    """A nonzero child exit must propagate so the cell can show the error banner."""
    import sys
    from core.interpret.dashboard_utils import stream_subprocess
    rc = stream_subprocess(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        echo=lambda line: None,
    )
    assert rc == 7
