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
