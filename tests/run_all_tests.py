"""
tests/run_all_tests.py
────────────────────────────────────────────────────────────────────────────
COMPLETE SYSTEM TEST — one file that runs EVERY test in the tests/ tree (unit +
integration) in a single invocation, prints a clear summary, and exposes a
programmatic API so Jordan can read the results.

WHY THIS EXISTS:
  - One command to validate the whole system:   python tests/run_all_tests.py
  - inspect_system.py calls run_full_suite() to gate the build.
  - Jordan (read-only) calls get_last_results() / run_full_suite() to report
    test health in the dashboard WITHOUT writing code or modifying anything.

JORDAN ACCESS (read-only, honors HARD RULE 6):
  - jordan_summary()       -> short plain-English status string for the vitals card
  - get_last_results()     -> dict of the most recent run (cached in memory + JSON)
  - run_full_suite()       -> runs pytest programmatically, returns a result dict
  Jordan NEVER writes test files or fixes failures — it only reads these results
  and can raise an IRAC card (via irac_engine) suggesting what a human should fix.

USAGE:
  python tests/run_all_tests.py                 # run everything, exit 0/1
  python tests/run_all_tests.py --unit          # unit tests only
  python tests/run_all_tests.py --integration   # integration tests only
  python tests/run_all_tests.py --quiet         # summary only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Make the repo root importable so `from core...` and test fixtures resolve.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Where the latest machine-readable result is cached for Jordan to read.
RESULTS_JSON = os.path.join(_REPO_ROOT, "logs", "last_test_results.json")

# In-process cache so Jordan can read results without touching disk.
_LAST_RESULTS: dict | None = None


def _discover_targets(scope: str = "all") -> list:
    """Return the list of test directories/paths to run for the given scope."""
    unit = os.path.join(_THIS_DIR, "unit")
    integ = os.path.join(_THIS_DIR, "integration")
    if scope == "unit":
        return [unit]
    if scope == "integration":
        return [integ]
    return [unit, integ]


def run_full_suite(scope: str = "all", quiet: bool = False) -> dict:
    """
    Run the test suite programmatically via pytest and return a result dict:

        {
          "timestamp": ISO8601,
          "scope": "all" | "unit" | "integration",
          "exit_code": int,         # 0 = all green
          "passed": bool,
          "summary": str,           # human-readable one-liner
          "targets": [paths...]
        }

    Caches the result in memory and to logs/last_test_results.json so Jordan and
    the dashboard can read it. Safe to call repeatedly.
    """
    global _LAST_RESULTS
    try:
        import pytest  # imported here so the module loads even if pytest is absent
    except ImportError:
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope": scope, "exit_code": 2, "passed": False,
            "summary": "pytest not installed — run: pip install pytest",
            "targets": [],
        }
        _LAST_RESULTS = result
        return result

    targets = _discover_targets(scope)
    args = list(targets) + (["-q"] if quiet else ["-v"])
    # rootdir is the repo so conftest.py (path setup) is picked up
    args += ["--rootdir", _REPO_ROOT]

    exit_code = int(pytest.main(args))
    passed = exit_code == 0
    summary = (f"{'✅ ALL TESTS PASS' if passed else '❌ TEST FAILURES'} "
               f"(scope={scope}, exit={exit_code})")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "exit_code": exit_code,
        "passed": passed,
        "summary": summary,
        "targets": targets,
    }
    _LAST_RESULTS = result
    # Persist for Jordan/dashboard (best-effort; never crash the suite on IO error)
    try:
        os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:  # pragma: no cover
        pass
    return result


# ── Jordan read-only API ─────────────────────────────────────────────────────
def get_last_results() -> dict:
    """
    Return the most recent results. Prefers the in-memory cache; falls back to
    the JSON file written by the last run. Returns a 'never run' stub otherwise.
    Jordan uses this to report test health WITHOUT running anything.
    """
    if _LAST_RESULTS is not None:
        return _LAST_RESULTS
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                return json.load(f)
        except Exception:  # pragma: no cover
            pass
    return {"timestamp": None, "passed": None,
            "summary": "No test run recorded yet — run python tests/run_all_tests.py",
            "exit_code": None, "scope": "all", "targets": []}


def jordan_summary() -> str:
    """One-line plain-English status for Jordan's vitals card (read-only)."""
    r = get_last_results()
    if r.get("passed") is None:
        return "Tests: not run yet."
    when = r.get("timestamp", "?")
    return f"Tests: {r['summary']} — last run {when}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the complete rl-trading-live test suite.")
    ap.add_argument("--unit", action="store_true", help="run unit tests only")
    ap.add_argument("--integration", action="store_true",
                    help="run integration tests only")
    ap.add_argument("--quiet", action="store_true", help="summary output only")
    args = ap.parse_args()

    scope = "all"
    if args.unit:
        scope = "unit"
    elif args.integration:
        scope = "integration"

    print("=" * 70)
    print(f"  rl-trading-live — COMPLETE SYSTEM TEST  (scope={scope})")
    print("=" * 70)
    result = run_full_suite(scope=scope, quiet=args.quiet)
    print("\n" + result["summary"])
    print(f"  cached -> {RESULTS_JSON}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
