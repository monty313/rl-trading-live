"""
scripts/validate_phases_yaml.py
────────────────────────────────────────────────────────────────────────────
Schema-check config/phases.yaml. For every phase: confirm required fields are
present and every condition variable is in conditions_engine.VARIABLE_REGISTRY.

On an unknown variable, print the exact IRAC remediation. Exit 0 if all phases
pass, 1 if any fail. Run after editing phases.yaml (also called by CELL 5 /
inspect_system.py).

    python scripts/validate_phases_yaml.py [--phases config/phases.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys

# Make repo root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from core.env import conditions_engine as CE  # noqa: E402

REQUIRED_FIELDS = ["name", "order", "instruments", "entry_conditions", "max_episodes"]
VALID_INSTRUMENTS = {"EURUSD", "GBPUSD", "XAUUSD", "US30"}


def validate_phase(phase: dict) -> list:
    """Return a list of IRAC error strings for one phase (empty = pass)."""
    errors = []
    name = phase.get("name", "?")
    for field in REQUIRED_FIELDS:
        if field not in phase:
            errors.append(
                f"**ISSUE**: Phase '{name}' missing required field '{field}'.\n"
                f"**RULE**: Each phase needs {REQUIRED_FIELDS}.\n"
                f"**APPLICATION**: Add '{field}:' to the phase block in phases.yaml.\n"
                f"**CONCLUSION**: Re-run validate_phases_yaml.py — phase should PASS."
            )
    for inst in phase.get("instruments", []) or []:
        if inst not in VALID_INSTRUMENTS:
            errors.append(
                f"**ISSUE**: Phase '{name}' uses unknown instrument '{inst}'.\n"
                f"**RULE**: instruments must be a subset of {sorted(VALID_INSTRUMENTS)}.\n"
                f"**APPLICATION**: Fix the instruments list in phases.yaml, or add "
                f"'{inst}' to INSTRUMENT_DATA_FILES + VALID_INSTRUMENTS.\n"
                f"**CONCLUSION**: Re-run validate_phases_yaml.py — phase should PASS."
            )
    ec = phase.get("entry_conditions", {}) or {}
    for side in ("buy", "sell"):
        cond = ec.get(side, "any")
        try:
            CE.validate_condition(cond, name)
        except CE.ConfigError as exc:
            errors.append(str(exc))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "phases.yaml"))
    args = ap.parse_args()

    with open(args.phases) as f:
        data = yaml.safe_load(f)
    phases = data.get("phases", []) if data else []

    n_pass = n_fail = 0
    for phase in phases:
        errs = validate_phase(phase)
        if errs:
            n_fail += 1
            print(f"❌ FAIL — phase '{phase.get('name','?')}'")
            for e in errs:
                print(e + "\n")
        else:
            n_pass += 1
            print(f"✅ PASS — phase '{phase.get('name','?')}'")

    print(f"\nphases.yaml: {len(phases)} phases validated — "
          f"{n_pass} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
