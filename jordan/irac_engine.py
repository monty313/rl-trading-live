"""
jordan/irac_engine.py
────────────────────────────────────────────────────────────────────────────
IRAC (Issue / Rule / Application / Conclusion) generator — TEXT ONLY.
Jordan NEVER writes files here (HARD RULE 6) — generate_irac returns markdown.

generate_irac(event_type, event_data) -> markdown string with the four sections.
Supported event_types: flatline_detected, dd_breach, slippage_spike,
pass_rate_drop, import_error, test_failure (and a generic fallback).
"""
from __future__ import annotations


def _block(issue, rule, application, conclusion) -> str:
    return (f"**ISSUE**: {issue}\n\n"
            f"**RULE**: {rule}\n\n"
            f"**APPLICATION**: {application}\n\n"
            f"**CONCLUSION**: {conclusion}")


def generate_irac(event_type: str, event_data: dict) -> str:
    d = event_data or {}
    if event_type == "flatline_detected":
        return _block(
            f"PASS rate stuck at {d.get('pass_rate','?')} across "
            f"{d.get('eval_count','10')} consecutive evals.",
            "Stagnant PASS rate signals the reward landscape is flat or the "
            "current phase is exhausted.",
            "In config/phases.yaml, add a new phase with tighter entry_conditions, "
            "or raise SHAPE_ALPHA in core/settings.py, then re-validate.",
            "A fresh gradient should resume PASS-rate improvement within ~50-100 episodes.")
    if event_type == "dd_breach":
        return _block(
            f"Daily DD breached {d.get('count','several')} times "
            f"(limit {d.get('limit','1.0')}%).",
            "Frequent DD breaches mean position sizing or SL is too loose for the "
            "configured risk budget.",
            "Lower max_lot in config/trading_policy.yaml or bias the agent toward "
            "tighter SL buckets (action_space SL_PIPS).",
            "DD breaches should fall below 1 per 10 days after the change.")
    if event_type == "slippage_spike":
        return _block(
            f"Execution cost rose to {d.get('slippage','?')} pips.",
            "Rising slippage erodes edge, usually around news or low-liquidity sessions.",
            "Add a session/spread filter in conditions_engine or raise slippage_pips "
            "in trading_policy.yaml to reflect reality.",
            "Net per-trade cost should return to baseline within a session.")
    if event_type == "pass_rate_drop":
        return _block(
            f"PASS rate dropped from {d.get('prev','?')} to {d.get('now','?')}.",
            "A sudden decline points to reward, data, or hyperparameter drift.",
            "Diff recent metrics CSV rows; if epsilon spiked post-transfer, hold "
            "TRANSFER_EPSILON longer in core/settings.py.",
            "PASS rate should recover toward its prior level within ~100 episodes.")
    if event_type == "import_error":
        return _block(
            f"Import failed: {d.get('error','?')} in {d.get('file','?')}.",
            "Every .py module must import cleanly (inspect_system walks the repo).",
            f"Fix the import at {d.get('file','?')}:{d.get('line','?')} — check the "
            f"module path and that __init__.py exists.",
            "`python -c 'import <module>'` should return zero errors after the fix.")
    if event_type == "test_failure":
        return _block(
            f"Test {d.get('test','?')} failed: {d.get('assertion','?')}.",
            "All tests in tests/ must pass before any advancement (RULE A/B).",
            f"Open {d.get('file','?')} at the failing assertion and correct the "
            f"logic or the expectation, then re-run that test.",
            "`pytest {d.get('file','tests/')}` should report the test passing.")
    return _block(
        d.get("issue", f"Unclassified event: {event_type}"),
        d.get("rule", "Follow the system's hard rules and parity guarantees."),
        d.get("application", "Inspect the referenced component and apply the fix."),
        d.get("conclusion", "Re-run inspect_system.py to confirm green."))
