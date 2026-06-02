"""
core/env/conditions_engine.py
────────────────────────────────────────────────────────────────────────────
Parses the `entry_conditions` strings from config/phases.yaml and produces the
per-step action mask (HARD RULE 12).

VARIABLE_REGISTRY is the set of names a condition string may reference. It must
stay in sync with core/env/indicators.py FEATURE_COLUMNS. If a phase references
a name that is not registered, evaluate()/validate raise ConfigError with EXACT
IRAC remediation instructions — so adding a new indicator is a guided process.

ACTION MASK RULE (RULE 12), shape (NUM_ACTIONS,), 1.0=allowed / 0.0=masked:
  - buy condition True  -> allow only BUY actions   (mask all HOLD + SELL)
  - sell condition True -> allow only SELL actions  (mask all HOLD + BUY)
  - both False          -> allow everything (model chooses freely)
  - "any"               -> allow everything
The environment converts 0.0 -> additive -1e9 on the Q-values before argmax.
"""
from __future__ import annotations

from typing import Dict

import torch

from core.agent.action_space import (
    NUM_ACTIONS, N_LOT, N_SL, N_TP, HOLD, BUY, SELL, decode,
)
from core.env.indicators import FEATURE_COLUMNS

# Names a condition string may use. Kept in sync with indicators.FEATURE_COLUMNS.
VARIABLE_REGISTRY = set(FEATURE_COLUMNS)

# Operators/keywords permitted inside a condition expression (safe eval allowlist).
_ALLOWED_TOKENS = {"and", "or", "not", "True", "False"}


class ConfigError(Exception):
    """Raised when a phase condition references an unregistered variable."""


def _irac_unknown_var(var: str, phase_name: str = "?") -> str:
    return (
        f"**ISSUE**: Variable '{var}' used in phase '{phase_name}' is not in "
        f"VARIABLE_REGISTRY.\n"
        f"**RULE**: All condition variables must be computed in "
        f"core/env/indicators.py (build_feature_matrix) and registered in "
        f"core/env/conditions_engine.py VARIABLE_REGISTRY.\n"
        f"**APPLICATION**: Add '{var}' as a column in indicators.FEATURE_COLUMNS "
        f"(and compute it in build_feature_matrix); it is auto-added to "
        f"VARIABLE_REGISTRY since the registry derives from FEATURE_COLUMNS.\n"
        f"**CONCLUSION**: Re-run `python scripts/validate_phases_yaml.py` — "
        f"the phase should show PASS."
    )


def _extract_identifiers(expr: str):
    """Yield identifier-like tokens from an expression (for validation)."""
    token = ""
    for ch in expr:
        if ch.isalnum() or ch == "_":
            token += ch
        else:
            if token:
                yield token
                token = ""
    if token:
        yield token


def validate_condition(condition_str: str, phase_name: str = "?") -> None:
    """
    Raise ConfigError (with IRAC) if the condition references an unknown variable.
    Numbers and allowed keywords are fine. "any" is always valid.
    """
    cond = (condition_str or "").strip()
    if cond == "any" or cond == "":
        return
    for tok in _extract_identifiers(cond):
        if tok in _ALLOWED_TOKENS:
            continue
        if tok.replace(".", "").replace("e", "").replace("E", "").isdigit():
            continue  # numeric literal
        # pure-number check (handles 1.5, 100, -3 handled by sign outside token)
        try:
            float(tok)
            continue
        except ValueError:
            pass
        if tok not in VARIABLE_REGISTRY:
            raise ConfigError(_irac_unknown_var(tok, phase_name))


def evaluate(condition_str: str, bar_features_dict: Dict[str, float]) -> bool:
    """
    Evaluate a condition string against a bar's feature dict.

    "any" -> True. Empty -> False. Uses a restricted eval: no builtins, only the
    registered feature variables in scope, so arbitrary code cannot run.
    """
    cond = (condition_str or "").strip()
    if cond == "any":
        return True
    if cond == "":
        return False
    validate_condition(cond)
    safe_globals = {"__builtins__": {}}
    safe_locals = {k: bar_features_dict.get(k, 0.0) for k in VARIABLE_REGISTRY}
    try:
        return bool(eval(cond, safe_globals, safe_locals))  # noqa: S307 (sandboxed)
    except Exception as exc:                                  # pragma: no cover
        raise ConfigError(f"Failed to evaluate condition '{cond}': {exc}") from exc


def _direction_of(action_int: int) -> int:
    """Return the direction component (HOLD/BUY/SELL) of an action id."""
    direction, _lot, _sl, _tp = decode(action_int)
    return direction


def compute_action_mask(phase: dict, bar_features_dict: Dict[str, float],
                        device: torch.device, num_actions: int = NUM_ACTIONS
                        ) -> torch.Tensor:
    """
    Build a (num_actions,) float32 mask for one bar given a phase's conditions.

    phase: a dict with entry_conditions {buy, sell} (as in phases.yaml).
    Returns 1.0 for allowed actions, 0.0 for masked (RULE 12).
    """
    ec = (phase or {}).get("entry_conditions", {}) or {}
    buy_cond = (ec.get("buy", "any") or "").strip()
    sell_cond = (ec.get("sell", "any") or "").strip()

    mask = torch.ones(num_actions, dtype=torch.float32, device=device)

    # "any" means "no gating on this side" — it is NOT an active directional
    # signal. A side only gates (masks the other directions) when it is a real
    # condition that currently evaluates True. So if both sides are "any"
    # (the free LIVE_IMPROVE phase), everything is allowed, HOLD included.
    buy_true = (buy_cond != "any") and evaluate(buy_cond, bar_features_dict)
    sell_true = (sell_cond != "any") and evaluate(sell_cond, bar_features_dict)

    # If both conditions are "any" (or both False), nothing is masked.
    if buy_true and not sell_true:
        # allow only BUY: mask HOLD + SELL
        for a in range(num_actions):
            if _direction_of(a) != BUY:
                mask[a] = 0.0
    elif sell_true and not buy_true:
        # allow only SELL: mask HOLD + BUY
        for a in range(num_actions):
            if _direction_of(a) != SELL:
                mask[a] = 0.0
    elif buy_true and sell_true:
        # both signals fire: allow BUY and SELL, mask only HOLD
        for a in range(num_actions):
            if _direction_of(a) == HOLD:
                mask[a] = 0.0
    # else: both False -> all allowed (mask stays all ones)
    return mask
