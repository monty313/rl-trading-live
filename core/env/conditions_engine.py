"""
core/env/conditions_engine.py
────────────────────────────────────────────────────────────────────────────
Phase gating + action masking. Two mechanisms, both supported:

  1. NAMED phase masks (authoritative curriculum, ported from REPO1):
     7 mask functions (phase0..phase6) registered in MASK_REGISTRY with a
     mask_type and the timeframes the gate evaluates on. Each function takes
     two per-TF feature-row dicts and returns a bool "condition met".
     mask_type semantics (REPO1 _eval_mask):
        force_in_and_gate : condition True  -> agent MUST be in a trade
                            (force entry if flat); False -> block new entries.
        open_gate         : condition gates OPENING only; never forces. Existing
                            positions may stay open (agent learns when to exit).
        free              : no masking.

  2. STRING conditions (custom strategies, no code): entry_conditions {buy,sell}
     evaluated against the VARIABLE_REGISTRY (RULE 12 boolean masking).

compute_action_mask(phase, rows_by_tf|features, device, ...) returns
    (mask: (NUM_ACTIONS,) float, must_enter: bool)
where mask is 1.0 allowed / 0.0 masked. The env adds -1e9 to masked Q-values
and, if must_enter and the agent is flat, forces a non-HOLD action.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch

from core.agent.action_space import NUM_ACTIONS, HOLD, BUY, SELL, decode
from core.env.indicators import FEATURE_COLUMNS

VARIABLE_REGISTRY = set(FEATURE_COLUMNS)
_ALLOWED_TOKENS = {"and", "or", "not", "True", "False"}
_NEG_INF = -1e9


class ConfigError(Exception):
    """Raised when a phase condition references an unregistered variable."""


# ════════════════════════════════════════════════════════════════════════════
# Named phase-mask functions (ported verbatim from REPO1 env/indicators.py)
# Each takes two dicts (row for TF a, row for TF b) and returns bool.
# ════════════════════════════════════════════════════════════════════════════
def _g(row, key):
    v = row.get(key)
    return None if v is None or (isinstance(v, float) and v != v) else v  # NaN check


def _aligned(val, ref) -> int:
    if val is None or ref is None:
        return 0
    return 1 if val > ref else (-1 if val < ref else 0)


def phase0_cci_extreme(r1, r2) -> bool:
    """CCI30 AND CCI100 both > +100 OR both < -100 on both TFs, same direction."""
    def _ex(r):
        c30, c100 = _g(r, "cci30"), _g(r, "cci100")
        if c30 is None or c100 is None:
            return (False, 0)
        if c30 > 100 and c100 > 100:
            return (True, 1)
        if c30 < -100 and c100 < -100:
            return (True, -1)
        return (False, 0)
    a1, d1 = _ex(r1)
    a2, d2 = _ex(r2)
    return bool(a1 and a2 and d1 == d2)


def phase1_cci_align(r1, r2) -> bool:
    """CCI30 & CCI100 each above/below their SMA(1,+8) on both TFs, all agree."""
    def _dir(r):
        d30 = _aligned(_g(r, "cci30"), _g(r, "cci30_sma1_sh8"))
        d100 = _aligned(_g(r, "cci100"), _g(r, "cci100_sma1_sh8"))
        return d30 if (d30 != 0 and d30 == d100) else 0
    d1, d2 = _dir(r1), _dir(r2)
    return d1 != 0 and d1 == d2


def phase2_hilo_trend(r1, r2) -> bool:
    """close above BOTH high/low sma4_sh8 OR below BOTH, same dir on both TFs."""
    def _dir(r):
        p, hi, lo = _g(r, "close"), _g(r, "high_sma4_sh8"), _g(r, "low_sma4_sh8")
        if None in (p, hi, lo):
            return 0
        if p > hi and p > lo:
            return 1
        if p < hi and p < lo:
            return -1
        return 0
    d1, d2 = _dir(r1), _dir(r2)
    return d1 != 0 and d1 == d2


def phase3_hilo_counter(r1, r2) -> bool:
    """1m and 15m on OPPOSITE sides of the high/low sma4_sh8 band."""
    def _dir(r):
        p, hi, lo = _g(r, "close"), _g(r, "high_sma4_sh8"), _g(r, "low_sma4_sh8")
        if None in (p, hi, lo):
            return 0
        if p > hi and p > lo:
            return 1
        if p < hi and p < lo:
            return -1
        return 0
    d1, d2 = _dir(r1), _dir(r2)
    return d1 != 0 and d2 != 0 and d1 != d2


def phase4_bb_position(r1, r2) -> bool:
    """1m: close>bb200_mid & >bb20_upper (bull) or <bb200_mid & <bb20_lower (bear).
    15m: close vs bb200_mid & bb20_mid. Both TFs agree direction."""
    def _d1(r):
        p, m200, u20, l20 = (_g(r, "close"), _g(r, "bb200_mid"),
                             _g(r, "bb20_upper"), _g(r, "bb20_lower"))
        if None in (p, m200, u20, l20):
            return 0
        if p > m200 and p > u20:
            return 1
        if p < m200 and p < l20:
            return -1
        return 0
    def _d2(r):
        p, m200, m20 = _g(r, "close"), _g(r, "bb200_mid"), _g(r, "bb20_mid")
        if None in (p, m200, m20):
            return 0
        if p > m200 and p > m20:
            return 1
        if p < m200 and p < m20:
            return -1
        return 0
    d1, d2 = _d1(r1), _d2(r2)
    return d1 != 0 and d1 == d2


def phase5_sma_stack(r1, r2) -> bool:
    """sma2_sh0>sh1>...>sh4 (bull) or strictly reversed (bear) on both TFs."""
    def _dir(r):
        vals = [_g(r, f"sma2_sh{i}") for i in range(5)]
        if any(v is None for v in vals):
            return 0
        if all(vals[i] > vals[i + 1] for i in range(4)):
            return 1
        if all(vals[i] < vals[i + 1] for i in range(4)):
            return -1
        return 0
    d1, d2 = _dir(r1), _dir(r2)
    return d1 != 0 and d1 == d2


def phase6_atr_expansion(r1, r2) -> bool:
    """ATR14>atr14_sma1_sh8 AND ATR45>atr45_sma1_sh8 on BOTH TFs."""
    def _exp(r):
        a14, a14r = _g(r, "atr14"), _g(r, "atr14_sma1_sh8")
        a45, a45r = _g(r, "atr45"), _g(r, "atr45_sma1_sh8")
        if None in (a14, a14r, a45, a45r):
            return False
        return a14 > a14r and a45 > a45r
    return _exp(r1) and _exp(r2)


# name -> (function, mask_type, [tf_a, tf_b])
MASK_REGISTRY: Dict[str, Tuple[Callable, str, list]] = {
    "phase0_cci_extreme":  (phase0_cci_extreme,  "force_in_and_gate", [1, 15]),
    "phase1_cci_align":    (phase1_cci_align,     "open_gate",         [1, 15]),
    "phase2_hilo_trend":   (phase2_hilo_trend,    "force_in_and_gate", [1, 30]),
    "phase3_hilo_counter": (phase3_hilo_counter,  "force_in_and_gate", [1, 15]),
    "phase4_bb_position":  (phase4_bb_position,   "force_in_and_gate", [1, 15]),
    "phase5_sma_stack":    (phase5_sma_stack,      "force_in_and_gate", [1, 60]),
    "phase6_atr_expansion":(phase6_atr_expansion,  "force_in_and_gate", [1, 60]),
}


# ════════════════════════════════════════════════════════════════════════════
# String-condition path (custom strategies)
# ════════════════════════════════════════════════════════════════════════════
def _identifiers(expr: str):
    tok = ""
    for ch in expr:
        if ch.isalnum() or ch == "_":
            tok += ch
        else:
            if tok:
                yield tok
                tok = ""
    if tok:
        yield tok


def validate_condition(condition_str: str, phase_name: str = "?") -> None:
    cond = (condition_str or "").strip()
    if cond in ("", "any"):
        return
    for tok in _identifiers(cond):
        if tok in _ALLOWED_TOKENS:
            continue
        try:
            float(tok)
            continue
        except ValueError:
            pass
        if tok not in VARIABLE_REGISTRY:
            raise ConfigError(
                f"**ISSUE**: Variable '{tok}' in phase '{phase_name}' not in "
                f"VARIABLE_REGISTRY.\n**RULE**: condition variables must be in "
                f"indicators.FEATURE_COLUMNS.\n**APPLICATION**: add '{tok}' to "
                f"FEATURE_COLUMNS (and compute it in build_feature_matrix).\n"
                f"**CONCLUSION**: re-run validate_phases_yaml.py — should PASS.")


def evaluate(condition_str: str, bar_features: Dict[str, float]) -> bool:
    cond = (condition_str or "").strip()
    if cond == "any":
        return True
    if cond == "":
        return False
    validate_condition(cond)
    safe = {k: bar_features.get(k, 0.0) for k in VARIABLE_REGISTRY}
    return bool(eval(cond, {"__builtins__": {}}, safe))  # noqa: S307 (sandboxed)


# ════════════════════════════════════════════════════════════════════════════
# Action mask construction
# ════════════════════════════════════════════════════════════════════════════
def _dir_of(a: int) -> int:
    return decode(a)[0]


def _mask_for_allowed_dirs(allowed: set, device) -> torch.Tensor:
    m = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=device)
    for a in range(NUM_ACTIONS):
        if _dir_of(a) in allowed:
            m[a] = 1.0
    return m


def compute_action_mask(phase: dict, rows_by_tf: Dict[int, dict], device: torch.device,
                        num_actions: int = NUM_ACTIONS, is_flat: bool = True
                        ) -> Tuple[torch.Tensor, bool]:
    """
    Returns (mask (num_actions,), must_enter bool).

    phase may specify either:
      - a named mask:   {"mask": "phase0_cci_extreme", "mask_type": "...",
                         "gate_timeframes": [1,15]}
      - string conds:   {"entry_conditions": {"buy": "...", "sell": "..."}}
      - free:           mask None / mask_type "free"

    rows_by_tf: {timeframe_minutes: feature_row_dict} for the current bar.
    is_flat: whether the agent currently holds NO position (affects must_enter
             and whether HOLD/exit stays allowed under open_gate / force modes).
    """
    mask_name = phase.get("mask")
    has_string_conditions = "entry_conditions" in phase
    # Default mask_type is 'free' ONLY when there is no named mask and no string
    # conditions; otherwise honor the declared mask_type / fall through to strings.
    mask_type = phase.get("mask_type", "free" if not has_string_conditions else None)

    # ── free ──
    if mask_type == "free" and mask_name is None and not has_string_conditions:
        return torch.ones(num_actions, dtype=torch.float32, device=device), False

    # ── named mask path ──
    if mask_name and mask_name in MASK_REGISTRY:
        fn, mtype, tfs = MASK_REGISTRY[mask_name]
        tfs = phase.get("gate_timeframes", tfs)
        r1 = rows_by_tf.get(tfs[0]) if len(tfs) > 0 else None
        r2 = rows_by_tf.get(tfs[1]) if len(tfs) > 1 else None
        if r1 is None or r2 is None:
            return torch.ones(num_actions, dtype=torch.float32, device=device), False
        condition = fn(r1, r2)

        if mtype == "force_in_and_gate":
            if condition:
                # must be in a trade: allow BUY/SELL; mask HOLD (force entry if flat)
                mask = _mask_for_allowed_dirs({BUY, SELL}, device)
                return mask, bool(is_flat)
            # condition false: block opening new trades. Allow HOLD (and exits) so
            # existing positions can be managed; if flat, only HOLD remains.
            return _mask_for_allowed_dirs({HOLD}, device), False

        if mtype == "open_gate":
            if condition:
                return torch.ones(num_actions, dtype=torch.float32, device=device), False
            # gate closed: no NEW entries; HOLD/exits allowed (learn when to close)
            return _mask_for_allowed_dirs({HOLD}, device), False

        return torch.ones(num_actions, dtype=torch.float32, device=device), False

    # ── string-condition path ──
    ec = phase.get("entry_conditions", {}) or {}
    buy_c = (ec.get("buy", "any") or "").strip()
    sell_c = (ec.get("sell", "any") or "").strip()
    feats = rows_by_tf.get(1, {}) if rows_by_tf else {}
    buy_true = buy_c != "any" and evaluate(buy_c, feats)
    sell_true = sell_c != "any" and evaluate(sell_c, feats)
    if buy_true and not sell_true:
        return _mask_for_allowed_dirs({BUY}, device), False
    if sell_true and not buy_true:
        return _mask_for_allowed_dirs({SELL}, device), False
    if buy_true and sell_true:
        return _mask_for_allowed_dirs({BUY, SELL}, device), False
    return torch.ones(num_actions, dtype=torch.float32, device=device), False
