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

compute_action_mask(phase, rows_by_tf, device, ...) returns
    (dir_mask: (DIRECTION_DIM,) float, must_enter: bool)
where dir_mask is 1.0 allowed / 0.0 masked over {FLAT, BUY, SELL}. The PPO
agent zeroes masked directions before sampling. CRITICAL (DESIGN_DECISIONS #2):
when a strategy gate is active we mask ONLY FLAT so the agent MUST open a trade,
but BUY and SELL both stay available — the code NEVER chooses the direction.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch

from core.agent.action_space import DIRECTION_DIM, FLAT, BUY, SELL
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
    """CCI10 AND CCI30 both > +100 OR both < -100 on both TFs, same direction."""
    def _ex(r):
        c10, c30 = _g(r, "cci10"), _g(r, "cci30")
        if c10 is None or c30 is None:
            return (False, 0)
        if c10 > 100 and c30 > 100:
            return (True, 1)
        if c10 < -100 and c30 < -100:
            return (True, -1)
        return (False, 0)
    a1, d1 = _ex(r1)
    a2, d2 = _ex(r2)
    return bool(a1 and a2 and d1 == d2)


def phase1_cci_align(r1, r2) -> bool:
    """All FOUR of CCI(30) & CCI(100) above their SMA(1,+8) — OR all four below —
    at the SAME bar on BOTH timeframes (1m AND 15m), direction agreeing.

    Per-TF direction: +1 only if BOTH CCI(30)>its SMA(1,+8) AND CCI(100)>its
    SMA(1,+8); -1 only if BOTH are below; 0 otherwise. The gate fires when both
    TFs are non-zero and agree (all bullish, or all bearish). SMA(1,+8) is the
    CCI value forward-shifted 8 bars (period-1 SMA == the value itself)."""
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
    "phase1_cci_align":    (phase1_cci_align,     "force_in_and_gate", [1, 15]),
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
def _dir_mask(allowed: set, device) -> torch.Tensor:
    """(DIRECTION_DIM,) mask: 1.0 for allowed directions in {FLAT,BUY,SELL}."""
    m = torch.zeros(DIRECTION_DIM, dtype=torch.float32, device=device)
    for d in allowed:
        m[d] = 1.0
    return m


def compute_action_mask(phase: dict, rows_by_tf: Dict[int, dict], device: torch.device,
                        is_flat: bool = True) -> Tuple[torch.Tensor, bool]:
    """
    Returns (dir_mask (DIRECTION_DIM,), must_enter bool) for the PPO direction head.

    phase may specify either:
      - a named mask:   {"mask": "phase0_cci_extreme", "mask_type": "...",
                         "gate_timeframes": [1,15]}
      - string conds:   {"entry_conditions": {"buy": "...", "sell": "..."}}
      - free:           mask None / mask_type "free"

    rows_by_tf: {timeframe_minutes: feature_row_dict} for the current bar.
    is_flat: whether the agent holds NO position.

    DESIGN_DECISIONS.md #2: when a gate is ACTIVE we mask ONLY FLAT so the agent
    must open a trade; BUY and SELL stay open — the code never picks the side.
    """
    allow_all = _dir_mask({FLAT, BUY, SELL}, device)
    mask_name = phase.get("mask")
    has_string_conditions = "entry_conditions" in phase
    mask_type = phase.get("mask_type", "free" if not has_string_conditions else None)

    # ── free ──
    if mask_type == "free" and mask_name is None and not has_string_conditions:
        return allow_all, False

    # ── named mask path ──
    if mask_name and mask_name in MASK_REGISTRY:
        fn, mtype, tfs = MASK_REGISTRY[mask_name]
        tfs = phase.get("gate_timeframes", tfs)
        r1 = rows_by_tf.get(tfs[0]) if len(tfs) > 0 else None
        r2 = rows_by_tf.get(tfs[1]) if len(tfs) > 1 else None
        if r1 is None or r2 is None:
            return allow_all, False
        condition = fn(r1, r2)

        if mtype == "force_in_and_gate":
            # ── GATE ON ──────────────────────────────────────────────────────
            # Rule: a trade MUST be active at all times while the gate is ON.
            # This is enforced bar-by-bar — the mask is evaluated every single
            # bar, so the agent can never be flat for even one bar while the
            # gate is firing.
            #
            #   Gate ON + no trades open
            #       → strategy just triggered, agent MUST get in immediately.
            #         BUY or SELL only (agent picks direction + lot size).
            #
            #   Gate ON + already in a trade
            #       → strategy still active, agent has COMPLETE FREEDOM:
            #         add positions, hold, flip direction, partial close,
            #         full close — anything. Lot size and exit timing are
            #         always the agent's decision, never forced.
            #
            #   Gate ON + agent just closed everything (went flat mid-gate)
            #       → must immediately re-enter THIS SAME BAR.
            #         mask returns {BUY, SELL} with must_enter=True so the
            #         agent is forced to open again before the step completes.
            if condition:
                if is_flat:
                    return _dir_mask({BUY, SELL}, device), True
                else:
                    return _dir_mask({FLAT, BUY, SELL}, device), False
            # ── GATE OFF ─────────────────────────────────────────────────────
            #   Gate OFF + no trades open
            #       → no valid setup right now, stay out.
            #         No new entries allowed from flat.
            #
            #   Gate OFF + already in a trade
            #       → strategy ended but position is still open.
            #         HOLD or EXIT only — no flips. Direction head is pinned
            #         to FLAT so neither BUY nor SELL can re-enter; PPO still
            #         decides when to close via the exit head (EXIT_CLOSE)
            #         or to hold until SL/TP fires (EXIT_HOLD).
            #         (User rule: gate-off + in-trade restricts to {HOLD, EXIT}.)
            return _dir_mask({FLAT}, device), False

        if mtype == "open_gate":
            # Same rules as force_in_and_gate (see above).
            if condition:
                if is_flat:
                    return _dir_mask({BUY, SELL}, device), True
                else:
                    return _dir_mask({FLAT, BUY, SELL}, device), False
            # Gate OFF: FLAT only whether or not we hold a position; exit head
            # still controls whether to actually close (user rule, see above).
            return _dir_mask({FLAT}, device), False

        return allow_all, False

    # ── string-condition path ──
    ec = phase.get("entry_conditions", {}) or {}
    buy_c = (ec.get("buy", "any") or "").strip()
    sell_c = (ec.get("sell", "any") or "").strip()
    feats = rows_by_tf.get(1, {}) if rows_by_tf else {}
    buy_true = buy_c != "any" and evaluate(buy_c, feats)
    sell_true = sell_c != "any" and evaluate(sell_c, feats)
    if buy_true and not sell_true:
        return _dir_mask({BUY}, device), False
    if sell_true and not buy_true:
        return _dir_mask({SELL}, device), False
    if buy_true and sell_true:
        return _dir_mask({BUY, SELL}, device), False
    return allow_all, False


# ════════════════════════════════════════════════════════════════════════════
# Vectorized per-episode action mask (BUG-4 FIX)
# ════════════════════════════════════════════════════════════════════════════
def compute_action_mask_batch(phase: dict, rows_by_tf: Dict[int, list], B: int,
                              device: torch.device,
                              is_flat: "torch.Tensor"
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the direction mask + force-entry flag INDEPENDENTLY for each of the B
    episodes. Each episode has its own per-TF feature row (aligned to its own bar)
    and its own flat/in-trade state.

    Args:
      phase       : the active phase dict (same schema as compute_action_mask).
      rows_by_tf  : {tf_minutes: [row_dict per episode]} — each list has length B.
      B           : number of episodes.
      is_flat     : (B,) bool tensor — True where that episode holds NO position.

    Returns:
      dir_mask    : (B, DIRECTION_DIM) float (1.0 allowed / 0.0 masked).
      must_enter  : (B,) bool — True where the gate is ON and the episode is flat.

    The single-episode compute_action_mask() is reused per episode so the gate
    semantics stay in exactly one place; only the per-episode dispatch is new.
    """
    flat_list = is_flat.detach().cpu().tolist()
    masks = torch.empty(B, DIRECTION_DIM, dtype=torch.float32, device=device)
    must = torch.zeros(B, dtype=torch.bool, device=device)
    # transpose {tf: [rows]} -> per-episode {tf: row}
    tfs = list(rows_by_tf.keys())
    for i in range(B):
        rows_i = {tf: rows_by_tf[tf][i] for tf in tfs}
        m, me = compute_action_mask(phase, rows_i, device, is_flat=bool(flat_list[i]))
        masks[i] = m
        must[i] = bool(me)
    return masks, must
