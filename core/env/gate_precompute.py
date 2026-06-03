"""
core/env/gate_precompute.py
────────────────────────────────────────────────────────────────────────────
VECTORIZED phase-gate precomputation — the fix for the 4-hour phase0 stall.

WHY THIS EXISTS (root cause of the stall)
─────────────────────────────────────────
The original hot path recomputed the per-episode action mask by calling
`pandas.DataFrame.to_dict("records")` on each timeframe's full indicator frame
**every single bar**, then looping in Python over all B episodes and building a
fresh dict per episode (`environment._rows_by_tf_batch` +
`conditions_engine.compute_action_mask_batch`). Profiling a 300-step run showed
~90% of wall-clock time inside `to_dict`/`itertuples` (≈105 ms/step at B=4 on
CPU). Over a 43,200-bar episode at B=64 that is multiple hours per episode with
zero printed progress — exactly the reported phase0 stall.

THE INSIGHT
───────────
A named phase mask's decision splits cleanly into two parts:

  1. ``condition`` — a per-bar boolean that depends ONLY on indicator values at
     that bar (and the gate's two timeframes). It does NOT depend on the agent's
     position. This is fully precomputable for the whole series, ONCE, with
     vectorized numpy — no per-bar Python, no per-episode Python.

  2. the mask itself — derived cheaply at run time from ``condition`` and the
     per-episode ``is_flat`` flag using pure tensor ops (see
     ``environment.current_mask_and_force`` / ``_gate_on_batch``).

So we precompute ``gate_on`` as a length-T boolean tensor here (vectorized),
cache it on the env, and the per-bar cost collapses to a tensor gather:
``gate_on[abs_idx]``. No pandas in the hot loop at all.

For the STRING-condition path (custom strategies) we precompute two length-T
boolean tensors ``buy_on`` / ``sell_on`` the same way.

PARITY GUARANTEE
────────────────
The vectorized gate functions below reproduce EXACTLY the same boolean logic as
the scalar functions in ``conditions_engine`` (phase0..phase6). The unit test
``tests/unit/test_gate_precompute.py`` asserts bar-for-bar equality between the
vectorized result and the original scalar ``compute_action_mask`` path, so the
optimization can never silently change trading behavior.

ALIGNMENT
─────────
Higher-TF bars align to 1m bar ``i`` by integer position ``i // tf`` (the most
recent completed higher-TF bar at/under bar ``i``) — identical to
``environment._tf_pos``. We build, for each gate timeframe, a length-T array
that maps every 1m bar to its aligned higher-TF row's indicator columns, then
evaluate the gate condition across all T bars at once.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core.env import conditions_engine as CE


# ════════════════════════════════════════════════════════════════════════════
# Helpers: pull an aligned (T,) numpy column from a per-TF indicator frame
# ════════════════════════════════════════════════════════════════════════════
def _aligned_col(df_tf: pd.DataFrame, col: str, tf: int, T: int) -> np.ndarray:
    """
    Return a length-T float array where entry i = df_tf[col] at higher-TF row
    ``min(i // tf, len(df_tf)-1)``. Missing columns yield all-NaN (the scalar
    gate treats missing/NaN as "no signal", so the vectorized path matches).
    """
    if col not in df_tf.columns or len(df_tf) == 0:
        return np.full(T, np.nan, dtype=np.float64)
    src = df_tf[col].to_numpy(np.float64)
    tf_len = len(src)
    # integer-position alignment, identical to environment._tf_pos
    pos = np.minimum(np.arange(T) // max(tf, 1), tf_len - 1)
    return src[pos]


def _cols_for(df_tf: pd.DataFrame, names: List[str], tf: int, T: int
              ) -> Dict[str, np.ndarray]:
    """Build {name: aligned (T,) array} for the indicator columns a gate needs."""
    return {n: _aligned_col(df_tf, n, tf, T) for n in names}


# ════════════════════════════════════════════════════════════════════════════
# Vectorized equivalents of the phase0..phase6 scalar gate functions.
# Each takes the two aligned column dicts (TF a, TF b) and returns a (T,) bool.
# The logic mirrors conditions_engine.phaseN_* EXACTLY (NaN -> no signal).
# ════════════════════════════════════════════════════════════════════════════
def _finite(*arrs: np.ndarray) -> np.ndarray:
    """Elementwise AND of np.isfinite over all inputs (NaN-guard, like _g())."""
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _v_phase0(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> np.ndarray:
    """CCI10 & CCI30 both >+100 (dir +1) OR both <-100 (dir -1) on BOTH TFs,
    same direction. Mirrors phase0_cci_extreme."""
    def _ex(d):
        c10, c30 = d["cci10"], d["cci30"]
        ok = _finite(c10, c30)
        up = ok & (c10 > 100) & (c30 > 100)
        dn = ok & (c10 < -100) & (c30 < -100)
        direction = np.where(up, 1, np.where(dn, -1, 0))
        active = up | dn
        return active, direction
    a1, d1 = _ex(a)
    a2, d2 = _ex(b)
    return a1 & a2 & (d1 == d2)


def _aligned_sign(val: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Vectorized _aligned(): +1 if val>ref, -1 if val<ref, 0 otherwise/NaN."""
    ok = _finite(val, ref)
    s = np.where(val > ref, 1, np.where(val < ref, -1, 0))
    return np.where(ok, s, 0)


def _v_phase1(a, b) -> np.ndarray:
    """CCI(30) & CCI(100) each vs their SMA(1,+8) on both TFs, all four agree
    (all above, or all below). Mirrors phase1_cci_align EXACTLY."""
    def _dir(d):
        d30 = _aligned_sign(d["cci30"], d["cci30_sma1_sh8"])
        d100 = _aligned_sign(d["cci100"], d["cci100_sma1_sh8"])
        return np.where((d30 != 0) & (d30 == d100), d30, 0)
    d1, d2 = _dir(a), _dir(b)
    return (d1 != 0) & (d1 == d2)


def _v_hilo_dir(d) -> np.ndarray:
    """close above BOTH high/low sma4_sh8 (+1) or below BOTH (-1)."""
    p, hi, lo = d["close"], d["high_sma4_sh8"], d["low_sma4_sh8"]
    ok = _finite(p, hi, lo)
    up = ok & (p > hi) & (p > lo)
    dn = ok & (p < hi) & (p < lo)
    return np.where(up, 1, np.where(dn, -1, 0))


def _v_phase2(a, b) -> np.ndarray:
    """Same dir on both TFs. Mirrors phase2_hilo_trend."""
    d1, d2 = _v_hilo_dir(a), _v_hilo_dir(b)
    return (d1 != 0) & (d1 == d2)


def _v_phase3(a, b) -> np.ndarray:
    """OPPOSITE sides on the two TFs. Mirrors phase3_hilo_counter."""
    d1, d2 = _v_hilo_dir(a), _v_hilo_dir(b)
    return (d1 != 0) & (d2 != 0) & (d1 != d2)


def _v_phase4(a, b) -> np.ndarray:
    """1m bb position vs 15m bb position, both agree. Mirrors phase4_bb_position."""
    def _d1(d):
        p, m200, u20, l20 = d["close"], d["bb200_mid"], d["bb20_upper"], d["bb20_lower"]
        ok = _finite(p, m200, u20, l20)
        up = ok & (p > m200) & (p > u20)
        dn = ok & (p < m200) & (p < l20)
        return np.where(up, 1, np.where(dn, -1, 0))
    def _d2(d):
        p, m200, m20 = d["close"], d["bb200_mid"], d["bb20_mid"]
        ok = _finite(p, m200, m20)
        up = ok & (p > m200) & (p > m20)
        dn = ok & (p < m200) & (p < m20)
        return np.where(up, 1, np.where(dn, -1, 0))
    d1, d2 = _d1(a), _d2(b)
    return (d1 != 0) & (d1 == d2)


def _v_phase5(a, b) -> np.ndarray:
    """sma2_sh0>sh1>...>sh4 (bull) or strictly reversed (bear) on both TFs.
    Mirrors phase5_sma_stack."""
    def _dir(d):
        vals = [d[f"sma2_sh{i}"] for i in range(5)]
        ok = _finite(*vals)
        bull = ok.copy()
        bear = ok.copy()
        for i in range(4):
            bull &= vals[i] > vals[i + 1]
            bear &= vals[i] < vals[i + 1]
        return np.where(bull, 1, np.where(bear, -1, 0))
    d1, d2 = _dir(a), _dir(b)
    return (d1 != 0) & (d1 == d2)


def _v_phase6(a, b) -> np.ndarray:
    """ATR14>ref AND ATR45>ref on BOTH TFs. Mirrors phase6_atr_expansion."""
    def _exp(d):
        a14, a14r = d["atr14"], d["atr14_sma1_sh8"]
        a45, a45r = d["atr45"], d["atr45_sma1_sh8"]
        ok = _finite(a14, a14r, a45, a45r)
        return ok & (a14 > a14r) & (a45 > a45r)
    return _exp(a) & _exp(b)


# name -> (vectorized_fn, [columns TF-a needs], [columns TF-b needs])
# Columns are the same set both TFs read (the scalar fns read the same keys from
# each row), so we list them once and reuse for both timeframes.
_VEC_REGISTRY = {
    "phase0_cci_extreme":   (_v_phase0, ["cci10", "cci30"]),
    "phase1_cci_align":     (_v_phase1, ["cci30", "cci100", "cci30_sma1_sh8",
                                         "cci100_sma1_sh8"]),
    "phase2_hilo_trend":    (_v_phase2, ["close", "high_sma4_sh8", "low_sma4_sh8"]),
    "phase3_hilo_counter":  (_v_phase3, ["close", "high_sma4_sh8", "low_sma4_sh8"]),
    "phase4_bb_position":   (_v_phase4, ["close", "bb200_mid", "bb20_upper",
                                         "bb20_lower", "bb20_mid"]),
    "phase5_sma_stack":     (_v_phase5, [f"sma2_sh{i}" for i in range(5)]),
    "phase6_atr_expansion": (_v_phase6, ["atr14", "atr14_sma1_sh8",
                                         "atr45", "atr45_sma1_sh8"]),
}


# ════════════════════════════════════════════════════════════════════════════
# Public: precompute the per-bar gate signal for the active phase
# ════════════════════════════════════════════════════════════════════════════
def precompute_gate_signal(
    phase: dict,
    tf_indicators: Dict[int, pd.DataFrame],
    feature_matrix: torch.Tensor,
    T: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Build the per-bar gate signal for ``phase`` over the whole length-T series,
    ONCE, fully vectorized. Returns a dict of length-T tensors, or None when the
    phase needs no gating (free phase) so the env can skip masking entirely.

    Return schema (all tensors are length T on ``device``):
      For a NAMED mask phase:
        {"kind": "named", "gate_on": (T,) bool}
          gate_on[i] == True  <=>  the named gate condition fires at bar i.
      For a STRING-condition phase:
        {"kind": "string", "buy_on": (T,) bool, "sell_on": (T,) bool}
      None: free phase / unknown mask -> no gating (allow-all every bar).

    The env turns these per-bar booleans into the (B, DIRECTION_DIM) mask with a
    couple of tensor ops (see environment), so the per-step cost is O(B) tensor
    work instead of O(B × TF_len) pandas work.
    """
    mask_name = phase.get("mask")
    has_string = "entry_conditions" in phase
    mask_type = phase.get("mask_type", "free" if not has_string else None)

    # ── free phase: no gating, mask is allow-all every bar ──
    if mask_type == "free" and mask_name is None and not has_string:
        return None

    # ── named mask path (the curriculum phases) ──
    if mask_name and mask_name in _VEC_REGISTRY:
        vec_fn, cols = _VEC_REGISTRY[mask_name]
        # The gate's two timeframes (phase override or the registry default).
        _fn, _mtype, default_tfs = CE.MASK_REGISTRY[mask_name]
        tfs = phase.get("gate_timeframes", default_tfs)
        tf_a = tfs[0] if len(tfs) > 0 else 1
        tf_b = tfs[1] if len(tfs) > 1 else tf_a

        df_a = tf_indicators.get(tf_a)
        df_b = tf_indicators.get(tf_b)
        if df_a is None or df_b is None:
            # No TF frames (e.g. prebuilt feature matrix, no raw OHLCV): the
            # scalar path returns allow-all in that case, so we do too.
            return None

        a_cols = _cols_for(df_a, cols, tf_a, T)
        b_cols = _cols_for(df_b, cols, tf_b, T)
        gate_on_np = vec_fn(a_cols, b_cols).astype(bool)
        gate_on = torch.as_tensor(gate_on_np, dtype=torch.bool, device=device)
        return {"kind": "named", "gate_on": gate_on}

    # ── string-condition path (custom strategies, evaluated on 1m rows) ──
    if has_string:
        ec = phase.get("entry_conditions", {}) or {}
        buy_c = (ec.get("buy", "any") or "").strip()
        sell_c = (ec.get("sell", "any") or "").strip()
        # Evaluate each boolean expression across ALL bars at once using the
        # feature matrix columns (the same VARIABLE_REGISTRY names the scalar
        # evaluate() uses). "any" => never gates that side (matches evaluate()).
        from core.env.indicators import COL
        feat_np = feature_matrix.detach().cpu().numpy()

        def _eval_series(cond: str) -> np.ndarray:
            if cond in ("", "any"):
                # "any" returns True in evaluate(); but in compute_action_mask the
                # "any" side is treated as NOT a trigger (buy_true requires
                # buy_c != "any"). So a per-bar trigger for "any" is always False.
                return np.zeros(T, dtype=bool)
            CE.validate_condition(cond)
            # Build a namespace of (name -> (T,) array) for every registry var.
            ns = {name: (feat_np[:, idx] if idx < feat_np.shape[1]
                         else np.zeros(T)) for name, idx in COL.items()}
            # numpy evaluates the boolean expression elementwise across the series.
            res = eval(cond, {"__builtins__": {}}, ns)  # noqa: S307 (sandboxed)
            return np.asarray(res, dtype=bool).reshape(-1)[:T]

        buy_on = torch.as_tensor(_eval_series(buy_c), dtype=torch.bool, device=device)
        sell_on = torch.as_tensor(_eval_series(sell_c), dtype=torch.bool, device=device)
        return {"kind": "string", "buy_on": buy_on, "sell_on": sell_on}

    # Unknown mask name -> no gating (scalar path returns allow-all).
    return None
