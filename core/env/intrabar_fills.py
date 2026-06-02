"""
core/env/intrabar_fills.py
────────────────────────────────────────────────────────────────────────────
M1 intrabar fill approximation. No tick data exists, so fills are derived from
the M1 OHLC bar plus per-instrument spread/slippage from trading_policy.yaml.

PARITY (HARD RULE 10): this file's md5 is recorded in the manifest. Training,
backtest, and live_runner all import compute_fill() — fills are identical across
all three. Do not duplicate this math elsewhere.

FILL RULES (STEP 4.2, implemented exactly):
  BUY fill : open + (spread_pips * pip_value * 0.5) + (slippage_pips * pip_value)
  SELL fill: open - (spread_pips * pip_value * 0.5) - (slippage_pips * pip_value)
  SL for BUY : bar_low  - (sl_buffer_pips * pip_value)   unless model SL is tighter
  TP for BUY : bar_high + (tp_buffer_pips * pip_value)    unless model TP is farther
  SL for SELL: bar_high + (sl_buffer_pips * pip_value)
  TP for SELL: bar_low  - (tp_buffer_pips * pip_value)

The model's chosen SL/TP (in pips, from the action) overrides the OHLC-derived
buffer when it is TIGHTER (SL) or FARTHER (TP). Never allow SL wider than
atr_14 * 3.
"""
from __future__ import annotations

from typing import Dict, Optional

from core.agent.action_space import BUY, SELL, HOLD

# Defaults if an instrument is missing from policy (EURUSD-like FX).
_DEFAULT_INSTR = {"pip_value": 0.0001, "spread_pips": 1.0,
                  "slippage_pips": 0.5, "sl_buffer_pips": 2.0}


def _instr_params(instrument: str, policy: dict) -> dict:
    settings = (policy or {}).get("instrument_settings", {}) or {}
    return {**_DEFAULT_INSTR, **settings.get(instrument, {})}


def compute_fill(bar: Dict[str, float], direction: int, sl_pips: int, tp_pips: int,
                 instrument: str = "EURUSD", policy: Optional[dict] = None,
                 atr_14: Optional[float] = None) -> dict:
    """
    Compute the approximate fill for a trade on one M1 bar.

    Args:
        bar       : dict with keys open, high, low, close
        direction : action_space.HOLD / BUY / SELL
        sl_pips   : model-chosen stop-loss distance (pips)
        tp_pips   : model-chosen take-profit distance (pips)
        instrument: symbol key into policy["instrument_settings"]
        policy    : parsed trading_policy.yaml dict
        atr_14    : current ATR(14) in price units (caps SL width at 3*ATR)

    Returns dict: {direction, entry, sl, tp, spread_cost, lots_ok}
    For HOLD: returns entry=open, sl=tp=None, spread_cost=0.
    """
    p = _instr_params(instrument, policy or {})
    pip = float(p["pip_value"])
    spread_pips = float(p["spread_pips"])
    slippage_pips = float(p["slippage_pips"])
    sl_buffer_pips = float(p["sl_buffer_pips"])

    o = float(bar["open"])
    h = float(bar["high"])
    l = float(bar["low"])
    spread_cost = spread_pips * pip

    if direction == HOLD:
        return {"direction": HOLD, "entry": o, "sl": None, "tp": None,
                "spread_cost": 0.0}

    if direction == BUY:
        entry = o + (spread_pips * pip * 0.5) + (slippage_pips * pip)
        # OHLC-derived protective levels
        sl_ohlc = l - (sl_buffer_pips * pip)
        tp_ohlc = h + (tp_pips * pip)        # tp buffer comes from model tp_pips
        # model-chosen levels (pips from entry)
        sl_model = entry - (sl_pips * pip)
        tp_model = entry + (tp_pips * pip)
        # SL: take the TIGHTER (closer to entry) of the two
        sl = max(sl_ohlc, sl_model)
        # TP: take the FARTHER (further from entry) of the two
        tp = max(tp_ohlc, tp_model)
        # Cap SL width at 3*ATR (never let SL be wider than this)
        if atr_14 is not None and atr_14 > 0:
            min_sl = entry - 3.0 * atr_14
            sl = max(sl, min_sl)
        return {"direction": BUY, "entry": entry, "sl": sl, "tp": tp,
                "spread_cost": spread_cost}

    # SELL
    entry = o - (spread_pips * pip * 0.5) - (slippage_pips * pip)
    sl_ohlc = h + (sl_buffer_pips * pip)
    tp_ohlc = l - (tp_pips * pip)
    sl_model = entry + (sl_pips * pip)
    tp_model = entry - (tp_pips * pip)
    # SL: tighter = closer to entry = the SMALLER value (above entry)
    sl = min(sl_ohlc, sl_model)
    # TP: farther = further below entry = the SMALLER value
    tp = min(tp_ohlc, tp_model)
    if atr_14 is not None and atr_14 > 0:
        max_sl = entry + 3.0 * atr_14
        sl = min(sl, max_sl)
    return {"direction": SELL, "entry": entry, "sl": sl, "tp": tp,
            "spread_cost": spread_cost}
