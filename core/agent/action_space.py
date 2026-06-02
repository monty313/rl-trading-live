"""
core/agent/action_space.py
────────────────────────────────────────────────────────────────────────────
The 756-action discrete action space for the DQN agent.

An action encodes a full trade decision in one integer:

    DIRECTION  : [0=HOLD, 1=BUY, 2=SELL]                       -> 3 choices
    LOT_BUCKET : [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, max_lot] -> 7 choices
    SL_PIPS    : [5, 10, 15, 20, 30, 50]                       -> 6 choices
    TP_PIPS    : [5, 10, 15, 20, 30, 50]                       -> 6 choices

    TOTAL = 3 * 7 * 6 * 6 = 756 actions  (ids 0..755)

WHY ONE INTEGER: the DQN outputs Q-values over a flat discrete space. Packing
(direction, lot, sl, tp) into a single index lets a standard argmax pick a
complete order in one shot, while encode()/decode() keep the mapping reversible
and unit-testable (roundtrip is asserted for all 756 ids).

The mixed-radix layout (direction is the most significant "digit", tp the
least) is:
    action = ((direction * 7 + lot_idx) * 6 + sl_idx) * 6 + tp_idx

Other modules MUST import NUM_ACTIONS from here — never hardcode 756.
"""
from __future__ import annotations

from typing import Tuple

# ── Dimension sizes (mixed-radix digits, most-significant first) ─────────────
N_DIRECTION = 3   # HOLD / BUY / SELL
N_LOT       = 7   # lot buckets
N_SL        = 6   # stop-loss pip buckets
N_TP        = 6   # take-profit pip buckets

NUM_ACTIONS = N_DIRECTION * N_LOT * N_SL * N_TP   # = 756

# ── Direction constants ─────────────────────────────────────────────────────
HOLD = 0
BUY  = 1
SELL = 2
DIRECTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}

# ── Bucket value tables ─────────────────────────────────────────────────────
# Lot buckets: index 6 is special — it resolves to the account's max_lot at
# runtime (see get_lot). The fixed buckets cover common discretionary sizes.
LOT_BUCKETS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, None]  # None -> max_lot
SL_PIPS_TABLE = [5, 10, 15, 20, 30, 50]
TP_PIPS_TABLE = [5, 10, 15, 20, 30, 50]


def encode(direction: int, lot_idx: int, sl_idx: int, tp_idx: int) -> int:
    """
    Pack (direction, lot_idx, sl_idx, tp_idx) into a single action id in 0..755.

    Raises ValueError if any component is out of range, so bad calls fail loudly
    instead of silently wrapping.
    """
    if not (0 <= direction < N_DIRECTION):
        raise ValueError(f"direction {direction} out of range [0,{N_DIRECTION})")
    if not (0 <= lot_idx < N_LOT):
        raise ValueError(f"lot_idx {lot_idx} out of range [0,{N_LOT})")
    if not (0 <= sl_idx < N_SL):
        raise ValueError(f"sl_idx {sl_idx} out of range [0,{N_SL})")
    if not (0 <= tp_idx < N_TP):
        raise ValueError(f"tp_idx {tp_idx} out of range [0,{N_TP})")
    return ((direction * N_LOT + lot_idx) * N_SL + sl_idx) * N_TP + tp_idx


def decode(action_int: int) -> Tuple[int, int, int, int]:
    """
    Inverse of encode(). Returns (direction, lot_idx, sl_idx, tp_idx).

    Raises ValueError if action_int is outside 0..755.
    """
    if not (0 <= action_int < NUM_ACTIONS):
        raise ValueError(f"action_int {action_int} out of range [0,{NUM_ACTIONS})")
    tp_idx = action_int % N_TP
    action_int //= N_TP
    sl_idx = action_int % N_SL
    action_int //= N_SL
    lot_idx = action_int % N_LOT
    action_int //= N_LOT
    direction = action_int  # already < N_DIRECTION by range check above
    return direction, lot_idx, sl_idx, tp_idx


def get_lot(lot_idx: int, max_lot: float) -> float:
    """
    Resolve a lot bucket index to an actual lot size.

    The final bucket (index 6) maps to the account's max_lot. Every result is
    floored at the MT5 minimum (0.01) and capped at max_lot. position_sizer.py
    applies the same clamp — this is the single source of bucket->lot truth.
    """
    raw = LOT_BUCKETS[lot_idx]
    lot = float(max_lot) if raw is None else float(raw)
    lot = max(0.01, min(lot, float(max_lot)))
    return round(lot, 2)


def get_sl_pips(sl_idx: int) -> int:
    """Return stop-loss pip count for a bucket index."""
    return SL_PIPS_TABLE[sl_idx]


def get_tp_pips(tp_idx: int) -> int:
    """Return take-profit pip count for a bucket index."""
    return TP_PIPS_TABLE[tp_idx]


def describe(action_int: int, max_lot: float = 2.0) -> dict:
    """Human-readable expansion of an action id (used by Jordan / dashboard)."""
    direction, lot_idx, sl_idx, tp_idx = decode(action_int)
    return {
        "action_int": action_int,
        "direction": DIRECTION_NAMES[direction],
        "lot": get_lot(lot_idx, max_lot),
        "sl_pips": get_sl_pips(sl_idx),
        "tp_pips": get_tp_pips(tp_idx),
    }
