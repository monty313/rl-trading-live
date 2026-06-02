"""
core/agent/action_space.py
────────────────────────────────────────────────────────────────────────────
PPO action space (single source of truth). See DESIGN_DECISIONS.md #1.

The PPO agent emits a STRUCTURED action with three components:

  DIRECTION (categorical, 3) : FLAT=0, BUY=1, SELL=2
  LOT       (continuous, 1)  : raw scalar in [0,1] -> mapped to [min_lot, max_lot]
  EXIT      (categorical, 3) : HOLD=0, REDUCE=1, CLOSE=2   (manage open positions)

WHY THIS SHAPE:
  - The model learns direction on its own (the code NEVER picks buy vs sell —
    DESIGN_DECISIONS.md #2). Under a force_in_and_gate phase, only the FLAT
    option of DIRECTION is masked when flat, so the agent MUST choose BUY or
    SELL — but which one is entirely the policy's choice.
  - Lot size is the agent's continuous decision (PPO sizing head).
  - Exit lets the agent manage/scale/close positions it already holds.

There is no DQN flat-index space anymore. `DIRECTION_DIM` / `EXIT_DIM` are the
categorical sizes PPO's policy heads use; lot is a single squashed continuous
output. Other modules import these constants — never hardcode the integers.
"""
from __future__ import annotations

from typing import Tuple

# ── Direction head ───────────────────────────────────────────────────────────
FLAT = 0
BUY = 1
SELL = 2
DIRECTION_DIM = 3
DIRECTION_NAMES = {FLAT: "FLAT", BUY: "BUY", SELL: "SELL"}

# Back-compat alias: some risk/fill code refers to HOLD meaning "no new position".
HOLD = FLAT

# ── Exit head ────────────────────────────────────────────────────────────────
EXIT_HOLD = 0
EXIT_REDUCE = 1
EXIT_CLOSE = 2
EXIT_DIM = 3
EXIT_NAMES = {EXIT_HOLD: "HOLD", EXIT_REDUCE: "REDUCE", EXIT_CLOSE: "CLOSE"}

# ── Lot head (continuous) ────────────────────────────────────────────────────
LOT_DIM = 1          # one squashed continuous scalar in [0,1]
MIN_LOT = 0.01       # MT5 minimum


def map_lot(raw: float, max_lot: float, min_lot: float = MIN_LOT) -> float:
    """
    Map a raw policy scalar in [0,1] to an actual lot in [min_lot, max_lot].
    Clamped both ends; rounded to 0.01. The Policy/PositionSizer apply the same
    mapping so training and live agree.
    """
    raw = max(0.0, min(1.0, float(raw)))
    lot = min_lot + raw * (max_lot - min_lot)
    lot = max(min_lot, min(lot, float(max_lot)))
    return round(lot, 2)


def describe(direction: int, lot_raw: float, exit_act: int, max_lot: float = 2.0
             ) -> dict:
    """Human-readable expansion of a structured PPO action (dashboard/Jordan)."""
    return {
        "direction": DIRECTION_NAMES.get(int(direction), "FLAT"),
        "lot": map_lot(lot_raw, max_lot),
        "exit": EXIT_NAMES.get(int(exit_act), "HOLD"),
    }


def decode(action: Tuple[int, float, int], max_lot: float = 2.0) -> dict:
    """
    Decode a structured action tuple (direction, lot_raw, exit_act) into concrete
    trade fields. Kept as the single decode point used by env / live_runner.
    """
    direction, lot_raw, exit_act = action
    return {
        "direction": int(direction),
        "lot": map_lot(lot_raw, max_lot),
        "exit": int(exit_act),
    }
