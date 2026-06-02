"""
core/risk/position_sizer.py
────────────────────────────────────────────────────────────────────────────
Maps a lot-bucket index (0..6) to an actual lot size, clamped to the account's
max_lot from trading_policy.yaml. Enforces the MT5 minimum (0.01) and emits a
risk warning when the resulting notional risk exceeds ~5% of balance.

This maps the PPO continuous lot scalar via action_space.map_lot (single source).
(they use the same table). Training uses a coarse internal sizing; live trading
uses this sizer for exact lots before send_order.
"""
from __future__ import annotations

from core.agent.action_space import map_lot, MIN_LOT

RISK_WARN_FRACTION = 0.05   # warn if lot notional risk > 5% of balance


class PositionSizer:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}

    def size(self, lot_raw: float, max_lot: float, balance: float = None,
             sl_pips: int = None, pip_value: float = 0.0001) -> float:
        """
        Map a continuous PPO lot scalar in [0,1] to an actual lot in
        [MIN_LOT, max_lot] (same mapping as action_space.map_lot). Optional
        non-fatal 5%-of-balance risk warning.
        """
        lot = map_lot(lot_raw, max_lot)
        if balance and sl_pips:
            risk = lot * 100_000.0 * sl_pips * pip_value
            if risk > RISK_WARN_FRACTION * float(balance):
                print(f"[sizer] WARNING: lot {lot} risks {risk / balance:.1%} of "
                      f"balance (> {RISK_WARN_FRACTION:.0%}) at {sl_pips} pip SL",
                      flush=True)
        return lot
