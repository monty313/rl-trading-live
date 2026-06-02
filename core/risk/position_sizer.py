"""
core/risk/position_sizer.py
────────────────────────────────────────────────────────────────────────────
Maps a lot-bucket index (0..6) to an actual lot size, clamped to the account's
max_lot from trading_policy.yaml. Enforces the MT5 minimum (0.01) and emits a
risk warning when the resulting notional risk exceeds ~5% of balance.

This is the single source of bucket->lot truth alongside action_space.get_lot
(they use the same table). Training uses a coarse internal sizing; live trading
uses this sizer for exact lots before send_order.
"""
from __future__ import annotations

from core.agent.action_space import get_lot

MIN_LOT = 0.01
RISK_WARN_FRACTION = 0.05   # warn if lot notional risk > 5% of balance


class PositionSizer:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}

    def size(self, lot_idx: int, max_lot: float, balance: float = None,
             sl_pips: int = None, pip_value: float = 0.0001) -> float:
        """
        Resolve a lot-bucket index to a clamped lot size.

        Args:
            lot_idx  : 0..6 (6 -> max_lot)
            max_lot  : account ceiling from trading_policy.yaml
            balance  : optional account balance for the 5% risk warning
            sl_pips  : optional stop distance for the risk estimate
            pip_value: instrument pip value (default FX 0.0001)
        Returns the final lot (>= 0.01, <= max_lot).
        """
        lot = get_lot(lot_idx, max_lot)             # already clamped + floored
        lot = max(MIN_LOT, min(lot, float(max_lot)))

        # Optional risk check (non-fatal warning only — never blocks here).
        if balance and sl_pips:
            # notional risk ≈ lot * 100_000 * sl_pips * pip_value (FX-style)
            risk = lot * 100_000.0 * sl_pips * pip_value
            if risk > RISK_WARN_FRACTION * float(balance):
                print(f"[sizer] WARNING: lot {lot} risks "
                      f"{risk / balance:.1%} of balance (> "
                      f"{RISK_WARN_FRACTION:.0%}) at {sl_pips} pip SL", flush=True)
        return round(lot, 2)
