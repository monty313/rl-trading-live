"""
core/risk/trade_gate.py
────────────────────────────────────────────────────────────────────────────
TradeGate — the single approval choke point for every order (HARD RULE 5).

Every order execution path MUST call trade_gate.approve(order) -> bool before
sending. There is no bypass. approve() returns False when:
  - the DailyGuard is halted (DD breach, trade cap, or force_halt)
  - consent has not been granted (consent_flow unresolved)
Rejected orders are logged as BLOCKED to logs/daily_trade_log.csv; approved
orders are logged as APPROVED. The log is created on first write.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone


class TradeGate:
    def __init__(self, daily_guard=None, log_path: str = "logs/daily_trade_log.csv"):
        self.daily_guard = daily_guard
        self.log_path = log_path
        self.consent_granted = True   # default operation; consent_flow may set False

    def set_consent(self, granted: bool):
        """consent_flow.py toggles this; False blocks approve() until resolved."""
        self.consent_granted = bool(granted)

    def approve(self, order: dict) -> bool:
        """
        Return True only if the guard is not halted AND consent is granted.
        Logs the decision (APPROVED / BLOCKED) regardless of outcome.
        """
        halted = bool(self.daily_guard.get_status()["halted"]) if self.daily_guard else False
        approved = (not halted) and self.consent_granted
        reason = "APPROVED" if approved else (
            "BLOCKED_HALTED" if halted else "BLOCKED_NO_CONSENT")
        self._log(order, "APPROVED" if approved else "BLOCKED", reason)
        return approved

    def _log(self, order: dict, status: str, reason: str):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "status", "reason", "symbol",
                            "direction", "lot", "sl", "tp", "entry"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                status, reason,
                order.get("symbol", ""), order.get("direction", ""),
                order.get("lot", ""), order.get("sl", ""),
                order.get("tp", ""), order.get("entry", ""),
            ])
