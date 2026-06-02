"""
core/risk/daily_guard.py
────────────────────────────────────────────────────────────────────────────
DailyGuard — enforces drawdown halts and records PASS/FAIL (HARD RULE 7).

FTMO mode:
  - trailing DD baseline = prior-day close balance (day 1 = initial balance)
  - HALT if equity drops daily_max_dd_pct below baseline
  - HALT if trade count reaches max_trades_per_day (default 800)
  - reset_day() at midnight server time rolls the baseline forward

Beast mode:
  - trailing DD from PEAK equity (high-water mark; never decreases)
  - HALT if equity drops trailing_dd_from_peak_pct from peak
  - no trade-count cap unless explicitly set

Both:
  - force_halt() halts immediately (EMERGENCY HALT button)
  - get_status() -> dict; on halt, calls alert_dispatcher.fire (never crashes)

PASS/FAIL (RULE 7), evaluated against the day's starting balance:
  PASS  = end_balance >= start * (1 + target_pct)   (regardless of DD)
  FAIL  = DD breach occurred AND end_balance < start * (1 + target_pct)
"""
from __future__ import annotations

from typing import Optional


class DailyGuard:
    def __init__(self, mode: str, initial_balance: float, cfg: dict,
                 alert_dispatcher=None):
        self.mode = (mode or "ftmo").lower()
        self.initial_balance = float(initial_balance)
        self.cfg = cfg or {}
        self.alert = alert_dispatcher

        self.target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
        self.max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
        self.beast_dd_pct = float(cfg.get("BEAST_TRAILING_DD_PCT", 0.05))
        self.max_trades = cfg.get("MAX_TRADES_PER_DAY", 800)

        self.equity = self.initial_balance
        self.baseline = self.initial_balance       # FTMO: prior-day close
        self.peak = self.initial_balance           # Beast: high-water mark
        self.day_start = self.initial_balance
        self.trades_today = 0
        self.halted = False
        self.dd_breached = False
        self._forced = False

    # ── update each bar/trade ──────────────────────────────────────────────
    def update(self, equity: float, trade_count: Optional[int] = None) -> dict:
        self.equity = float(equity)
        if trade_count is not None:
            self.trades_today = int(trade_count)
        self.peak = max(self.peak, self.equity)

        if self.mode == "beast":
            dd = (self.peak - self.equity) / (self.peak + 1e-9)
            if dd >= self.beast_dd_pct:
                self._trigger_halt(f"Beast DD {dd:.2%} >= {self.beast_dd_pct:.2%}")
        else:  # ftmo
            dd = (self.baseline - self.equity) / (self.baseline + 1e-9)
            if dd >= self.max_dd_pct:
                self._trigger_halt(f"FTMO DD {dd:.2%} >= {self.max_dd_pct:.2%}")
            if self.max_trades and self.trades_today >= int(self.max_trades):
                self._trigger_halt(f"Trade cap {self.trades_today} >= {self.max_trades}")

        if dd >= (self.max_dd_pct if self.mode != "beast" else self.beast_dd_pct):
            self.dd_breached = True
        return self.get_status()

    def _trigger_halt(self, reason: str):
        if not self.halted:
            self.halted = True
            self._fire("CRITICAL", f"HALT: {reason}")

    def force_halt(self):
        """Immediate halt regardless of DD (EMERGENCY HALT button)."""
        self.halted = True
        self._forced = True
        self._fire("EMERGENCY_HALT", "Manual EMERGENCY HALT engaged")

    def reset_day(self):
        """Roll baseline to current equity for a new trading day (FTMO)."""
        self.baseline = self.equity
        self.day_start = self.equity
        self.trades_today = 0
        self.dd_breached = False
        if not self._forced:
            self.halted = False

    # ── PASS/FAIL ───────────────────────────────────────────────────────────
    def pass_fail(self) -> str:
        target = self.day_start * (1.0 + self.target_pct)
        if self.equity >= target:
            return "PASS_NO_BREACH" if not self.dd_breached else "PASS"
        return "FAIL" if self.dd_breached else "IN_PROGRESS"

    def get_status(self) -> dict:
        baseline = self.peak if self.mode == "beast" else self.baseline
        dd_pct = (baseline - self.equity) / (baseline + 1e-9)
        return {
            "mode": self.mode,
            "equity": self.equity,
            "peak": self.peak,
            "baseline": self.baseline,
            "dd_pct": dd_pct,
            "trades_today": self.trades_today,
            "halted": self.halted,
            "pass_fail": self.pass_fail(),
        }

    def _fire(self, level: str, message: str):
        """Dispatch an alert; never crash if the dispatcher fails."""
        if self.alert is None:
            return
        try:
            self.alert.fire(level, message)
        except Exception as exc:                           # pragma: no cover
            print(f"[daily_guard] alert failed: {exc}", flush=True)
