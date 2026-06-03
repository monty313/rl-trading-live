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

PASS/FAIL — STRICTLY BINARY (ftmo_rules_fix.md RULES 1-2), evaluated against the
day's OPENING balance plus a FIXED daily increment:
  daily_increment = initial_balance * target_pct   # fixed $, computed once at open
  daily_target    = day_start + daily_increment
  PASS  = (equity >= daily_target)   # at end of day OR at a DD halt (RULE 3)
  FAIL  = everything else (a DD breach does NOT auto-fail; balance-at-halt decides)
There is NO "OK" and NO "SKIP" — pass_fail() returns only "PASS" or "FAIL".

SINGLE SOURCE OF TRUTH for the FTMO principles (daily target, trailing DD, binary
classification, runtime config) is the principles block at the top of
core/env/environment.py — read it first. That env is the authoritative training/
classification path (per-bar peak that resets each day). THIS guard is the live/
backtest risk gate; it mirrors the same binary PASS/FAIL target rule, and its DD
halt enforces the same "halt the day on a trailing-DD breach" principle (the FTMO
branch trails from the day baseline, the beast branch from the high-water peak).
target_pct / max_dd_pct here are read from cfg at RUNTIME — never hardcoded.
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
        # FIXED daily profit increment (RULE 1): a flat dollar amount = initial *
        # target_pct, computed ONCE at open. The target each day is the day's
        # OPENING balance + this increment — never target_pct of the live balance.
        self.daily_increment = self.initial_balance * self.target_pct
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

    # ── PASS/FAIL — BINARY (RULES 1-2) ────────────────────────────────────────
    def pass_fail(self) -> str:
        """PASS iff equity has reached the day's fixed-increment target; else FAIL.
        Binary — a DD breach does NOT auto-fail (balance-at-halt decides), and a
        day under target is FAIL regardless of whether a breach occurred."""
        daily_target = self.day_start + self.daily_increment
        return "PASS" if self.equity >= daily_target else "FAIL"

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
