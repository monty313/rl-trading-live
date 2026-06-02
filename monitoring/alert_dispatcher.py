"""
monitoring/alert_dispatcher.py
────────────────────────────────────────────────────────────────────────────
Routes alerts to multiple channels. NEVER crashes if a channel fails — it logs
the failure and continues (HARD RULE: alert failures must not break trading).

Channels (in order):
  1. Streamlit session state (st.session_state["alerts"]) — best-effort
  2. Telegram (only if TELEGRAM_BOT_TOKEN set; failures -> logs/alert_failures.txt)
  3. MT5 push (mt5.send_notification) — only if an mt5 module is provided

levels: INFO | WARNING | CRITICAL | EMERGENCY_HALT | FLATLINE
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

VALID_LEVELS = {"INFO", "WARNING", "CRITICAL", "EMERGENCY_HALT", "FLATLINE"}


class AlertDispatcher:
    def __init__(self, mt5_module=None, telegram_sender=None,
                 failure_log="logs/alert_failures.txt"):
        self.mt5 = mt5_module
        self.telegram = telegram_sender   # object with .send(token, chat_id, text)
        self.failure_log = failure_log
        self.alerts = []                  # mirrors st.session_state["alerts"]

    def fire(self, level: str, message: str, details: dict = None):
        level = level if level in VALID_LEVELS else "INFO"
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "level": level, "message": message, "details": details or {}}
        # 1) session state
        self.alerts.append(entry)
        try:
            import streamlit as st
            st.session_state.setdefault("alerts", []).append(entry)
        except Exception:
            pass
        # 2) Telegram
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if token and self.telegram is not None:
            try:
                self.telegram.send(token, chat, f"[{level}] {message}")
            except Exception as exc:
                self._log_failure(f"telegram: {exc}")
        # 3) MT5 push
        if self.mt5 is not None:
            try:
                self.mt5.send_notification(f"[{level}] {message}")
            except Exception as exc:
                self._log_failure(f"mt5: {exc}")
        return entry

    def _log_failure(self, msg: str):
        try:
            os.makedirs(os.path.dirname(self.failure_log) or ".", exist_ok=True)
            with open(self.failure_log, "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
        except Exception:
            pass
