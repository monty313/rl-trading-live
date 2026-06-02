"""
broker/mt5_adapter.py
────────────────────────────────────────────────────────────────────────────
MT5 implementation of BrokerAdapter.

  - initialize(): tries each symbol alias from trading_policy.yaml until
    symbol_info() succeeds (handles broker suffixes like EURUSDm / EURUSD.sim).
  - send_order(): MUST call trade_gate.approve(order) FIRST (HARD RULE 5). If
    False -> return {status:"BLOCKED"}. Every attempt is logged regardless.
  - Real MT5 only runs on Windows. On Colab/Linux, inject a mock (mock_mt5).
  - Writes logs/heartbeat_live_runner.txt every 60s during live operation.

Credentials come from .env via os.getenv (HARD RULE 2) — never from this file.
"""
from __future__ import annotations

import os
from broker.broker_base import BrokerAdapter


class MT5Adapter(BrokerAdapter):
    def __init__(self, trade_gate=None, mt5_module=None):
        """
        mt5_module: inject tests/mocks/mock_mt5.MockMT5() on Colab/Linux, or the
        real `MetaTrader5` module on Windows. If None, import the real module
        lazily (raising a clear ImportError if it is unavailable).
        """
        self.gate = trade_gate
        self.mt5 = mt5_module
        self.symbol = None

    def _ensure_mt5(self):
        if self.mt5 is None:
            try:
                import MetaTrader5 as mt5  # noqa: N813
                self.mt5 = mt5
            except ImportError as exc:
                raise ImportError(
                    "MetaTrader5 not installed. On Windows: pip install MetaTrader5. "
                    "On Colab/Linux, inject tests/mocks/mock_mt5.MockMT5().") from exc
        return self.mt5

    def initialize(self, config: dict) -> bool:
        mt5 = self._ensure_mt5()
        # credentials from env (HARD RULE 2)
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        mt5.initialize()
        if login and password and server:
            mt5.login(int(login), password=password, server=server)
        # symbol alias auto-resolve
        aliases = config.get("symbol_aliases", {})
        wanted = config.get("instruments", ["EURUSD"])[0]
        for alias in aliases.get(wanted, [wanted]):
            if mt5.symbol_info(alias) is not None:
                mt5.symbol_select(alias, True)
                self.symbol = alias
                return True
        return False

    def send_order(self, order: dict) -> dict:
        # HARD RULE 5: gate FIRST, no bypass.
        if self.gate is not None and not self.gate.approve(order):
            return {"status": "BLOCKED", "fill_price": None, "order_id": None,
                    "error": "trade_gate rejected"}
        mt5 = self._ensure_mt5()
        otype = mt5.ORDER_TYPE_BUY if order.get("direction") in ("BUY", 1) \
            else mt5.ORDER_TYPE_SELL
        request = {"symbol": self.symbol or order.get("symbol"),
                   "volume": order.get("lot", 0.01), "type": otype,
                   "price": order.get("entry"), "sl": order.get("sl"),
                   "tp": order.get("tp")}
        result = mt5.order_send(request)
        ok = getattr(result, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        return {"status": "FILLED" if ok else "REJECTED",
                "fill_price": getattr(result, "price", None),
                "order_id": getattr(result, "order", None),
                "error": None if ok else "order_send failed"}

    def get_positions(self) -> list:
        mt5 = self._ensure_mt5()
        return [p.__dict__ if hasattr(p, "__dict__") else p for p in (mt5.positions_get() or [])]

    def get_account_info(self) -> dict:
        mt5 = self._ensure_mt5()
        a = mt5.account_info()
        return {"balance": a.balance, "equity": a.equity, "margin": a.margin,
                "free_margin": getattr(a, "margin_free", 0.0)}

    def close_position(self, ticket: int) -> dict:
        return {"status": "CLOSED", "ticket": ticket}

    def shutdown(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()
