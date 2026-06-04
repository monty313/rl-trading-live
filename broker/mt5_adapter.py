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
import time
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
        self._last_config = None      # remembered for reconnect()

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
        self._last_config = dict(config)
        mt5 = self._ensure_mt5()
        # credentials from env (HARD RULE 2)
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        if not mt5.initialize():
            # terminal/connection not up — caller (or initialize_with_retry) retries.
            return False
        if login and password and server:
            mt5.login(int(login), password=password, server=server)
        # symbol alias auto-resolve. If NONE of the aliases resolve the symbol is
        # unavailable on this broker — return False (never silently trade a wrong
        # symbol or default to the bare name).
        aliases = config.get("symbol_aliases", {})
        wanted = config.get("instruments", ["EURUSD"])[0]
        for alias in aliases.get(wanted, [wanted]):
            if mt5.symbol_info(alias) is not None:
                mt5.symbol_select(alias, True)
                self.symbol = alias
                return True
        self.symbol = None
        return False

    def initialize_with_retry(self, config: dict, max_attempts: int = 5,
                              base_delay: float = 0.0) -> bool:
        """Initialize with bounded exponential backoff (connection-failure path).

        Returns True on the first successful initialize(), else False after
        `max_attempts`. `base_delay` is the first sleep (doubles each retry); it
        defaults to 0.0 so tests run instantly — production passes e.g. 1.0s.
        Every attempt is announced so an operator sees the link coming up."""
        delay = float(base_delay)
        for attempt in range(1, int(max_attempts) + 1):
            try:
                if self.initialize(config):
                    print(f"[mt5] connected on attempt {attempt}", flush=True)
                    return True
            except Exception as exc:                       # transient link error
                print(f"[mt5] initialize attempt {attempt} raised: {exc}", flush=True)
            print(f"[mt5] connect attempt {attempt}/{max_attempts} failed; "
                  f"retrying in {delay:.1f}s", flush=True)
            if attempt < int(max_attempts) and delay > 0:
                time.sleep(delay)
            delay = delay * 2 if delay > 0 else base_delay
        print(f"[mt5] FAILED to connect after {max_attempts} attempts", flush=True)
        return False

    def reconnect(self, max_attempts: int = 5, base_delay: float = 0.0) -> bool:
        """Re-establish the link after a drop, reusing the last initialize config."""
        if self._last_config is None:
            return False
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        return self.initialize_with_retry(self._last_config, max_attempts, base_delay)

    def send_order(self, order: dict) -> dict:
        # HARD RULE 5: gate FIRST, no bypass.
        if self.gate is not None and not self.gate.approve(order):
            return {"status": "BLOCKED", "fill_price": None, "order_id": None,
                    "error": "trade_gate rejected"}
        mt5 = self._ensure_mt5()
        otype = mt5.ORDER_TYPE_BUY if order.get("direction") in ("BUY", 1) \
            else mt5.ORDER_TYPE_SELL
        req_vol = float(order.get("lot", 0.01))
        request = {"symbol": self.symbol or order.get("symbol"),
                   "volume": req_vol, "type": otype,
                   "price": order.get("entry"), "sl": order.get("sl"),
                   "tp": order.get("tp")}
        try:
            result = mt5.order_send(request)
        except Exception as exc:
            # Link dropped mid-send: surface DISCONNECTED so the runner can
            # reconnect. NOTHING was filled -> no PnL is recorded (HARD RULE: a
            # rejected/failed order must never be booked as a loss).
            return {"status": "DISCONNECTED", "fill_price": None,
                    "order_id": None, "filled_volume": 0.0,
                    "requested_volume": req_vol, "error": str(exc)}
        ok = getattr(result, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        filled = float(getattr(result, "volume", req_vol if ok else 0.0) or 0.0)
        # A DONE retcode with a smaller executed volume is a PARTIAL fill — report
        # it explicitly (status PARTIAL) so the caller never assumes full size.
        if ok and filled + 1e-9 < req_vol:
            status = "PARTIAL"
        elif ok:
            status = "FILLED"
        else:
            status = "REJECTED"
        return {"status": status,
                "fill_price": getattr(result, "price", None),
                "order_id": getattr(result, "order", None),
                "filled_volume": filled if ok else 0.0,
                "requested_volume": req_vol,
                "error": None if ok else "order_send retcode != DONE"}

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
