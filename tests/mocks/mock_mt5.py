"""
tests/mocks/mock_mt5.py
In-memory stand-in for the MetaTrader5 module so broker code can be tested on
Colab/Linux (where the real MT5 terminal cannot run). Mirrors the subset of the
MT5 API that mt5_adapter.py uses.
"""
from __future__ import annotations

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_RETCODE_DONE = 10009


class _SymbolInfo:
    def __init__(self, name): self.name = name; self.visible = True


class MockMT5:
    """A callable mock with the MT5 functions the adapter needs."""
    # expose MT5 constants as instance attributes too (adapter reads mt5.ORDER_TYPE_*)
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE

    def __init__(self, known_symbols=("EURUSD",)):
        self.known = set(known_symbols)
        self._positions = []
        self._ticket = 1000
        self.sent_orders = []
        self._balance = 100000.0

    def initialize(self, **kwargs): return True
    def login(self, *a, **k): return True
    def shutdown(self): return None

    def symbol_info(self, name):
        return _SymbolInfo(name) if name in self.known else None

    def symbol_select(self, name, enable=True):
        return name in self.known

    def order_send(self, request):
        self.sent_orders.append(request)
        self._ticket += 1
        return type("Result", (), {"retcode": TRADE_RETCODE_DONE,
                                   "order": self._ticket,
                                   "price": request.get("price", 0.0)})()

    def positions_get(self, **k): return list(self._positions)

    def account_info(self):
        return type("Acct", (), {"balance": self._balance, "equity": self._balance,
                                 "margin": 0.0, "margin_free": self._balance})()

    def send_notification(self, msg): return True
