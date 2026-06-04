"""
tests/mocks/mock_mt5.py
────────────────────────────────────────────────────────────────────────────
In-memory stand-in for the MetaTrader5 module so broker code can be tested on
Colab/Linux (where the real MT5 terminal cannot run). Mirrors the subset of the
MT5 API that mt5_adapter.py / live_runner.py use.

PASS-2 STEP 9 expanded the mock to script the failure surface a real MT5 link
exhibits, so the live runner's resilience can be exercised entirely offline:
  • initialize() can be told to fail the first N attempts then succeed
    (connection-failure -> retry/backoff path);
  • a symbol can be marked unavailable (symbol_info() -> None);
  • order_send() can be told to REJECT (retcode != DONE) or PARTIALLY FILL
    (returns a smaller executed volume than requested);
  • a "link drop" can be toggled so order_send() raises (reconnect path);
  • every request is recorded in `sent_orders` so duplicate-order prevention
    and order-format assertions can inspect exactly what was transmitted.

Nothing here ever touches a real account — it is pure in-memory state.
"""
from __future__ import annotations

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_NO_MONEY = 10019


class _SymbolInfo:
    def __init__(self, name): self.name = name; self.visible = True


class _Result:
    def __init__(self, retcode, order, price, volume):
        self.retcode = retcode
        self.order = order
        self.price = price
        self.volume = volume          # executed volume (may be < requested = partial)


class MockMT5:
    """A callable mock with the MT5 functions the adapter/runner need.

    Scenario knobs (all default to the happy path):
      fail_initialize_times : int  — initialize() returns False this many times
                                      before succeeding (connection retry test).
      reject_orders         : bool — order_send() returns a non-DONE retcode.
      partial_fill_ratio    : float|None — if set in (0,1), order_send() reports
                                      an executed volume = round(req*ratio, 2).
      drop_link             : bool — order_send()/positions_get() raise
                                      ConnectionError until cleared (reconnect).
    """
    # expose MT5 constants as instance attributes too (adapter reads mt5.ORDER_TYPE_*)
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_RETCODE_REJECT = TRADE_RETCODE_REJECT

    def __init__(self, known_symbols=("EURUSD",), *,
                 fail_initialize_times=0, reject_orders=False,
                 partial_fill_ratio=None, drop_link=False):
        self.known = set(known_symbols)
        self._positions = []
        self._ticket = 1000
        self.sent_orders = []
        self._balance = 100000.0
        # scenario state
        self.fail_initialize_times = int(fail_initialize_times)
        self.reject_orders = bool(reject_orders)
        self.partial_fill_ratio = partial_fill_ratio
        self.drop_link = bool(drop_link)
        self.init_calls = 0

    # ── connection lifecycle ────────────────────────────────────────────────
    def initialize(self, **kwargs):
        self.init_calls += 1
        if self.fail_initialize_times > 0:
            self.fail_initialize_times -= 1
            return False
        return True

    def login(self, *a, **k): return True
    def shutdown(self): return None

    # toggle helpers used by reconnect tests
    def set_link_down(self): self.drop_link = True
    def set_link_up(self): self.drop_link = False

    # ── symbol resolution ───────────────────────────────────────────────────
    def symbol_info(self, name):
        return _SymbolInfo(name) if name in self.known else None

    def symbol_select(self, name, enable=True):
        return name in self.known

    # ── order transmission ──────────────────────────────────────────────────
    def order_send(self, request):
        if self.drop_link:
            raise ConnectionError("MT5 link down")
        self.sent_orders.append(request)
        self._ticket += 1
        req_vol = float(request.get("volume", 0.01))
        if self.reject_orders:
            return _Result(TRADE_RETCODE_REJECT, self._ticket,
                           request.get("price", 0.0), 0.0)
        exec_vol = req_vol
        if self.partial_fill_ratio is not None:
            exec_vol = round(req_vol * float(self.partial_fill_ratio), 2)
        return _Result(TRADE_RETCODE_DONE, self._ticket,
                       request.get("price", 0.0), exec_vol)

    def positions_get(self, **k):
        if self.drop_link:
            raise ConnectionError("MT5 link down")
        return list(self._positions)

    def account_info(self):
        return type("Acct", (), {"balance": self._balance, "equity": self._balance,
                                 "margin": 0.0, "margin_free": self._balance})()

    def send_notification(self, msg): return True
