"""
broker/account_manager.py
Reads the accounts list from trading_policy.yaml and creates one
(BrokerAdapter + DailyGuard + TradeGate) context per account. live_runner loops
over all active accounts each bar.
"""
from __future__ import annotations
from core.risk.daily_guard import DailyGuard
from core.risk.trade_gate import TradeGate
from broker.mt5_adapter import MT5Adapter


class AccountManager:
    def __init__(self, policy: dict, cfg: dict, mt5_module=None):
        self.policy = policy or {}
        self.cfg = cfg or {}
        self.mt5_module = mt5_module
        self.contexts = []

    def build(self) -> list:
        initial = float(self.cfg.get("INITIAL_EQUITY", 100000.0))
        for acct in self.policy.get("accounts", []):
            guard = DailyGuard(acct.get("mode", "ftmo"), initial, self.cfg)
            gate = TradeGate(guard)
            adapter = MT5Adapter(trade_gate=gate, mt5_module=self.mt5_module)
            self.contexts.append({"id": acct.get("id"), "config": acct,
                                  "adapter": adapter, "guard": guard, "gate": gate})
        return self.contexts

    def get_active_accounts(self) -> list:
        if not self.contexts:
            self.build()
        return self.contexts
