"""
broker/schwab_stub.py — future broker. Intentional stub (per spec).
Implements BrokerAdapter so the interface is documented; every method raises
NotImplementedError until Schwab support is built.
"""
from __future__ import annotations
from broker.broker_base import BrokerAdapter

_MSG = "Schwab adapter not implemented yet — future broker support."


class SchwabAdapter(BrokerAdapter):
    def initialize(self, config: dict) -> bool: raise NotImplementedError(_MSG)
    def send_order(self, order: dict) -> dict: raise NotImplementedError(_MSG)
    def get_positions(self) -> list: raise NotImplementedError(_MSG)
    def get_account_info(self) -> dict: raise NotImplementedError(_MSG)
    def close_position(self, ticket: int) -> dict: raise NotImplementedError(_MSG)
    def shutdown(self) -> None: raise NotImplementedError(_MSG)
