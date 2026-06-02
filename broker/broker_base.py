"""
broker/broker_base.py
Abstract BrokerAdapter — the interface every concrete broker implements. Keeps
live_runner broker-agnostic so MT5 / Tradovate / Schwab can be swapped later.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BrokerAdapter(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> bool: ...
    @abstractmethod
    def send_order(self, order: dict) -> dict: ...   # {status, fill_price, order_id, error}
    @abstractmethod
    def get_positions(self) -> list: ...
    @abstractmethod
    def get_account_info(self) -> dict: ...          # {balance, equity, margin, free_margin}
    @abstractmethod
    def close_position(self, ticket: int) -> dict: ...
    @abstractmethod
    def shutdown(self) -> None: ...
