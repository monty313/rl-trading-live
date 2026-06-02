"""
tests/mocks/mock_telegram.py
Mock Telegram sender to verify alert_dispatcher calls the Telegram channel
without making real network requests.
"""
from __future__ import annotations


class MockTelegram:
    def __init__(self, fail=False):
        self.fail = fail
        self.messages = []

    def send(self, token, chat_id, text):
        if self.fail:
            raise RuntimeError("mock telegram failure")
        self.messages.append({"token": token, "chat_id": chat_id, "text": text})
        return True
