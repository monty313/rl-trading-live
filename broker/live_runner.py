"""
broker/live_runner.py
────────────────────────────────────────────────────────────────────────────
Per-bar live inference loop (runs on the Windows MT5 machine, NOT in Colab).

On each new M1 bar close:
  1. fetch latest candle (MT5 or mock)
  2. build observation via core/env/indicators (PARITY REQUIRED)
  3. agent.select_action(obs, deterministic=True)
  4. action_space.decode -> (direction, lot_idx, sl_idx, tp_idx)
  5. intrabar_fills.compute_fill -> entry/sl/tp
  6. trade_gate.approve(order) -> if False, log BLOCKED, skip
  7. mt5_adapter.send_order(order) -> log result
  8. update rolling 20-trade accuracy SMA -> logs/accuracy_sma.json

Loads live_trading.pt from the checkpoint dir. Handles daily reset at midnight.
NOTE (HARD RULE 3): never connects to a real MT5 account unless the user has
explicitly enabled live mode AND provided a real adapter; default uses the mock.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

import torch

from core.agent.action_space import decode, get_lot, get_sl_pips, get_tp_pips, HOLD
from core.env.intrabar_fills import compute_fill


class LiveRunner:
    def __init__(self, agent, adapter, trade_gate, daily_guard, cfg, policy,
                 instrument="EURUSD", accuracy_path="logs/accuracy_sma.json",
                 heartbeat_path="logs/heartbeat_live_runner.txt"):
        self.agent = agent
        self.adapter = adapter
        self.gate = trade_gate
        self.guard = daily_guard
        self.cfg = cfg
        self.policy = policy
        self.instrument = instrument
        self.accuracy_path = accuracy_path
        self.heartbeat_path = heartbeat_path
        self._recent = deque(maxlen=20)   # rolling win/loss for accuracy SMA
        self._last_hb = 0.0

    def step_bar(self, obs: torch.Tensor, bar: dict, max_lot: float = 2.0,
                 atr_14: float = None) -> dict:
        """Process one closed M1 bar. Returns the order result dict."""
        action = self.agent.select_action(obs, deterministic=True)
        direction, lot_idx, sl_idx, tp_idx = decode(action)
        if direction == HOLD:
            self._heartbeat()
            return {"status": "HOLD", "action": action}

        lot = get_lot(lot_idx, max_lot)
        fill = compute_fill(bar, direction, get_sl_pips(sl_idx),
                            get_tp_pips(tp_idx), self.instrument, self.policy, atr_14)
        order = {"symbol": self.instrument, "direction": direction, "lot": lot,
                 "entry": fill["entry"], "sl": fill["sl"], "tp": fill["tp"]}

        if not self.gate.approve(order):
            self._heartbeat()
            return {"status": "BLOCKED", "order": order}

        result = self.adapter.send_order(order)
        self._heartbeat()
        return result

    def record_trade_close(self, won: bool):
        """Call when a trade closes; updates the rolling 20-trade accuracy SMA."""
        self._recent.append(1 if won else 0)
        acc = (sum(self._recent) / len(self._recent) * 100.0) if self._recent else 0.0
        os.makedirs(os.path.dirname(self.accuracy_path) or ".", exist_ok=True)
        with open(self.accuracy_path, "w") as f:
            json.dump({"accuracy_sma": acc, "n": len(self._recent),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        return acc

    def reset_day(self):
        self.guard.reset_day()

    def _heartbeat(self):
        now = time.time()
        if now - self._last_hb >= 60:
            os.makedirs(os.path.dirname(self.heartbeat_path) or ".", exist_ok=True)
            with open(self.heartbeat_path, "w") as f:
                json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "status": "running"}, f)
            self._last_hb = now
