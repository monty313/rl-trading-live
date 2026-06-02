"""
jordan/vitals_daemon.py
────────────────────────────────────────────────────────────────────────────
Background reader (15-min interval) that compiles plain-English "vitals" for the
dashboard. READ-ONLY (HARD RULE 6): reads files/queues, never writes trading
state. Never crashes if a source is missing — shows "N/A".

Sources: trading_policy.yaml, logs/daily_trade_log.csv, manifest.json,
logs/accuracy_sma.json, logs/last_test_results.json (the complete system test),
and approved market URLs from jordan_sources.yaml.

The COMPLETE SYSTEM TEST results are surfaced here so Jordan can report build
health (per the user's request to "allow jordan access to it also").
"""
from __future__ import annotations

import json
import os
import threading
import time


class VitalsDaemon:
    def __init__(self, repo_root=".", interval_sec=900):
        self.repo_root = repo_root
        self.interval = interval_sec
        self.latest = "Jordan vitals: initializing..."
        self._thread = None
        self._stop = threading.Event()

    # ── safe readers ─────────────────────────────────────────────────────────
    def _read_json(self, rel):
        try:
            with open(os.path.join(self.repo_root, rel)) as f:
                return json.load(f)
        except Exception:
            return None

    def _test_health(self) -> str:
        # reads the complete-system-test cache; Jordan-accessible by design
        try:
            from tests.run_all_tests import jordan_summary
            return jordan_summary()
        except Exception:
            r = self._read_json("logs/last_test_results.json")
            return (r or {}).get("summary", "Tests: N/A")

    def compose(self) -> str:
        acc = self._read_json("logs/accuracy_sma.json") or {}
        manifest = self._read_json("RL-Trading-Checkpoints/manifest.json") \
            or self._read_json("logs/manifest.json") or {}
        live = "N/A"
        cks = (manifest or {}).get("checkpoints", {})
        if "live_trading.pt" in cks:
            live = f"live_trading.pt (Φ={cks['live_trading.pt'].get('phi','?')})"
        lines = [
            f"Jordan System Vitals – {time.strftime('%Y-%m-%d %H:%M')}",
            f"Live Accuracy SMA (last 20): {acc.get('accuracy_sma','N/A')}%",
            f"Live Model: {live}",
            f"Build Health: {self._test_health()}",
            "Market Context: N/A",
        ]
        self.latest = "\n".join(str(x) for x in lines)
        return self.latest

    # ── thread control ────────────────────────────────────────────────────────
    def _loop(self):
        while not self._stop.is_set():
            try:
                self.compose()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.compose()   # immediate first read
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
