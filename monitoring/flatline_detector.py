"""
monitoring/flatline_detector.py
────────────────────────────────────────────────────────────────────────────
Watches eval PASS-rate over time. Flatline = PASS rate has not improved by more
than 1% across 10 consecutive eval runs. On flatline, fires an alert and asks
irac_engine for a remediation card (stored for Jordan).
"""
from __future__ import annotations

from collections import deque


class FlatlineDetector:
    def __init__(self, window=10, min_improvement=0.01, alert_dispatcher=None):
        self.window = window
        self.min_improvement = min_improvement
        self.alert = alert_dispatcher
        self._history = deque(maxlen=window)
        self.last_irac = None

    def record(self, pass_rate: float) -> bool:
        """Record an eval's pass_rate. Returns True if a flatline is detected."""
        self._history.append(float(pass_rate))
        if len(self._history) < self.window:
            return False
        spread = max(self._history) - min(self._history)
        if spread <= self.min_improvement:
            self._fire(pass_rate)
            return True
        return False

    def _fire(self, pass_rate):
        try:
            from jordan.irac_engine import generate_irac
            self.last_irac = generate_irac("flatline_detected",
                                           {"pass_rate": round(pass_rate, 3),
                                            "eval_count": self.window})
        except Exception:
            self.last_irac = None
        if self.alert is not None:
            self.alert.fire("FLATLINE", f"PASS rate flat at {pass_rate:.1%} "
                            f"over {self.window} evals",
                            {"pass_rate": pass_rate})
