"""
core/reward/shaper.py
────────────────────────────────────────────────────────────────────────────
EpisodeRewardShaper — ported from gpu_rl_trading/training/train.py (REPO1).

Computes an episode-end bonus from a potential function Φ that rewards
CONSISTENCY over days/weeks/months rather than one-off spikes:

    Φ = (pass_rate × avg_ret_norm) / (1 + λ × avg_dd_norm)

Because Φ is normalized to configured targets, the same α/clip work for any
daily_target_pct / max_dd_pct combination (no retuning).

Changes vs REPO1 (per STEP 4.7):
  (a) PASS rule = end_balance >= start_balance * 1.025 (RULE 7).
  (b) weekly_consistency_bonus: if this week's 7-day pass rate beats last week's,
      add +0.02. Tracks a 14-entry deque of daily pass/fail outcomes.
"""
from __future__ import annotations

from collections import deque
from typing import List

import numpy as np


class EpisodeRewardShaper:
    def __init__(self, cfg: dict):
        self.target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
        self.max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
        self.alpha = float(cfg.get("SHAPE_ALPHA", 0.006))
        self.clip_val = float(cfg.get("SHAPE_CLIP", 0.006))
        self.lam = float(cfg.get("SHAPE_LAMBDA", 5.0))
        self.warmup = int(cfg.get("SHAPE_WARMUP", 150))
        self.weekly_bonus = float(cfg.get("WEEKLY_BONUS", 0.02))
        self.window = 20

        self._phi_history: List[float] = []
        self.global_ep = 0
        # 14-day deque of daily PASS(1)/FAIL(0) outcomes for the weekly bonus.
        self._daily_pass = deque(maxlen=14)

    # ── Φ potential ─────────────────────────────────────────────────────────
    def _phi(self, pass_rate: float, avg_ret: float, avg_dd: float) -> float:
        ret_norm = avg_ret / (self.target_pct + 1e-8)
        dd_norm = avg_dd / (self.max_dd_pct + 1e-8)
        return (pass_rate * max(ret_norm, 0.0)) / (1.0 + self.lam * dd_norm)

    def compute_bonus(self, daily_log: list) -> float:
        """
        daily_log: list of per-day dicts with keys {pass: bool, ret: float, dd: float}.
        Returns a scalar episode-end bonus (0.0 during warm-up).
        """
        if not daily_log:
            return 0.0
        # record daily PASS/FAIL for the weekly bonus
        for d in daily_log:
            self._daily_pass.append(1 if d.get("pass") else 0)

        pass_rate = float(np.mean([1.0 if d.get("pass") else 0.0 for d in daily_log]))
        avg_ret = float(np.mean([d.get("ret", 0.0) for d in daily_log]))
        avg_dd = float(np.mean([d.get("dd", 0.0) for d in daily_log]))
        phi = self._phi(pass_rate, avg_ret, avg_dd)

        self._phi_history.append(phi)
        if len(self._phi_history) > self.window:
            self._phi_history = self._phi_history[-self.window:]

        if self.global_ep < self.warmup:
            return 0.0

        phi_smooth = float(np.mean(self._phi_history[:-1])) if len(self._phi_history) > 1 else phi
        sigma = float(np.std(self._phi_history)) or 1.0
        bonus = self.alpha * (phi - phi_smooth) / sigma
        bonus = float(np.clip(bonus, -self.clip_val, self.clip_val))
        return bonus + self.weekly_consistency_bonus()

    def weekly_consistency_bonus(self) -> float:
        """
        +WEEKLY_BONUS when the most recent 7-day pass rate exceeds the prior
        7-day pass rate. Needs a full 14-day window; returns 0.0 otherwise.
        """
        if len(self._daily_pass) < 14:
            return 0.0
        days = list(self._daily_pass)
        prev_week = sum(days[:7]) / 7.0
        this_week = sum(days[7:]) / 7.0
        return self.weekly_bonus if this_week > prev_week else 0.0
