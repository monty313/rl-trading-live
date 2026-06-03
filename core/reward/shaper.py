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

        # ── Progressive cross-day reward weights (ported from training_config.yaml
        # REWARD block). These make consistency the primary signal: passing days,
        # streaks of passing days, and days with DD well under the limit all add up.
        rw = cfg.get("REWARD", {}) or {}
        self.pass_day_bonus = float(rw.get("pass_day_bonus", cfg.get("PASS_DAY_BONUS", 2.0)))
        # NOTE: no ok_day_bonus anymore — classification is strictly binary (RULE 2).
        self.fail_day_penalty = float(rw.get("fail_day_penalty", cfg.get("FAIL_DAY_PENALTY", -2.0)))
        self.streak_scale = float(rw.get("streak_scale", cfg.get("STREAK_SCALE", 0.1)))
        self.low_dd_threshold = float(rw.get("low_dd_threshold", cfg.get("LOW_DD_THRESHOLD", 0.005)))
        self.low_dd_bonus = float(rw.get("low_dd_bonus", cfg.get("LOW_DD_BONUS", 0.3)))

        self._phi_history: List[float] = []
        self.global_ep = 0
        # 14-day deque of daily PASS(1)/FAIL(0) outcomes for the weekly bonus.
        self._daily_pass = deque(maxlen=14)
        self._pass_streak = 0   # consecutive passing days (for streak bonus)

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

    def daily_reward(self, r_d: float, dd_d: float) -> float:
        """
        Cross-day terminal reward applied at day end — STRICTLY BINARY now
        (ftmo_rules_fix.md RULE 2): a day is PASS or FAIL, there is NO "OK" tier.

          pass  (r_d >= target_pct AND dd_d <= max_dd) -> +pass_day_bonus
          fail  (everything else, incl. green-but-below-target and DD breach)
                                                        -> +fail_day_penalty
          streak bonus: +streak_scale * consecutive_pass_days (reset on any fail)
          low-DD bonus: +low_dd_bonus on a PASS day finishing well under the limit

        NOTE: this helper is percent-based (it only sees the day's RETURN r_d, not
        absolute equity), so it uses r_d >= target_pct as the binary pass test.
        The authoritative classification — the FIXED dollar increment off INITIAL
        equity — lives in BatchedFTMOEnv.step (which trains the agent). This helper
        mirrors the binary PASS/FAIL split for the Φ/diagnostics path.

        Returns the scalar daily reward and updates the internal pass streak.
        """
        is_pass = (r_d >= self.target_pct) and (dd_d <= self.max_dd_pct)

        if is_pass:
            base = self.pass_day_bonus
            self._pass_streak += 1
        else:                       # binary: anything not a pass is a fail
            base = self.fail_day_penalty
            self._pass_streak = 0

        reward = base + self.streak_scale * self._pass_streak
        # low-DD bonus only sweetens a PASS (a fail already takes the penalty).
        if is_pass and dd_d < self.low_dd_threshold:
            reward += self.low_dd_bonus
        return float(reward)

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
