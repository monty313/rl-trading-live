"""
core/reward/shaper.py
────────────────────────────────────────────────────────────────────────────
Reward shaping for the FTMO PPO agent. Two layers live here:

  1. EpisodeRewardShaper — the Φ potential-based EPISODE-end consistency bonus
     (ported from REPO1), PLUS the composite episode bonus + improvement
     multiplier (Section 4). Φ rewards CONSISTENCY over days/weeks rather than
     one-off spikes:  Φ = (pass_rate × avg_ret_norm) / (1 + λ × avg_dd_norm).

  2. The DAY-LEVEL reward toolkit (Sections 1-3): a set of PURE, TESTABLE
     functions that classify a day into one of FIVE TIERS and compute the
     tier reward, the exponential STREAK curve, the negative-streak / mulligan /
     recovery / momentum logic, the DD-efficiency multiplier, the red-day linear
     penalty, the intra-day give-back penalties, and the cross-day give-back
     penalty. BatchedFTMOEnv.step() calls a vectorized mirror of these; keeping
     the scalar reference here makes every rule independently unit-testable and
     gives a single human-readable source of truth for the reward maths.

THE FIVE TIERS (Section 1, RESOLVED DECISION 1 & 2) — classified purely by where
the ENDING (or DD-HALT) balance lands vs the daily target
(daily_target = day_start_equity + initial_equity*target_pct):

    FAIL     ending < 50% of the target's progress  -> full fail_day_penalty,
             scaled LINEARLY by how negative the day was (lose $80 == 8x $10).
    OK       50% <= ending < 100% of target          -> linear partial credit
             from ok_partial_lo..ok_partial_hi of pass_day_bonus.
    PASS     ending >= 100% of target (even after a DD halt, since the halt
             balance can still clear target) -> full pass_day_bonus.
    EXCEED   ending > 100% of target AND DD NEVER breached -> pass_day_bonus +
             a progressive bonus per % above target, with NO cap.
    SURVIVAL traded all day and NEVER breached the trailing DD -> a big bonus
             STACKED on top of whatever tier was earned.

A day that BREACHED the trailing DD can NEVER earn SURVIVAL or EXCEED (RESOLVED
DECISION 2): the halt balance still decides PASS/OK/FAIL, but survival requires
"never breached" and exceed requires "never breached".
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# ── Tier name constants (single source of truth; never hardcode the strings) ──
FAIL = "FAIL"
OK = "OK"
PASS = "PASS"
EXCEED = "EXCEED"
TIERS = (FAIL, OK, PASS, EXCEED)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — STREAK CURVE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The positive-streak EXTRA reward is a smooth, continuous, parametric curve
#       streak_extra(s) = a * (exp(b * (s - 1)) - 1)
# fit to the anchors Day1=+0 (base only), Day3=+0.3, Day5=+1.0, Day10=+5.0,
# Day15=+12.0. Day-1 contributes only the flat `streak_base`; the curve adds 0 at
# s=1 by construction (exp(0)-1 == 0) and grows toward the larger anchors.

# Cached fitted coefficients (used when scipy is unavailable, e.g. a minimal
# install). They reproduce the anchors closely — see the Day1/3/5/10/15 table in
# the final report. fit_streak_curve() re-fits with scipy when present.
_STREAK_A_DEFAULT = 0.616998
_STREAK_B_DEFAULT = 0.221749

# The anchor points the curve is fit to (day -> extra reward above the flat base).
STREAK_ANCHORS = {1: 0.0, 3: 0.3, 5: 1.0, 10: 5.0, 15: 12.0}


def fit_streak_curve(anchors: Dict[int, float] = None) -> Tuple[float, float]:
    """Fit streak_extra(s) = a*(exp(b*(s-1))-1) to the day->reward anchors and
    return (a, b). Uses scipy.optimize.curve_fit with relative-error weighting
    (so the small early anchors are honored as well as the large late ones) when
    scipy is importable; otherwise returns the cached defaults so the build never
    hard-depends on scipy. Deterministic — same anchors give the same (a, b)."""
    anchors = anchors or STREAK_ANCHORS
    try:
        from scipy.optimize import curve_fit
    except Exception:                                  # pragma: no cover
        return (_STREAK_A_DEFAULT, _STREAK_B_DEFAULT)
    days = np.array(sorted(anchors.keys()), dtype=float)
    vals = np.array([anchors[int(d)] for d in days], dtype=float)

    def _f(s, a, b):
        return a * (np.exp(b * (s - 1.0)) - 1.0)

    # Weight each residual ~ the anchor magnitude (floor so the zero anchor isn't
    # infinitely weighted) -> a good fit across the whole 0..12 range.
    sigma = np.maximum(np.abs(vals), 0.1)
    try:
        (a, b), _ = curve_fit(_f, days, vals, p0=[0.2, 0.3], sigma=sigma,
                              maxfev=200_000)
        return (float(a), float(b))
    except Exception:                                  # pragma: no cover
        return (_STREAK_A_DEFAULT, _STREAK_B_DEFAULT)


def streak_extra(streak: int, a: float, b: float) -> float:
    """Extra streak reward ABOVE the flat base for a positive streak length s>=1
    (0 for s<=0). Continuous exponential a*(exp(b*(s-1))-1)."""
    if streak <= 0:
        return 0.0
    return float(a * (np.exp(b * (float(streak) - 1.0)) - 1.0))


def streak_reward(streak: int, a: float, b: float, base: float,
                  negative_mult: float) -> float:
    """Full streak reward for a SIGNED streak (Section 2):
      • positive streak s>=1: base + streak_extra(s)   (the day passed).
      • negative streak s<=-1: MIRROR the positive curve at `negative_mult`x
        magnitude (S2.3): a length-5 negative streak == -(negative_mult * the
        +5 positive extra). The flat base is mirrored too so a single fail is
        -base * negative_mult.
      • s==0: 0 (no streak either way)."""
    if streak > 0:
        return float(base + streak_extra(streak, a, b))
    if streak < 0:
        mag = base + streak_extra(-streak, a, b)
        return float(-negative_mult * mag)
    return 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — FIVE-TIER DAY CLASSIFICATION + REWARD                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def classify_day(end_equity: float, day_start_equity: float,
                 daily_increment: float, dd_breached: bool,
                 traded: bool) -> str:
    """Return the day's TIER (FAIL/OK/PASS/EXCEED) from the ending/halt balance.

    progress = (end - day_start) / daily_increment   (fraction of target reached;
               1.0 == exactly hit the +increment target, can be negative on a loss).
      progress >= 1.0                  -> PASS, upgraded to EXCEED if progress>1.0
                                          AND the DD was never breached (RESOLVED
                                          DECISION 2: a breach forbids EXCEED).
      0.5 <= progress < 1.0            -> OK.
      progress < 0.5                   -> FAIL.

    `traded` and `dd_breached` do NOT change the FAIL/OK/PASS split (classification
    is purely by balance); they gate the SURVIVAL/EXCEED bonuses elsewhere. SURVIVAL
    (handled in day_reward) requires traded AND not breached."""
    inc = daily_increment if daily_increment != 0 else 1e-9
    progress = (end_equity - day_start_equity) / inc
    if progress >= 1.0:
        if progress > 1.0 and not dd_breached:
            return EXCEED
        return PASS
    if progress >= 0.5:
        return OK
    return FAIL


def dd_efficiency_multiplier(dd_used_pct: float, max_dd_pct: float,
                             weight: float) -> float:
    """Section 1.3 — multiplier in [1-weight, 1.0] applied to a POSITIVE day reward
    on OK/PASS/EXCEED days. Treats DD as a % budget: using NONE of it -> 1.0 (full
    reward); using the WHOLE budget -> (1 - weight). Linear in the fraction of the
    budget consumed. $20 of a $100 budget -> high; $90/$100 -> reduced."""
    budget = max_dd_pct if max_dd_pct > 0 else 1e-9
    used_frac = min(max(dd_used_pct / budget, 0.0), 1.0)
    return float(1.0 - weight * used_frac)


def day_reward(tier: str, end_equity: float, day_start_equity: float,
               daily_increment: float, dd_used_pct: float, max_dd_pct: float,
               dd_breached: bool, traded: bool, rw: dict) -> float:
    """Compute the TERMINAL day reward for one day from its tier + balances
    (Section 1), composing: the tier bonus/penalty, the EXCEED progressive bonus,
    the OK linear partial credit, the DD-efficiency multiplier (positive days),
    the linear RED-DAY penalty, and the stacked SURVIVAL bonus. Pure function over
    scalars so it is directly unit-testable; the env mirrors it vectorized.

    rw is the CFG["REWARD"] dict (weights). Returns a single float (normalized
    O(1) — callers feed percent-of-day-start balances or the raw equity; the math
    is ratio-based so either works as long as it is consistent)."""
    pass_b = float(rw.get("pass_day_bonus", 2.0))
    fail_b = float(rw.get("fail_day_penalty", -2.0))
    inc = daily_increment if daily_increment != 0 else 1e-9
    progress = (end_equity - day_start_equity) / inc

    reward = 0.0
    if tier == FAIL:
        # Full FAIL penalty, scaled LINEARLY by how negative the day was. A day at
        # 0..50% of target takes the base penalty; a day deep in the red takes a
        # multiple (the red-day term below adds the loss-proportional part). We use
        # max(1, -progress*..) so a flat 0%-progress fail still gets the base.
        severity = max(1.0, 1.0 - min(progress, 0.0))   # 0% -> 1x, -100% -> 2x, ...
        reward += fail_b * severity
    elif tier == OK:
        # Linear partial credit ok_partial_lo..ok_partial_hi of pass_b as progress
        # climbs 0.5 -> 1.0.
        lo = float(rw.get("ok_partial_lo", 0.25))
        hi = float(rw.get("ok_partial_hi", 0.95))
        frac = lo + (hi - lo) * ((progress - 0.5) / 0.5)
        reward += pass_b * frac
    elif tier in (PASS, EXCEED):
        reward += pass_b
        if tier == EXCEED:
            # Progressive bonus per +1.0 (==+100% of target) above target, NO cap.
            excess = progress - 1.0
            reward += float(rw.get("exceed_scale", 1.0)) * excess

    # ── Section 1.3 DD-efficiency multiplier (positive days only) ──
    if reward > 0.0 and tier in (OK, PASS, EXCEED):
        reward *= dd_efficiency_multiplier(
            dd_used_pct, max_dd_pct, float(rw.get("dd_efficiency_weight", 0.5)))

    # ── Section 1.2 RED-DAY linear penalty (ON TOP of the FAIL tier penalty) ──
    # Any negative-PnL day is punished in proportion to loss magnitude. Measured as
    # a fraction of the target (loss / daily_increment), so lose 1x the target == 1
    # unit of red_day_scale.
    if end_equity < day_start_equity:
        loss_frac = (day_start_equity - end_equity) / inc
        reward -= float(rw.get("red_day_scale", 1.0)) * loss_frac

    # ── Section 1.1 SURVIVAL bonus (stacked) ──
    # Traded all day AND never breached the trailing DD. A breached day can NEVER
    # earn it (RESOLVED DECISION 2).
    if traded and not dd_breached:
        reward += float(rw.get("survival_bonus", 1.5))
    return float(reward)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — STREAK STATE MACHINE (mulligan / recovery / momentum)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class StreakTracker:
    """Per-batch-item streak state machine implementing Section 2's mulligan,
    negative streaks, escalating consecutive-fail penalty, recovery, and momentum.
    The env keeps one logical tracker PER episode (vectorized), but the scalar
    reference here defines the exact rules and is unit-tested directly.

    State:
      pass_streak           consecutive passing days (>=0)
      fail_streak           consecutive failing days (>=0)
      consec_fail_count     consecutive fails since the last pass (mulligan logic)
      last_was_pass         was the immediately previous day a PASS (momentum)

    MULLIGAN (S2.2): one free fail per streak. TWO CONSECUTIVE fails break the
    pass streak; a SINGLE fail does NOT (the pass_streak is preserved while
    consec_fail_count <= mulligan_count). The mulligan recharges on the next pass.
    """

    def __init__(self, cfg_rw: dict):
        rw = cfg_rw or {}
        self.a = float(rw.get("streak_curve_a", _STREAK_A_DEFAULT))
        self.b = float(rw.get("streak_curve_b", _STREAK_B_DEFAULT))
        self.base = float(rw.get("streak_base", 0.5))
        self.mulligan = int(rw.get("mulligan_count", 1))
        self.neg_mult = float(rw.get("negative_streak_mult", 1.5))
        self.escalation = float(rw.get("consec_fail_escalation", 0.5))
        self.recovery = float(rw.get("recovery_bonus", 3.0))
        self.momentum = float(rw.get("momentum_bonus", 0.2))
        self.reset()

    def reset(self):
        self.pass_streak = 0
        self.fail_streak = 0
        self.consec_fail_count = 0
        self.last_was_pass = False

    def update(self, passed: bool) -> float:
        """Advance the state machine for one closed day and return the streak
        component of that day's reward (Section 2). Composes: the momentum bias
        (carried from the prior day), the positive/negative streak curve, the
        escalating consecutive-fail penalty, and the recovery bonus."""
        reward = 0.0
        # Momentum (S2.6): a small positive bias on the day AFTER a pass, applied
        # regardless of today's outcome (it biases the start of today).
        if self.last_was_pass:
            reward += self.momentum

        if passed:
            recovering = self.fail_streak > 0   # breaking a fail streak with a pass
            self.pass_streak += 1
            self.fail_streak = 0
            self.consec_fail_count = 0
            reward += streak_reward(self.pass_streak, self.a, self.b, self.base,
                                    self.neg_mult)
            if recovering:
                reward += self.recovery       # S2.5 flat recovery bonus
            self.last_was_pass = True
        else:
            self.consec_fail_count += 1
            self.fail_streak += 1
            # MULLIGAN (S2.2): the pass streak survives a SINGLE fail (consec<=
            # mulligan); a SECOND consecutive fail breaks it.
            if self.consec_fail_count > self.mulligan:
                self.pass_streak = 0
            reward += streak_reward(-self.fail_streak, self.a, self.b, self.base,
                                    self.neg_mult)
            # Escalating consecutive-fail penalty (S2.4), IN ADDITION to the
            # negative-streak penalty above: each consecutive fail hurts more.
            reward -= self.escalation * self.consec_fail_count
            self.last_was_pass = False
        return float(reward)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — INTRA-DAY + CROSS-DAY GIVE-BACK PENALTIES                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def intraday_wipeout_penalty(intraday_high_eq: float, end_equity: float,
                             day_start_equity: float, accrued_progress_reward: float,
                             rw: dict) -> float:
    """Section 3.2 — on a FAIL day, retroactively ERASE that day's accrued
    intra-day progress reward and ADD a penalty proportional to the give-back from
    the intra-day HIGH down to the close. Returns the (negative) adjustment to add
    to the day's reward. A day that ran to $200 above start then closed at -$30 is
    punished HARDER than one that never progressed.

      adjustment = -accrued_progress_reward                       (wipe the gains)
                   - giveback_from_high_scale * (high - end)/start (give-back pain)
    """
    if not bool(rw.get("intraday_wipeout", True)):
        return 0.0
    giveback = max(0.0, (intraday_high_eq - end_equity)) / (day_start_equity + 1e-9)
    return float(-accrued_progress_reward
                 - float(rw.get("giveback_from_high_scale", 1.0)) * giveback)


def cross_day_giveback_penalty(multi_day_peak_eq: float, end_equity: float,
                               initial_equity: float, rw: dict) -> float:
    """Section 3.3 — if equity has fallen from the MULTI-DAY peak, penalize the
    drop (protects multi-day gains). Returns a (negative) adjustment proportional
    to (peak - end)/initial. Zero while at/above the peak."""
    drop = max(0.0, (multi_day_peak_eq - end_equity)) / (initial_equity + 1e-9)
    return float(-float(rw.get("cross_day_giveback_scale", 0.5)) * drop)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EpisodeRewardShaper — Φ consistency bonus + composite episode bonus       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class EpisodeRewardShaper:
    def __init__(self, cfg: dict):
        self.target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
        self.max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
        self.alpha = float(cfg.get("SHAPE_ALPHA", 0.006))
        self.clip_val = float(cfg.get("SHAPE_CLIP", 0.006))
        self.lam = float(cfg.get("SHAPE_LAMBDA", 5.0))
        # Section 4.1: SHAPE_WARMUP REMOVED — Φ is active from EPISODE 1. The cfg
        # key defaults to 0 now; we still read it (so an explicit override works)
        # but the shipped config makes the bonus live immediately.
        self.warmup = int(cfg.get("SHAPE_WARMUP", 0))
        self.weekly_bonus = float(cfg.get("WEEKLY_BONUS", 0.02))
        self.window = 20

        rw = cfg.get("REWARD", {}) or {}
        self.pass_day_bonus = float(rw.get("pass_day_bonus", cfg.get("PASS_DAY_BONUS", 2.0)))
        self.fail_day_penalty = float(rw.get("fail_day_penalty", cfg.get("FAIL_DAY_PENALTY", -2.0)))
        self.streak_scale = float(rw.get("streak_scale", cfg.get("STREAK_SCALE", 0.1)))
        self.low_dd_threshold = float(rw.get("low_dd_threshold", cfg.get("LOW_DD_THRESHOLD", 0.005)))
        self.low_dd_bonus = float(rw.get("low_dd_bonus", cfg.get("LOW_DD_BONUS", 0.3)))
        self._rw = rw

        # Fit (or load cached) the streak curve once per shaper (Section 2).
        self.streak_a = float(rw.get("streak_curve_a", _STREAK_A_DEFAULT))
        self.streak_b = float(rw.get("streak_curve_b", _STREAK_B_DEFAULT))

        self._phi_history: List[float] = []
        self.global_ep = 0
        self._daily_pass = deque(maxlen=14)
        self._pass_streak = 0   # consecutive passing days (legacy binary streak)
        # Section 4.3 — previous-episode pass rate for the improvement multiplier.
        self._prev_pass_rate: float = None

    # ── Φ potential ─────────────────────────────────────────────────────────
    def _phi(self, pass_rate: float, avg_ret: float, avg_dd: float) -> float:
        ret_norm = avg_ret / (self.target_pct + 1e-8)
        dd_norm = avg_dd / (self.max_dd_pct + 1e-8)
        return (pass_rate * max(ret_norm, 0.0)) / (1.0 + self.lam * dd_norm)

    def compute_bonus(self, daily_log: list) -> float:
        """daily_log: per-day dicts {pass: bool, ret: float, dd: float}. Returns a
        scalar episode-end bonus. Section 4.1 removed the warmup gate, so this is
        live from episode 1 (warmup defaults to 0)."""
        if not daily_log:
            return 0.0
        for d in daily_log:
            self._daily_pass.append(1 if d.get("pass") else 0)

        pass_rate = float(np.mean([1.0 if d.get("pass") else 0.0 for d in daily_log]))
        avg_ret = float(np.mean([d.get("ret", 0.0) for d in daily_log]))
        avg_dd = float(np.mean([d.get("dd", 0.0) for d in daily_log]))
        phi = self._phi(pass_rate, avg_ret, avg_dd)

        self._phi_history.append(phi)
        if len(self._phi_history) > self.window:
            self._phi_history = self._phi_history[-self.window:]

        if self.global_ep < self.warmup:        # warmup defaults to 0 (S4.1)
            return 0.0

        phi_smooth = float(np.mean(self._phi_history[:-1])) if len(self._phi_history) > 1 else phi
        sigma = float(np.std(self._phi_history)) or 1.0
        bonus = self.alpha * (phi - phi_smooth) / sigma
        bonus = float(np.clip(bonus, -self.clip_val, self.clip_val))
        return bonus + self.weekly_consistency_bonus()

    # ── Section 4.2 / 4.3 — composite episode bonus + improvement multiplier ──
    def episode_bonus(self, best_streak: int, pass_rate: float,
                      dd_efficiency: float) -> float:
        """Composite EPISODE-level bonus (Section 4.2):
          primary   = the best STREAK length achieved this episode (priority #1),
                      run through the same exponential streak curve.
          secondary = pass rate + DD efficiency (priorities #2, #3).
        Then Section 4.3 IMPROVEMENT MULTIPLIER: if this episode's pass rate beats
        the previous episode's, AMPLIFY the whole bonus (prevents early
        discouragement — a 30%->50% jump gets a meaningful boost). The previous
        pass rate is tracked across calls.

        best_streak   : longest consecutive-pass run in the episode.
        pass_rate     : fraction of days passed this episode (0..1).
        dd_efficiency : mean DD-efficiency multiplier over the episode's good days
                        (0..1; 1.0 == used ~no DD budget).
        """
        primary = streak_extra(int(best_streak), self.streak_a, self.streak_b)
        secondary = float(pass_rate) + float(dd_efficiency)
        bonus = primary + secondary

        # Improvement multiplier (S4.3): amplify when pass rate improved.
        mult = 1.0
        if self._prev_pass_rate is not None and pass_rate > self._prev_pass_rate:
            # Up to +100% amplification proportional to the improvement, so a big
            # jump (e.g. +20pp) is rewarded more than a tiny one.
            mult = 1.0 + min(1.0, (pass_rate - self._prev_pass_rate) * 5.0)
        self._prev_pass_rate = float(pass_rate)
        return float(bonus * mult)

    def daily_reward(self, r_d: float, dd_d: float) -> float:
        """Legacy percent-based daily reward kept for the Φ/diagnostics path and
        backward compatibility. Binary pass test r_d >= target_pct. The
        authoritative 5-tier reward lives in classify_day/day_reward + the env."""
        is_pass = (r_d >= self.target_pct) and (dd_d <= self.max_dd_pct)
        if is_pass:
            base = self.pass_day_bonus
            self._pass_streak += 1
        else:
            base = self.fail_day_penalty
            self._pass_streak = 0
        reward = base + self.streak_scale * self._pass_streak
        if is_pass and dd_d < self.low_dd_threshold:
            reward += self.low_dd_bonus
        return float(reward)

    def weekly_consistency_bonus(self) -> float:
        """+WEEKLY_BONUS when the most recent 7-day pass rate exceeds the prior
        7-day pass rate. Needs a full 14-day window; returns 0.0 otherwise."""
        if len(self._daily_pass) < 14:
            return 0.0
        days = list(self._daily_pass)
        prev_week = sum(days[:7]) / 7.0
        this_week = sum(days[7:]) / 7.0
        return self.weekly_bonus if this_week > prev_week else 0.0
