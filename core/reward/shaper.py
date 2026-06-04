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

Thresholds are measured AGAINST INITIAL equity (fixed-$ increments: +$250 full /
+$125 half on a $10k account, per dd_classification_refine.md), with a NEW
capital-loss guard checked FIRST. Precedence (first match wins):

    FAIL_CAPITAL_LOSS  final < prior_day_balance (yesterday's close == today's
             start) -> FAIL. Gave back capital vs yesterday. CHECKED FIRST.
    PASS     final >= initial*(1+target_pct)  (>= +2.5% of INITIAL) -> full
             pass_day_bonus.
    OK       final >= initial*(1+half_target_pct) (>= +1.25%, i.e. >=50% of the
             target) but not yet PASS -> linear partial credit.
    FAIL     below half target but NOT below prior-day close -> full
             fail_day_penalty, scaled LINEARLY by how negative the day was.
    EXCEED   PASS AND final strictly above the full target AND DD NEVER breached
             -> pass_day_bonus + a progressive bonus per % above target, NO cap.
    SURVIVAL traded all day and NEVER breached the trailing DD -> a big bonus
             STACKED on top of whatever tier was earned.

A DD BREACH IS NOT AN AUTO-FAIL (dd_classification_refine.md): the HALT balance is
classified by the SAME logic above (halt balance >= full target still PASSES). But
a day that BREACHED can NEVER earn SURVIVAL or EXCEED (RESOLVED DECISION 2). A
zero-trade day ends flat (final == start == prior), so it is not below prior and is
below half -> FAIL. OK does NOT advance the pass-streak; PASS/EXCEED do; all FAIL_*
break it per the mulligan rules.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# ── Tier name constants (single source of truth; never hardcode the strings) ──
FAIL = "FAIL"                       # under target AND not below prior-day close
FAIL_CAPITAL_LOSS = "FAIL_CAPITAL_LOSS"   # gave back capital vs yesterday's close
OK = "OK"
PASS = "PASS"
EXCEED = "EXCEED"
# All FAIL_* variants are non-passing; OK is non-passing too (see classify_day).
TIERS = (FAIL, FAIL_CAPITAL_LOSS, OK, PASS, EXCEED)
FAIL_TIERS = (FAIL, FAIL_CAPITAL_LOSS)


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
def resolve_half_target_pct(cfg: dict) -> float:
    """Resolve the OK-tier half-target fraction from CFG (dd_classification_refine).

    DAILY_HALF_TARGET_PCT pins it explicitly; None DERIVES it as half of
    DAILY_TARGET_PCT. NOTHING downstream hardcodes 0.0125 — change the target and
    OK tracks at exactly half unless half is pinned."""
    half = cfg.get("DAILY_HALF_TARGET_PCT", None)
    if half is None:
        return float(cfg.get("DAILY_TARGET_PCT", 0.025)) / 2.0
    return float(half)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPUTER-SIMPLE RULES — SINGLE SOURCE OF TRUTH (dd_classification_refine) ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  TRAILING 1% DAILY DD (ratcheting floor) — see BatchedFTMOEnv.step:        ║
# ║    new day:  start_balance = day-open balance                             ║
# ║              max_equity_today = start_balance                             ║
# ║              daily_dd_floor   = start_balance * (1 - max_dd_pct)  [*0.99]  ║
# ║    each bar: equity = balance + open-position MTM  (commissions already    ║
# ║              in balance, per the 569aeca equity fix)                       ║
# ║              if equity > max_equity_today:                                 ║
# ║                  max_equity_today = equity                                 ║
# ║                  daily_dd_floor   = max_equity_today * (1 - max_dd_pct)    ║
# ║              # floor RATCHETS UP on new highs, NEVER down within a day      ║
# ║              breach when equity < daily_dd_floor  ->  HALT the day         ║
# ║              (flatten realizing MTM, suppress force-entry until next day)  ║
# ║    A BREACH IS NOT AN AUTOMATIC FAIL: after the halt, classify the HALT     ║
# ║    balance with the SAME tier logic below.                                 ║
# ║                                                                            ║
# ║  CLASSIFICATION (5-tier; applied to END-OF-DAY balance, or HALT balance if  ║
# ║  breached — identical calc). Let initial = account INITIAL equity,         ║
# ║  prior_day = yesterday's close (== today's start_balance), final = end/halt ║
# ║  balance. Thresholds are off INITIAL (fixed $ on a 10k acct: $250 / $125),  ║
# ║  NOT the day's opening balance. Precedence (FIRST match wins):             ║
# ║    1. final < prior_day                     -> FAIL_CAPITAL_LOSS  [NEW]    ║
# ║    2. elif final >= initial*(1+target_pct)  -> PASS  (>= +2.5%)            ║
# ║    3. elif final >= initial*(1+half_pct)    -> OK    (>= +1.25%, >=50% tgt)║
# ║    4. else                                  -> FAIL  (< half, not < prior) ║
# ║  THEN (no-breach gates, kept from the 5-tier system):                      ║
# ║    • EXCEED: PASS AND final > initial*(1+target_pct) AND never breached.   ║
# ║    • SURVIVAL bonus: traded AND never breached (a breached day, even one    ║
# ║      whose halt balance is PASS/OK, can earn NEITHER EXCEED NOR SURVIVAL). ║
# ║  Zero-trade day: final == start == prior -> not < prior, < half -> FAIL.   ║
# ║  Streak: PASS/EXCEED advance the pass-streak; OK does NOT; all FAIL_* break ║
# ║  per the mulligan rules. Reward ordering holds: PASS/EXCEED > OK > FAIL,    ║
# ║  and a breach/capital-loss day never out-rewards an OK day.                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def classify_day(end_equity: float, day_start_equity: float,
                 daily_increment: float, dd_breached: bool,
                 traded: bool, *, initial_equity: float | None = None,
                 target_pct: float | None = None,
                 half_target_pct: float | None = None,
                 prior_day_balance: float | None = None) -> str:
    """Return the day's TIER from the ending/halt balance (dd_classification_refine).

    Precedence (FIRST match wins; identical whether or not a DD breach occurred —
    a breach merely makes `end_equity` the HALT balance, it is NOT an auto-fail):
      1. end < prior_day_balance              -> FAIL_CAPITAL_LOSS  (gave back
                                                 capital vs yesterday's close).
      2. end >= initial*(1 + target_pct)      -> PASS   (>= +2.5% of INITIAL).
                                                 upgraded to EXCEED iff strictly
                                                 above AND never breached.
      3. end >= initial*(1 + half_target_pct) -> OK     (>= +1.25% of INITIAL,
                                                 i.e. >= 50% of the target).
      4. else                                 -> FAIL   (< half, not < prior).

    Thresholds are measured against INITIAL equity (fixed-$ increments), NOT the
    day's opening balance — consistent with daily_increment = initial*target_pct.

    Back-compat: when initial_equity/target_pct are not supplied we DERIVE them
    from day_start_equity + daily_increment (so legacy callers that only pass the
    day-start frame still classify by the +increment target off day-start, with
    prior_day defaulting to day_start so a flat/zero-trade day is FAIL not
    capital-loss). `traded`/`dd_breached` never change the FAIL/OK/PASS split;
    they gate SURVIVAL/EXCEED (handled here for EXCEED, in day_reward for SURVIVAL)."""
    inc = daily_increment if daily_increment != 0 else 1e-9
    # Resolve the absolute INITIAL-relative thresholds. Legacy callers (no
    # initial_equity) fall back to the day-start frame: target == day_start+inc,
    # half == day_start + inc/2 — preserving their original semantics exactly.
    if initial_equity is not None and target_pct is not None:
        full_target_eq = initial_equity * (1.0 + target_pct)
        half_pct = (half_target_pct if half_target_pct is not None
                    else target_pct / 2.0)
        half_target_eq = initial_equity * (1.0 + half_pct)
    else:
        full_target_eq = day_start_equity + inc
        half_target_eq = day_start_equity + 0.5 * inc
    prior = prior_day_balance if prior_day_balance is not None else day_start_equity

    # 1. CAPITAL LOSS vs yesterday's close — checked FIRST (highest precedence).
    if end_equity < prior:
        return FAIL_CAPITAL_LOSS
    # 2. PASS (>= full target off INITIAL); EXCEED only if strictly above & clean.
    if end_equity >= full_target_eq:
        if end_equity > full_target_eq and not dd_breached:
            return EXCEED
        return PASS
    # 3. OK (>= half target off INITIAL == >= 50% of the target).
    if end_equity >= half_target_eq:
        return OK
    # 4. FAIL (below half, but not below prior-day close).
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
    if tier in FAIL_TIERS:
        # Full FAIL penalty, scaled LINEARLY by how negative the day was. A day at
        # 0..50% of target takes the base penalty; a day deep in the red takes a
        # multiple (the red-day term below adds the loss-proportional part). We use
        # max(1, -progress*..) so a flat 0%-progress fail still gets the base.
        # FAIL_CAPITAL_LOSS shares this path (it is a FAIL by precedence); because
        # final < prior_day the red-day term below ALSO fires, so a capital-loss day
        # is penalized at least as hard as a same-magnitude FAIL — it can never
        # out-reward an OK day (reward ordering: PASS/EXCEED > OK > FAIL/_LOSS).
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
