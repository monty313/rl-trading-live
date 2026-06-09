"""
tests/unit/test_reward_redesign.py
────────────────────────────────────────────────────────────────────────────
Locks the reward-system REDESIGN (reward_redesign_plan.md, Sections 1-11). The
authoritative pure-scalar reference functions live in core/reward/shaper.py and
the env mirrors them vectorized; we test BOTH layers here so the contract holds.

Coverage map (one section per block):
  • S1  five-tier classify_day (FAIL/OK/PASS/EXCEED) + halt-balance mapping +
        a breached day can NEVER be EXCEED/SURVIVAL; day_reward composition
        (tier bonus, DD-efficiency multiplier, red-day penalty, SURVIVAL stack).
  • S2  StreakTracker: exponential curve anchors, mulligan (1 free fail),
        negative streak mirror, escalating consecutive-fail penalty, recovery
        bonus, momentum bias.
  • S3  intraday wipeout-on-fail + cross-day give-back penalties.
  • S4  composite episode bonus + improvement multiplier (amplify on improvement).
  • S5  multi-asset commission: forex per_lot_round_trip (EURUSD 0.5-lot = $2.50
        RT), metals/crypto pct_notional, indices/oils/ag zero, per-side scaling.
  • S6  session code mapping + the 7 new observations present, correctly valued,
        and state_dim/obs-schema v3.
  • S7  env speed bonus arms/keeps on a fast profitable close.
  • S8  lot curriculum narrow->wide window per strategy phase.
"""
import math

import numpy as np
import torch

from core.settings import CFG, auto_tune_batch
from core.reward.shaper import (
    FAIL, OK, PASS, EXCEED, classify_day, day_reward, dd_efficiency_multiplier,
    streak_extra, streak_reward, StreakTracker, STREAK_ANCHORS,
    intraday_wipeout_penalty, cross_day_giveback_penalty, EpisodeRewardShaper,
)
from core.env.environment import (
    BatchedFTMOEnv, OBS_SCHEMA_VERSION, N_POSITION_FEATS, N_FTMO_FEATS,
    N_SESSION_FEATS, resolve_commission, classify_symbol, session_code_for_minute,
)
from core.agent.action_space import FLAT, BUY, EXIT_HOLD
from core.agent.ppo import PPOAgent
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")
RW = CFG["REWARD"]
INC = 250.0          # $10k @ 2.5% daily increment used throughout


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FIVE-TIER CLASSIFICATION + DAY REWARD
# ════════════════════════════════════════════════════════════════════════════
def test_classify_five_tiers_by_progress():
    s = 10_000.0
    assert classify_day(s + 0.40 * INC, s, INC, False, True) == FAIL   # <0.5
    assert classify_day(s + 0.50 * INC, s, INC, False, True) == OK     # 0.5
    assert classify_day(s + 0.99 * INC, s, INC, False, True) == OK     # <1.0
    assert classify_day(s + 1.00 * INC, s, INC, False, True) == PASS   # exactly target
    assert classify_day(s + 2.00 * INC, s, INC, False, True) == EXCEED # >target


def test_breached_day_can_never_be_exceed():
    """RESOLVED DECISION 2: a DD-breached day with progress>1 downgrades to PASS,
    never EXCEED."""
    s = 10_000.0
    assert classify_day(s + 2.0 * INC, s, INC, True, True) == PASS
    assert classify_day(s + 2.0 * INC, s, INC, False, True) == EXCEED


def test_halt_balance_decides_tier_not_the_breach():
    """The TIER is computed from the halt/closing balance; a breach above target
    is still a PASS, a breach below target is a FAIL."""
    s = 10_000.0
    assert classify_day(s + 1.2 * INC, s, INC, True, True) == PASS     # >= target
    assert classify_day(s + 0.2 * INC, s, INC, True, True) == FAIL     # < 0.5 target


def test_day_reward_pass_includes_survival_and_dd_efficiency():
    s = 10_000.0
    # PASS exactly at target, NO DD used, traded, no breach -> pass_b * 1.0 (eff) +
    # survival_bonus.
    r = day_reward(PASS, s + INC, s, INC, dd_used_pct=0.0, max_dd_pct=0.01,
                   dd_breached=False, traded=True, rw=RW)
    assert abs(r - (RW["pass_day_bonus"] + RW["survival_bonus"])) < 1e-6


def test_day_reward_dd_efficiency_reduces_positive_reward():
    s = 10_000.0
    full = day_reward(PASS, s + INC, s, INC, 0.0, 0.01, False, False, RW)
    used = day_reward(PASS, s + INC, s, INC, 0.01, 0.01, False, False, RW)  # whole budget
    assert used < full
    # using the WHOLE budget multiplies the tier reward by (1 - weight).
    assert abs(used - RW["pass_day_bonus"] * (1.0 - RW["dd_efficiency_weight"])) < 1e-6


def test_day_reward_fail_has_red_day_penalty_and_no_survival():
    s = 10_000.0
    # A red FAIL day: tier FAIL penalty (severity-scaled) PLUS red-day loss term,
    # and survival is forbidden (breached). Must be strongly negative.
    r = day_reward(FAIL, s - INC, s, INC, dd_used_pct=0.02, max_dd_pct=0.01,
                   dd_breached=True, traded=True, rw=RW)
    assert r < RW["fail_day_penalty"]            # worse than the bare base penalty
    # no survival was added (breached)
    assert r < 0.0


def test_dd_efficiency_multiplier_bounds():
    w = 0.5
    assert dd_efficiency_multiplier(0.0, 0.01, w) == 1.0
    assert abs(dd_efficiency_multiplier(0.01, 0.01, w) - (1.0 - w)) < 1e-9
    assert dd_efficiency_multiplier(0.05, 0.01, w) == (1.0 - w)   # clamped at budget


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STREAK CURVE + STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════
def test_streak_curve_matches_anchors():
    a = RW["streak_curve_a"]
    b = RW["streak_curve_b"]
    # The fitted curve should be monotone increasing and roughly hit the anchors.
    prev = -1.0
    for s in (1, 3, 5, 10, 15):
        v = streak_extra(s, a, b)
        assert v >= prev
        prev = v
    # streak 1 anchor is 0.0; long streaks dwarf short ones (priority #1).
    assert abs(streak_extra(1, a, b)) < 1e-6
    assert streak_extra(15, a, b) > 5.0 * streak_extra(5, a, b)


def test_streak_reward_negative_mirror():
    a, b, base, nm = RW["streak_curve_a"], RW["streak_curve_b"], RW["streak_base"], 1.5
    pos5 = streak_reward(5, a, b, base, nm)
    neg5 = streak_reward(-5, a, b, base, nm)
    assert abs(neg5 - (-nm * pos5)) < 1e-6
    assert streak_reward(0, a, b, base, nm) == 0.0


def test_streak_tracker_mulligan_one_free_fail():
    t = StreakTracker(RW)
    for _ in range(3):
        t.update(True)
    assert t.pass_streak == 3
    t.update(False)                 # ONE fail — mulligan keeps the streak
    assert t.pass_streak == 3
    t.update(False)                 # SECOND consecutive fail breaks it
    assert t.pass_streak == 0


def test_streak_tracker_recovery_and_escalation():
    t = StreakTracker(RW)
    t.update(False)
    r2 = t.update(False)            # two fails: escalating penalty grows
    r1_again = StreakTracker(RW)
    first = r1_again.update(False)
    assert r2 < first               # second consecutive fail hurts more
    # a pass after fails grants the recovery bonus
    rec = t.update(True)
    assert rec > RW["recovery_bonus"] - 1.0   # includes flat recovery + streak base


def test_streak_tracker_momentum_after_pass():
    t = StreakTracker(RW)
    t.update(True)                  # sets last_was_pass
    # next update carries the momentum bias regardless of outcome
    r = t.update(True)
    assert r > streak_reward(2, t.a, t.b, t.base, t.neg_mult)  # has +momentum on top


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GIVE-BACK PENALTIES
# ════════════════════════════════════════════════════════════════════════════
def test_intraday_wipeout_erases_progress_and_punishes_giveback():
    # Ran to +$200 then closed -$30 on a FAIL: wipe the accrued +reward AND penalize
    # the give-back from the high.
    adj = intraday_wipeout_penalty(intraday_high_eq=10_200.0, end_equity=9_970.0,
                                   day_start_equity=10_000.0,
                                   accrued_progress_reward=0.8, rw=RW)
    assert adj < -0.8               # at least the wiped progress, plus give-back


def test_cross_day_giveback_zero_at_peak_negative_below():
    assert cross_day_giveback_penalty(10_500.0, 10_500.0, 10_000.0, RW) == 0.0
    below = cross_day_giveback_penalty(10_500.0, 10_300.0, 10_000.0, RW)
    assert below < 0.0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPOSITE EPISODE BONUS + IMPROVEMENT MULTIPLIER
# ════════════════════════════════════════════════════════════════════════════
def test_episode_bonus_grows_with_streak_and_amplifies_on_improvement():
    sh = EpisodeRewardShaper(dict(CFG))
    b_small = sh.episode_bonus(best_streak=2, pass_rate=0.3, dd_efficiency=0.5)
    b_big = EpisodeRewardShaper(dict(CFG)).episode_bonus(10, 0.3, 0.5)
    assert b_big > b_small          # longer streak -> bigger primary term
    # improvement multiplier: a second call with a HIGHER pass rate is amplified.
    sh2 = EpisodeRewardShaper(dict(CFG))
    first = sh2.episode_bonus(3, 0.30, 0.5)
    improved = sh2.episode_bonus(3, 0.50, 0.5)   # +20pp pass rate
    flat = EpisodeRewardShaper(dict(CFG))
    flat.episode_bonus(3, 0.50, 0.5)
    flat_again = flat.episode_bonus(3, 0.50, 0.5)  # no improvement
    assert improved > flat_again    # the improvement got amplified


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MULTI-ASSET COMMISSION
# ════════════════════════════════════════════════════════════════════════════
def test_eurusd_forex_round_trip_and_per_side():
    # The active path: $5/std lot round trip -> 0.5 lot = $2.50 RT, $1.25/side.
    assert resolve_commission("EURUSD", 0.5, 1.10, CFG, "round_trip") == 2.50
    assert resolve_commission("EURUSD", 0.5, 1.10, CFG, "open") == 1.25
    assert resolve_commission("EURUSD", 2.0, 1.10, CFG, "round_trip") == 10.0


def test_commission_classes_route_correctly():
    assert classify_symbol("EURUSD", CFG) == "forex"
    assert classify_symbol("XAUUSD", CFG) == "metals"
    assert classify_symbol("BTCUSD", CFG) == "crypto"
    assert classify_symbol("UKOIL.cash", CFG) == "oils"


def test_metals_and_crypto_are_pct_notional_and_indices_zero():
    metal = resolve_commission("XAUUSD", 1.0, 2000.0, CFG, "round_trip")
    assert metal > 0.0              # pct of notional, price-dependent
    crypto = resolve_commission("BTCUSD", 1.0, 50_000.0, CFG, "round_trip")
    assert crypto > 0.0
    assert resolve_commission("UKOIL.cash", 5.0, 80.0, CFG, "round_trip") == 0.0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SESSION CODE + NEW OBSERVATIONS + SCHEMA v3
# ════════════════════════════════════════════════════════════════════════════
def _cfg(bars_per_day=60, **extra):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"EPISODE_BARS": bars_per_day * 6, "BARS_PER_DAY": bars_per_day,
              "LOOKBACK": 20, "ACCOUNT_SIZE": 10_000.0, "INITIAL_EQUITY": 10_000.0})
    c.update(extra)
    return c


def _env(bars_per_day=60, phase=None, **extra):
    arr = make_synthetic_ohlcv_array(n=600)
    phase = phase or {"entry_conditions": {"buy": "any", "sell": "any"}}
    return BatchedFTMOEnv(arr, _cfg(bars_per_day, **extra), DEV,
                          instrument="EURUSD", phase=phase)


def test_session_code_maps_minutes_to_sessions():
    sessions = CFG["TRADING_SESSIONS"]
    # session_code_for_minute returns the RAW session code (the env normalizes it
    # by N_SESSIONS for the observation). 15:00 CEST is inside london_ny_overlap.
    code = session_code_for_minute(15 * 60, sessions)
    assert code in {row[3] for row in sessions} and code > 0.0
    # the dead-of-night minute (outside all sessions) -> 0.0 (market thin/closed)
    assert session_code_for_minute(0, sessions) == 0.0


def test_obs_schema_v3_and_state_dim():
    env = _env()
    assert env.obs_schema_version == OBS_SCHEMA_VERSION == 3
    assert env.state_dim == (env.lkbk * env.F + N_POSITION_FEATS
                             + N_FTMO_FEATS + N_SESSION_FEATS)
    s = env.reset()
    assert s.shape == (env.B, env.state_dim)


def test_seven_session_features_present_and_valued():
    env = _env()
    s = env.reset()
    sess = s[:, -N_SESSION_FEATS:]
    assert sess.shape == (env.B, N_SESSION_FEATS)
    assert torch.isfinite(sess).all()
    # commission feature (last col) is the per-1.0-lot RT commission / day_start_eq,
    # i.e. $5 / $10,000 = 0.0005 for the active forex EURUSD path.
    assert torch.allclose(sess[:, -1], torch.full((env.B,), 5.0 / 10_000.0), atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5/hot-path — env deducts commission on the trade
# ════════════════════════════════════════════════════════════════════════════
def test_env_commission_helpers_scale_and_charge_both_sides():
    """The env's vectorized forex commission mirrors resolve_commission: per-side is
    half the round-trip, scales linearly with lots, and is price-independent for the
    active forex class. (PnL swamps the equity delta on moving synthetic data, so we
    test the helper directly — it is what step() deducts at open AND close.)"""
    env = _env(bars_per_day=120, LOT_CURRICULUM_ENABLED=False)
    env.reset()
    price = torch.full((env.B,), 1.10)
    half = env._commission_for_lots(torch.full((env.B,), 0.5), price, "open")
    full_lot = env._commission_for_lots(torch.full((env.B,), 1.0), price, "open")
    rt = env._commission_per_lot_round_trip(price)
    # 0.5 lot open side = $1.25 (half of the $5 round trip * 0.5).
    assert torch.allclose(half, torch.full((env.B,), 1.25), atol=1e-6)
    # linear in lots: 1.0 lot is double 0.5 lot.
    assert torch.allclose(full_lot, 2.0 * half, atol=1e-6)
    # per-1.0-lot round trip == $5 for forex, regardless of price.
    assert torch.allclose(rt, torch.full((env.B,), 5.0), atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SPEED BONUS (green within N minutes, kept on a profitable close)
# ════════════════════════════════════════════════════════════════════════════
def test_speed_bonus_config_plumbs_into_env():
    env = _env()
    assert env._speed_bonus == float(CFG["REWARD"].get("speed_bonus", 0.0))
    assert env._speed_window == int(CFG.get("SPEED_BONUS_MINUTES", 3))


def test_speed_bonus_arms_on_fast_green_position():
    """Open long on a flat bar, then the next bar ticks UP within the speed window:
    the position shows green inside the window so the speed bonus ARMS."""
    n = 400
    close = np.full(n, 1.10, dtype=np.float64)
    close[201:] = 1.1020                                   # +20 pips just after entry
    ohlcv = np.stack([close, close + 1e-4, close - 1e-4, close,
                      np.ones(n) * 100], axis=1).astype(np.float32)
    cfg = _cfg(bars_per_day=1440, MAX_TRADES_PER_DAY=100000,
               LOT_CURRICULUM_ENABLED=False)
    cfg["EPISODE_BARS"] = 1440
    env = BatchedFTMOEnv(ohlcv, cfg, DEV, instrument="EURUSD",
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    env.reset()
    env._start.fill_(0)
    env._step_i.fill_(200)
    buy = {"direction": torch.full((env.B,), BUY, dtype=torch.long),
           "lot_raw": torch.full((env.B,), 0.5),
           "exit": torch.full((env.B,), EXIT_HOLD, dtype=torch.long)}
    env.step(buy)                                          # open at bar 200 (1.10)
    env.step({"direction": torch.full((env.B,), FLAT, dtype=torch.long),
              "lot_raw": torch.zeros(env.B),
              "exit": torch.full((env.B,), EXIT_HOLD, dtype=torch.long)})  # +20 pips
    assert bool(env._speed_armed.all()), "fast green position should arm the speed bonus"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LOT CURRICULUM WINDOW
# ════════════════════════════════════════════════════════════════════════════
def test_lot_curriculum_window_narrows_early_phase():
    # phase1_cci_align was widened to [0.01, 1.00] so PPO has headroom to hit
    # the $250 daily target. The intent of this test — 'early phases narrow vs
    # the full [0.01, 2.00] head' — still holds (hi is half the full ceiling).
    env = _env(phase={"name": "phase1_cci_align",
                      "entry_conditions": {"buy": "any", "sell": "any"}})
    assert (env._lot_lo, env._lot_hi) == (0.01, 1.00)
    # Still narrower than the full head: hi < env.max_lot.
    assert env._lot_hi < env.max_lot


def test_lot_curriculum_disabled_uses_full_head():
    env = _env(LOT_CURRICULUM_ENABLED=False)
    assert env._lot_lo == 0.01 and env._lot_hi == env.max_lot


def test_lot_curriculum_widens_later_phase():
    env = _env(phase={"name": "phase4",
                      "entry_conditions": {"buy": "any", "sell": "any"}})
    assert env._lot_hi > 0.50           # later phase widens the window


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ENTROPY ANNEALING (high exploration -> stable ent_coef)
# ════════════════════════════════════════════════════════════════════════════
def test_entropy_anneals_from_start_to_stable():
    cfg = dict(CFG)
    cfg.update({"ENTROPY_ANNEAL_ENABLED": True, "ENTROPY_START_COEF": 0.10,
                "ENTROPY_ANNEAL_EPISODES": 20})
    cfg["PPO"] = {**cfg.get("PPO", {}), "ent_coef": 0.02}
    agent = PPOAgent(64, cfg, DEV)
    assert abs(agent.anneal_entropy(0) - 0.10) < 1e-9       # high at episode 0
    assert abs(agent.anneal_entropy(10) - 0.06) < 1e-9      # halfway, linear
    assert abs(agent.anneal_entropy(20) - 0.02) < 1e-9      # reaches stable exactly
    assert abs(agent.anneal_entropy(999) - 0.02) < 1e-9     # holds (no overshoot)


def test_entropy_anneal_disabled_is_constant():
    cfg = dict(CFG)
    cfg.update({"ENTROPY_ANNEAL_ENABLED": False})
    cfg["PPO"] = {**cfg.get("PPO", {}), "ent_coef": 0.02}
    agent = PPOAgent(64, cfg, DEV)
    assert agent.anneal_entropy(0) == agent.ent_coef_stable
    assert agent.anneal_entropy(50) == agent.ent_coef_stable
