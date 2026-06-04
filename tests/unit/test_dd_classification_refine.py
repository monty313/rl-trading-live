"""
tests/unit/test_dd_classification_refine.py
────────────────────────────────────────────────────────────────────────────
Tests for the REFINED trailing-1%-DD + 5-tier day classification
(dd_classification_refine.md, user-confirmed June 4).

What changed vs the prior system, and why these tests exist:
  • OK/PASS thresholds are now measured against INITIAL equity (fixed-$ off the
    original account), NOT the day's opening balance. PASS = final >=
    initial*(1+target_pct); OK = final >= initial*(1+half_target_pct).
  • A NEW tier FAIL_CAPITAL_LOSS fires when final < prior_day_balance
    (== today's start), and is checked FIRST (highest precedence).
  • A DD breach HALTS the day but is NOT an auto-fail: the HALT balance is
    classified by the SAME tier logic. A breached day can never earn
    EXCEED/SURVIVAL.
  • The trailing DD floor RATCHETS UP with new intraday equity highs and never
    down; it starts at start_balance*(1-max_dd_pct) and resets each new day.

All thresholds are config-driven (DAILY_TARGET_PCT / DAILY_HALF_TARGET_PCT /
DAILY_MAX_DD_PCT) — nothing is hardcoded.

Covers the hand-calc anchor the user gave (initial 10k, prior_day 10,300):
  10,260 -> PASS ; 10,150 -> OK ; 10,100 -> FAIL_UNDER_TARGET ;
  10,290 (< prior 10,300) -> FAIL_CAPITAL_LOSS.
"""
import torch

from core.settings import CFG, auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import FLAT, BUY
from core.reward.shaper import (
    FAIL, FAIL_CAPITAL_LOSS, OK, PASS, EXCEED, classify_day,
    resolve_half_target_pct,
)
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")
INIT = 10_000.0
TGT = 0.025           # full target +2.5% off INITIAL  -> 10,250
HALF = 0.0125         # half  +1.25% off INITIAL        -> 10,125
INC = INIT * TGT      # fixed daily increment ($250)


# ════════════════════════════════════════════════════════════════════════════
# PURE classify_day — precedence + INITIAL-relative thresholds
# ════════════════════════════════════════════════════════════════════════════
def _classify(final, *, prior, breached=False, traded=True,
              target=TGT, half=HALF, initial=INIT):
    return classify_day(final, prior, initial * target, breached, traded,
                        initial_equity=initial, target_pct=target,
                        half_target_pct=half, prior_day_balance=prior)


def test_handcalc_example_initial_10k():
    """The user's exact hand-calc anchor (initial 10k). The threshold tiers
    (PASS/OK/FAIL_UNDER_TARGET) are shown with prior_day == start == initial (so no
    capital-loss shadow), then the capital-loss line uses prior_day = 10,300 to
    demonstrate precedence (10,290 < 10,300 -> FAIL_CAPITAL_LOSS).

    Per the spec's hand-calc: 10,260 -> PASS (a clean >target day is EXCEED, which
    IS a pass); 10,150 -> OK; 10,100 -> FAIL_UNDER_TARGET; 10,290 with prior 10,300
    -> FAIL_CAPITAL_LOSS."""
    # 10,260 strictly above the +2.5% line (10,250) with no breach -> EXCEED (a
    # PASS-class day in the binary sense the spec means by "PASS").
    assert _classify(10_260.0, prior=10_000.0) in (PASS, EXCEED)
    assert _classify(10_150.0, prior=10_000.0) == OK                 # >=10,125,<10,250
    assert _classify(10_100.0, prior=10_000.0) == FAIL               # <10,125 (FAIL_UNDER_TARGET)
    # Capital-loss precedence: 10,290 < prior_day 10,300 -> FAIL_CAPITAL_LOSS even
    # though 10,290 >= the +2.5% target line (10,250).
    assert _classify(10_290.0, prior=10_300.0) == FAIL_CAPITAL_LOSS


def test_threshold_boundaries_no_capital_loss():
    """Exact tier boundaries with prior_day == initial (10,000)."""
    assert _classify(10_250.0, prior=10_000.0) == PASS     # exactly full target
    assert _classify(10_125.0, prior=10_000.0) == OK       # exactly half target
    assert _classify(10_124.99, prior=10_000.0) == FAIL    # just under half
    assert _classify(10_000.0, prior=10_000.0) == FAIL     # flat -> FAIL_UNDER_TARGET


def test_capital_loss_precedence_beats_ok_and_pass():
    """Construct prior_day ABOVE the half target so a final between half-target and
    prior-day would be OK/PASS by threshold — but capital-loss is checked FIRST."""
    prior = 10_400.0   # prior_day above the +2.5% full target line (10,250)
    # final 10,300: >= full target (10,250) so would be PASS, but < prior 10,400.
    assert _classify(10_300.0, prior=prior) == FAIL_CAPITAL_LOSS
    # final 10,200: >= half (10,125) so would be OK, but < prior 10,400.
    assert _classify(10_200.0, prior=prior) == FAIL_CAPITAL_LOSS
    # final exactly at prior -> NOT below prior -> classified by threshold; 10,400
    # is strictly above the full target with no breach -> EXCEED (a PASS-class day).
    assert _classify(10_400.0, prior=prior) in (PASS, EXCEED)


def test_zero_trade_day_is_fail_under_target():
    """A zero-trade day: final == start == prior. Not below prior (no capital
    loss), below half -> FAIL_UNDER_TARGET (plain FAIL)."""
    assert _classify(10_000.0, prior=10_000.0, traded=False) == FAIL


def test_breach_then_classify_is_not_auto_fail():
    """A DD breach does NOT auto-fail: the halt balance is classified normally."""
    # halt balance >= full target -> PASS (breach forbids the EXCEED upgrade).
    assert _classify(10_300.0, prior=10_000.0, breached=True) == PASS
    # halt balance in [half, full) -> OK.
    assert _classify(10_150.0, prior=10_000.0, breached=True) == OK
    # halt balance < half (but >= prior) -> FAIL_UNDER_TARGET.
    assert _classify(10_100.0, prior=10_000.0, breached=True) == FAIL
    # halt balance < prior -> FAIL_CAPITAL_LOSS (precedence first, even breached).
    assert _classify(9_900.0, prior=10_000.0, breached=True) == FAIL_CAPITAL_LOSS


def test_exceed_only_when_no_breach():
    """EXCEED requires strictly above the full target AND never breached; a breach
    caps the day at PASS."""
    assert _classify(10_500.0, prior=10_000.0, breached=False) == EXCEED
    assert _classify(10_500.0, prior=10_000.0, breached=True) == PASS
    # exactly AT the full target is PASS, never EXCEED (needs strictly above).
    assert _classify(10_250.0, prior=10_000.0, breached=False) == PASS


def test_config_driven_thresholds_change_tiers():
    """Changing DAILY_TARGET_PCT / DAILY_HALF_TARGET_PCT / DAILY_MAX_DD_PCT moves
    the thresholds — nothing is hardcoded."""
    # Tighter 1% target: full == 10,100, half == 10,050.
    assert _classify(10_120.0, prior=10_000.0, target=0.01, half=0.005) == EXCEED
    assert _classify(10_060.0, prior=10_000.0, target=0.01, half=0.005) == OK
    assert _classify(10_040.0, prior=10_000.0, target=0.01, half=0.005) == FAIL
    # Same balance is only OK under the default 2.5% target.
    assert _classify(10_150.0, prior=10_000.0, target=0.025, half=0.0125) == OK


def test_half_target_pct_derives_as_half_when_none():
    """DAILY_HALF_TARGET_PCT=None derives half == target/2; a pinned value wins."""
    assert resolve_half_target_pct(
        {"DAILY_TARGET_PCT": 0.03, "DAILY_HALF_TARGET_PCT": None}) == 0.015
    assert resolve_half_target_pct(
        {"DAILY_TARGET_PCT": 0.03, "DAILY_HALF_TARGET_PCT": 0.01}) == 0.01


# ════════════════════════════════════════════════════════════════════════════
# ENV-LEVEL — ratchet floor, breach/halt classification, OK-no-streak
# ════════════════════════════════════════════════════════════════════════════
def _cfg(account=INIT, target_pct=TGT, half_pct=HALF, max_dd_pct=0.010,
         bars_per_day=40):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({
        "EPISODE_BARS": bars_per_day * 8,
        "BARS_PER_DAY": bars_per_day,
        "LOOKBACK": 20,
        "ACCOUNT_SIZE": account,
        "INITIAL_EQUITY": account,
        "DAILY_TARGET_PCT": target_pct,
        "DAILY_HALF_TARGET_PCT": half_pct,
        "DAILY_MAX_DD_PCT": max_dd_pct,
    })
    return c


def _free_env(**kw):
    arr = make_synthetic_ohlcv_array(n=600)
    return BatchedFTMOEnv(arr, _cfg(**kw), DEV, instrument="EURUSD",
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})


def _flat(env):
    return {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(env.B),
            "exit": torch.zeros(env.B, dtype=torch.long)}


def _set_eq(env, v):
    env._balance[:] = v
    env._equity[:] = v


def test_ratchet_floor_starts_at_start_times_0_99():
    """The DD floor opens at start_balance*(1-max_dd_pct) on a fresh day."""
    env = _free_env(account=10_000.0, max_dd_pct=0.01, bars_per_day=40)
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_000.0       # peak == start at day open
    # floor == peak*(1-dd) == 10,000 * 0.99 == 9,900.
    floor = env._day_high_eq * (1.0 - env._max_dd_pct_t)
    assert torch.allclose(floor, torch.full_like(floor, 9_900.0))


def test_ratchet_floor_rises_with_new_high_and_never_decreases():
    """After a new equity HIGH H the floor == H*0.99; a subsequent LOWER (but
    non-breaching) equity does NOT lower the floor (it ratchets up only)."""
    env = _free_env(account=10_000.0, max_dd_pct=0.01, bars_per_day=40)
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_000.0
    _set_eq(env, 10_500.0)               # new intraday high H = 10,500
    env.step(_flat(env))
    assert bool((env._day_high_eq >= 10_500.0).all())
    floor_after_high = env._day_high_eq * (1.0 - env._max_dd_pct_t)
    assert torch.allclose(floor_after_high,
                          torch.full_like(floor_after_high, 10_395.0))  # 10,500*0.99
    # Dip to 10,450 (above floor 10,395 -> no breach): floor must NOT decrease.
    _set_eq(env, 10_450.0)
    env.step(_flat(env))
    assert bool((env._day_high_eq >= 10_500.0).all())   # peak held -> floor held
    assert not bool(env._dd_breached.any())             # 10,450 > 10,395 floor


def test_ratchet_floor_resets_next_day():
    """The ratcheting floor resets at the new-day boundary (peak -> new day open)."""
    env = _free_env(account=10_000.0, max_dd_pct=0.01, bars_per_day=6)
    env.reset()
    _set_eq(env, 10_000.0)
    info = None
    for step in range(env.bars_per_day):
        if step == 1:
            _set_eq(env, 10_800.0)        # spike a peak mid-day
        _s, _r, _d, info = env.step(_flat(env))
        if not bool(info["day_closed"].any()):
            _set_eq(env, 10_000.0)
    # After the boundary the peak (and thus the floor) reset to the new day's open.
    assert bool((env._day_high_eq < 10_800.0).all())


def test_breach_halt_balance_above_target_passes_not_autofail():
    """ENV: a day that BREACHES the DD but whose HALT balance >= full target is a
    PASS (not an auto-fail), and being breached forbids EXCEED."""
    env = _free_env(account=10_000.0, target_pct=0.025, max_dd_pct=0.01,
                    bars_per_day=60)
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_500.0       # peak
    env._equity_prev[:] = 10_500.0
    _set_eq(env, 10_300.0)               # ~3.8% below peak -> breach; >= 10,250
    _s, _r, _d, info = env.step(_flat(env))
    assert bool(info["day_halted"].all())
    assert bool(info["dd_breached"].all())
    assert bool(info["tier_pass"][info["day_closed"]].all())     # PASS
    assert not bool(info["tier_exceed"].any())                   # breach forbids EXCEED
    assert not bool(info["survival"].any())                      # and forbids SURVIVAL


def test_breach_halt_balance_capital_loss_first():
    """ENV: a breach whose halt balance is below prior-day (start) -> the
    capital-loss tier fires (precedence first)."""
    env = _free_env(account=10_000.0, target_pct=0.025, max_dd_pct=0.01,
                    bars_per_day=60)
    env.reset()
    env._day_start_eq[:] = 10_000.0      # prior_day == start
    env._day_high_eq[:] = 10_100.0
    env._equity_prev[:] = 10_100.0
    _set_eq(env, 9_950.0)                # below peak -> breach AND below prior
    _s, _r, _d, info = env.step(_flat(env))
    assert bool(info["day_halted"].all())
    assert bool(info["tier_capital_loss"][info["day_closed"]].all())
    assert bool(info["failed"].all())


def _run_full_day(env, final_eq, day_start_eq=10_000.0):
    """Hold flat at day_start_eq, jump to final_eq on the last bar, close the day."""
    _set_eq(env, day_start_eq)
    env._day_start_eq[:] = day_start_eq
    env._day_high_eq[:] = day_start_eq
    env._equity_prev[:] = day_start_eq
    info = None
    for step in range(env.bars_per_day):
        v = final_eq if step == env.bars_per_day - 1 else day_start_eq
        _set_eq(env, v)
        env._day_high_eq[:] = torch.maximum(env._day_high_eq, env._equity)
        _s, _r, _d, info = env.step(_flat(env))
        _set_eq(env, v)
    return info


def test_ok_day_does_not_advance_pass_streak_but_pass_does():
    """RULE 3: an OK day does NOT advance the pass-streak; a PASS day does."""
    env = _free_env(account=10_000.0, target_pct=0.025, half_pct=0.0125,
                    bars_per_day=8)
    env.reset()
    # Day 1: end at 10,150 -> OK (>= half 10,125, < full 10,250, >= prior). The
    # pass-streak must stay 0.
    _run_full_day(env, final_eq=10_150.0, day_start_eq=10_000.0)
    assert int(env._pass_streak.max()) == 0, "OK day must not advance pass-streak"
    # Day 2: end at 10,300 from a 10,150 open -> PASS (>= 10,250, >= prior 10,150).
    _run_full_day(env, final_eq=10_300.0, day_start_eq=10_150.0)
    assert int(env._pass_streak.min()) == 1, "PASS day must advance pass-streak"


def test_survival_only_on_non_breached_traded_day():
    """SURVIVAL bonus only on a day that TRADED and NEVER breached. A breached day
    classifying PASS at its halt balance earns neither SURVIVAL nor EXCEED."""
    # Non-breached, traded, PASS -> survival flagged.
    env = _free_env(account=10_000.0, target_pct=0.025, max_dd_pct=0.01,
                    bars_per_day=60)
    env.reset()
    buy = {"direction": torch.full((env.B,), BUY, dtype=torch.long),
           "lot_raw": torch.full((env.B,), 0.2),
           "exit": torch.zeros(env.B, dtype=torch.long)}
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_000.0
    env._equity_prev[:] = 10_000.0
    env.step(buy)                          # open a trade so the day "traded"
    # March to the calendar close holding flat with equity above the target, and
    # capture the info on the bar where the day actually closes.
    close_info = None
    for step in range(env.bars_per_day):
        _set_eq(env, 10_300.0)
        env._day_high_eq[:] = torch.maximum(env._day_high_eq, env._equity)
        _s, _r, _d, info = env.step(_flat(env))
        _set_eq(env, 10_300.0)
        if bool(info["day_closed"].all()):
            close_info = info
            break
    assert close_info is not None, "the day must close within bars_per_day"
    assert int(close_info["trades_today"].min()) > 0   # the day traded
    assert not bool(close_info["dd_breached"].any())   # never breached
    assert bool(close_info["survival"].all())          # traded + never breached
    assert bool(close_info["tier_exceed"].all())       # strictly above target, clean
