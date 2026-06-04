"""
tests/unit/test_ftmo_rules_fix.py
────────────────────────────────────────────────────────────────────────────
Regression tests for the FTMO PASS/FAIL + zero-trade fixes (ftmo_rules_fix.md).

Covers, exactly as the spec enumerates:
  • RULE 1 — daily target = FIXED increment off INITIAL equity (the +$20.86→FAIL
              regression that printed 🟢 by mistake).
  • RULE 2 — strictly BINARY classification: only PASS or FAIL; a zero-trade day
              is a FAIL (never SKIP/OK).
  • RULE 3 — true trailing DD per bar, halt ends the day, balance-at-halt decides
              PASS/FAIL (a breach does NOT auto-fail), peak resets per day, the
              peak is per-episode.
  • RULE 4 — force-entry: a gate-ON bar on an episode yields >=1 trade on THAT
              episode; never flat-through-gate (except after a DD halt).
  • RULE 5 — config: changing target_pct / max_dd_pct changes the target / breach
              threshold (nothing hardcodes 0.025 / 0.01 / 250 / 100).
"""
import torch

from core.settings import CFG, auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import FLAT, BUY
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def _cfg(account=10_000.0, target_pct=0.025, max_dd_pct=0.010, bars_per_day=60):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({
        "EPISODE_BARS": bars_per_day * 6,
        "BARS_PER_DAY": bars_per_day,
        "LOOKBACK": 20,
        "ACCOUNT_SIZE": account,
        "INITIAL_EQUITY": account,
        "DAILY_TARGET_PCT": target_pct,
        "DAILY_MAX_DD_PCT": max_dd_pct,
    })
    return c


def _free_env(**kw):
    """A free (ungated) env so we can script equity by hand and drive day closes
    without the gate forcing entries."""
    arr = make_synthetic_ohlcv_array(n=600)
    return BatchedFTMOEnv(arr, _cfg(**kw), DEV, instrument="EURUSD",
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})


def _flat_action(env):
    return {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(env.B),
            "exit": torch.zeros(env.B, dtype=torch.long)}


def _classify(env, day_start_eq, final_eq):
    """Drive one full day to its calendar close with the agent flat, scripting
    each episode's equity to `final_eq` (and opening eq to `day_start_eq`), and
    return the env's per-episode (passed, failed) booleans at the day close.

    We bypass the trading PnL by directly setting the equity tensors each bar —
    this isolates the CLASSIFICATION rule under test from fill mechanics."""
    env.reset()
    # Marked equity is recomputed each bar as balance + open-position MTM; with the
    # agent flat (MTM==0) marked equity == balance, so we script the REALIZED balance
    # (the persistent source of truth). Pinning only _equity would be overwritten by
    # the balance+MTM recompute inside step().
    env._balance[:] = day_start_eq
    env._equity[:] = day_start_eq
    env._day_start_eq[:] = day_start_eq
    env._day_high_eq[:] = day_start_eq
    env._equity_prev[:] = day_start_eq
    bpd = env.bars_per_day
    last_info = None
    for step in range(bpd):
        # Hold equity flat until the final bar, then jump to final_eq so the
        # day's closing equity is exactly final_eq.
        target = final_eq if step == bpd - 1 else day_start_eq
        env._balance[:] = target
        env._equity[:] = target
        env._day_high_eq[:] = torch.maximum(env._day_high_eq, env._equity)
        _s, _r, _d, last_info = env.step(_flat_action(env))
        # step() recomputes equity from balance (flat => unchanged); re-pin both to
        # our script for the next iteration.
        env._balance[:] = target
        env._equity[:] = target
    return last_info


# ════════════════════════════════════════════════════════════════════════════
# RULE 1 — FIXED daily increment off INITIAL equity
# ════════════════════════════════════════════════════════════════════════════
def test_daily_increment_is_fixed_off_initial_equity():
    env = _free_env(account=10_000.0, target_pct=0.025)
    # $10,000 @ 2.5% -> a flat $250 EVERY day, computed once at open.
    assert env.daily_increment == 250.0


def test_day_opening_at_10300_passes_on_initial_relative_target():
    """UPDATED for dd_classification_refine.md: the PASS/OK thresholds are now
    measured against INITIAL equity (fixed-$ off the original account), NOT the
    day's opening balance. PASS iff final >= initial*(1+target_pct) == 10,250 on a
    $10k @ 2.5% account, regardless of where the day OPENED.

    (Old semantics — kept here only as the explanation of WHY this test changed —
    classified by `final >= day_start + $250`, so a day opening at 10,300 needed
    10,550 and an end of 10,500 was a FAIL. The user's refined rule moves the
    target to the fixed INITIAL-relative line, so the SAME end of 10,500 is now a
    PASS: 10,500 >= 10,250 and 10,500 >= prior_day 10,300 (no capital loss).)

    The `daily_target` reported in info is still the day-start fixed increment
    (10,300 + 250 = 10,550) — kept for the observation/diagnostics — but it no
    longer gates the tier decision."""
    env = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info = _classify(env, day_start_eq=10_300.0, final_eq=10_600.0)
    assert bool(info["passed"].all())          # 10,600 >= 10,250 -> PASS
    # info's daily_target is still the day-start fixed increment (diagnostic only).
    assert info["daily_target"][0].item() == 10_550.0

    env2 = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info2 = _classify(env2, day_start_eq=10_300.0, final_eq=10_500.0)
    # 10,500 >= initial*1.025 (10,250) AND >= prior_day (10,300) -> PASS now.
    assert bool(info2["passed"].all())
    assert bool(info2["tier_pass"][info2["day_closed"]].all())


def test_plus_20_dollar_day_fails_regression():
    """THE BUG: a +$20.86 day on a $10k account printed 🟢 PASS. It must FAIL —
    the target is +$250, and +$20.86 is nowhere near it."""
    env = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info = _classify(env, day_start_eq=10_000.0, final_eq=10_020.86)
    assert not bool(info["passed"].any()), "+$20.86 day must NOT pass"
    assert bool(info["failed"].all()), "+$20.86 day must be a FAIL"


def test_full_2_5_percent_day_passes():
    """A genuine +2.5% day ($250 on $10k) clears the fixed increment -> PASS."""
    env = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info = _classify(env, day_start_eq=10_000.0, final_eq=10_250.0)
    assert bool(info["passed"].all())


# ════════════════════════════════════════════════════════════════════════════
# RULE 2 — strictly BINARY: PASS or FAIL; zero-trade day is a FAIL
# ════════════════════════════════════════════════════════════════════════════
def test_classification_is_binary_passed_xor_failed():
    env = _free_env(bars_per_day=40)
    info = _classify(env, day_start_eq=10_000.0, final_eq=10_300.0)
    closed = info["day_closed"]
    # On every closed episode exactly one of passed / failed is true.
    assert bool((info["passed"][closed] ^ info["failed"][closed]).all())


def test_zero_trade_day_is_a_fail_not_skip():
    """A whole day with ZERO trades (agent flat, free phase so nothing forces an
    entry) ends under target and must classify FAIL — never SKIP/OK."""
    env = _free_env(bars_per_day=40)
    env.reset()
    info = None
    for _ in range(env.bars_per_day):
        _s, _r, _d, info = env.step(_flat_action(env))
    assert int(info["trades_today"].sum()) == 0       # truly zero trades
    assert bool(info["day_closed"].all())
    assert bool(info["failed"].all())                 # zero-trade -> FAIL
    assert not bool(info["passed"].any())
    # The binary info contract carries no OK/SKIP key.
    assert "no_trade_penalty" not in info and "daily_target" in info


# ════════════════════════════════════════════════════════════════════════════
# RULE 3 — trailing DD halts; balance-at-halt decides; peak per bar/episode
# ════════════════════════════════════════════════════════════════════════════
def test_dd_breach_halts_and_blocks_new_trades():
    """A 1% trailing DD breach halts trading: no new entries open for the rest of
    that day even though the agent keeps trying to BUY."""
    env = _free_env(account=10_000.0, max_dd_pct=0.01, bars_per_day=60)
    env.reset()
    # Drive a breach on episode 0 by collapsing its equity below peak*(1-1%).
    env._day_high_eq[:] = 10_000.0
    env._balance[:] = 9_800.0         # 2% below peak -> breach (realized balance)
    env._equity[:] = 9_800.0
    buy = {"direction": torch.full((env.B,), BUY, dtype=torch.long),
           "lot_raw": torch.full((env.B,), 0.5),
           "exit": torch.zeros(env.B, dtype=torch.long)}
    env.step(buy)
    assert bool(env._day_halted.all())            # breach halted the day
    trades_at_halt = env._trades_today.clone()
    # Keep trying to BUY — halted day must NOT open anything new.
    for _ in range(5):
        env.step(buy)
    assert bool((env._trades_today <= trades_at_halt).all())
    assert bool((env._position == 0).all())       # positions flattened at halt


def test_balance_at_halt_decides_pass_even_after_breach():
    """A DD breach does NOT auto-fail: if the balance at the halt moment is >=
    the daily target, the day is a PASS (RULE 3)."""
    env = _free_env(account=10_000.0, target_pct=0.025, max_dd_pct=0.01,
                    bars_per_day=60)
    env.reset()
    # Episode opened at 10,000 (target 10,250). Run equity up to a peak, then dip
    # >1% off that peak BUT still above the target, so the halt equity passes.
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_400.0     # peak this day
    env._balance[:] = 10_400.0
    env._equity[:] = 10_400.0
    env._equity_prev[:] = 10_400.0
    # 10,280 is 1.15% below the 10,400 peak (breach) but >= 10,250 target.
    env._balance[:] = 10_280.0
    env._equity[:] = 10_280.0
    _s, _r, _d, info = env.step(_flat_action(env))
    assert bool(info["day_halted"].all())          # breached + halted
    assert bool(info["passed"].all())              # but balance-at-halt >= target
    assert not bool(info["failed"].any())


def test_balance_at_halt_fails_when_under_target():
    env = _free_env(account=10_000.0, target_pct=0.025, max_dd_pct=0.01,
                    bars_per_day=60)
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_100.0
    env._balance[:] = 10_100.0
    env._equity[:] = 10_100.0
    env._equity_prev[:] = 10_100.0
    env._balance[:] = 9_980.0          # >1% below peak (breach), under target
    env._equity[:] = 9_980.0
    _s, _r, _d, info = env.step(_flat_action(env))
    assert bool(info["day_halted"].all())
    assert bool(info["failed"].all())
    assert not bool(info["passed"].any())


def test_trailing_peak_updates_per_bar_and_resets_per_day():
    """Peak rises with equity each bar, and resets to the day's opening equity at
    the calendar day boundary (RULE 3)."""
    env = _free_env(account=10_000.0, bars_per_day=10)
    env.reset()
    env._day_high_eq[:] = 10_000.0
    env._balance[:] = 10_500.0
    env._equity[:] = 10_500.0
    env.step(_flat_action(env))
    assert bool((env._day_high_eq >= 10_500.0).all())   # peak tracked the new high
    # March to the calendar day boundary; on the new day peak resets to day open.
    env.reset()
    env._balance[:] = 10_000.0
    env._equity[:] = 10_000.0
    info = None
    for step in range(env.bars_per_day):
        if step == 2:
            env._balance[:] = 10_700.0         # spike a peak mid-day
            env._equity[:] = 10_700.0
        _s, _r, _d, info = env.step(_flat_action(env))
        if not info["day_closed"].any():
            env._balance[:] = 10_000.0
            env._equity[:] = 10_000.0
    # After the boundary the live peak has been reset to the new day's opening eq,
    # not the prior day's 10,700 high.
    assert bool((env._day_high_eq < 10_700.0).all())


def test_dd_peak_is_per_episode():
    """The trailing peak is independent per episode: spiking one episode's equity
    must not raise another episode's peak."""
    env = _free_env(bars_per_day=60)
    env.reset()
    env._day_high_eq[:] = 10_000.0
    eq = env._equity.clone()
    eq[0] = 10_900.0                    # only episode 0 spikes
    env._balance[:] = eq
    env._equity[:] = eq
    env.step(_flat_action(env))
    assert env._day_high_eq[0].item() >= 10_900.0
    assert env._day_high_eq[1].item() < 10_900.0


# ════════════════════════════════════════════════════════════════════════════
# RULE 4 — force-entry: never flat-through a gate-ON bar
# ════════════════════════════════════════════════════════════════════════════
def _phase1_env():
    """A real gated phase (phase1_cci_align, CCI30/CCI100 SMA(1,+8), [1,15]) on
    synthetic data so the gate genuinely fires on a fraction of bars."""
    arr = make_synthetic_ohlcv_array(n=4000, seed=7)
    c = _cfg(bars_per_day=120)
    c["EPISODE_BARS"] = 600
    phase = {"name": "phase1_cci_align", "mask": "phase1_cci_align",
             "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15]}
    return BatchedFTMOEnv(arr, c, DEV, instrument="EURUSD", phase=phase)


def test_gate_on_bar_forces_a_trade_on_that_episode():
    """The crux of RULE 4: on any bar where the gate is ON for an episode and that
    episode is flat (and not halted), the env MUST open a trade that bar on THAT
    episode. We sample FLAT every bar to prove force-entry — not the agent — is
    what guarantees the entry. Asserts: no flat-through-gate violations, and the
    gate actually fired (so the assertion is non-vacuous)."""
    env = _phase1_env()
    env.reset()
    gate_on_bars = 0
    violations = 0
    for _ in range(400):
        abs_idx = env._abs_idx()
        gate_on = env._gate_on_batch(abs_idx)          # per-episode, pre-step
        not_halted = ~env._day_halted
        env.step(_flat_action(env))                    # agent stays FLAT
        # Any episode that was gate-ON and not halted must now hold a position.
        flat_through_gate = gate_on & not_halted & (env._position == 0)
        violations += int(flat_through_gate.sum())
        gate_on_bars += int(gate_on.sum())
    assert gate_on_bars > 0, "gate never fired — test would be vacuous"
    assert violations == 0, f"{violations} flat-through-gate bars (force-entry bug)"


def test_force_entry_suppressed_after_dd_halt():
    """RULE 4(b): after a DD halt, force-entry is correctly suppressed — a halted
    episode legitimately holds no position even on a gate-ON bar."""
    env = _phase1_env()
    env.reset()
    env._day_halted[:] = True            # simulate an already-halted day
    for _ in range(120):
        env.step(_flat_action(env))
        if not env._day_halted.any():    # rolled into a fresh (un-halted) day
            break
        assert bool((env._position == 0).all()), "halted day must not force-enter"


# ════════════════════════════════════════════════════════════════════════════
# RULE 5 — config inputs (never hardcoded)
# ════════════════════════════════════════════════════════════════════════════
def test_target_pct_changes_the_increment_and_target():
    """A different target_pct must change the fixed increment and the pass test —
    proving $250 / 2.5% is not hardcoded."""
    env5 = _free_env(account=10_000.0, target_pct=0.05, bars_per_day=40)
    assert env5.daily_increment == 500.0            # 5% of 10k, not the 2.5% default
    info = _classify(env5, day_start_eq=10_000.0, final_eq=10_300.0)
    # +$300 clears the OLD $250 bar but NOT the new $500 bar -> FAIL.
    assert not bool(info["passed"].any())
    assert info["daily_target"][0].item() == 10_500.0


def test_account_size_scales_increment():
    """A $25k account @ 2.5% -> $625 fixed increment (scales with account size)."""
    env = _free_env(account=25_000.0, target_pct=0.025)
    assert env.daily_increment == 625.0


def test_max_dd_pct_changes_breach_threshold():
    """A wider max_dd_pct must NOT breach where the default 1% would. Same equity
    dip, two thresholds, two outcomes -> the 1% is not hardcoded."""
    dip_to = 9_850.0                     # 1.5% below a 10,000 peak
    tight = _free_env(account=10_000.0, max_dd_pct=0.01, bars_per_day=60)
    tight.reset(); tight._day_high_eq[:] = 10_000.0
    tight._balance[:] = dip_to; tight._equity[:] = dip_to
    tight.step(_flat_action(tight))
    assert bool(tight._day_halted.all())             # 1.5% > 1% -> breach

    wide = _free_env(account=10_000.0, max_dd_pct=0.02, bars_per_day=60)
    wide.reset(); wide._day_high_eq[:] = 10_000.0
    wide._balance[:] = dip_to; wide._equity[:] = dip_to
    wide.step(_flat_action(wide))
    assert not bool(wide._day_halted.any())          # 1.5% < 2% -> no breach


def test_no_hardcoded_constants_in_rule_paths():
    """Grep the rule-bearing source files: the magic numbers 0.025 / 0.01 / 250 /
    100 (as a daily target/increment) must not appear as bare literals driving the
    classification — they come from CFG. We allow them only inside CFG defaults
    and comments. This guards against a regression re-hardcoding the rule."""
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env_src = (repo / "core" / "env" / "environment.py").read_text()
    # The classification line must reference daily_increment / target_pct, never a
    # literal 0.025 / 250 in the pass test.
    cls_region = env_src[env_src.index("DAY CLASSIFICATION"):]
    cls_region = cls_region[:cls_region.index("info = {")]
    # The increment is config-derived (target_aware_policy.md item 2 made it a
    # PER-EPISODE tensor `self._daily_increment_t` so randomized/inference target
    # changes flow through; the scalar `self.daily_increment` remains for the
    # guard/eval). Either reference is acceptable — what matters is no bare literal.
    assert ("self._daily_increment_t" in cls_region
            or "self.daily_increment" in cls_region)
    assert not re.search(r">=\s*.*0\.025", cls_region)
    assert not re.search(r"\b250\b", cls_region)
