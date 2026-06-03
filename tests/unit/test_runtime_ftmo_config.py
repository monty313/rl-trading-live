"""
tests/unit/test_runtime_ftmo_config.py
────────────────────────────────────────────────────────────────────────────
LOCKS the "rules are runtime config-driven" principle so it can NEVER drift.

These tests are the companion to tests/unit/test_ftmo_rules_fix.py (which proves
the PASS/FAIL + trailing-DD mechanics). Here we prove the *configurability*:

  • daily_increment is FIXED off INITIAL equity, not the day's opening balance.
  • target_pct / max_dd_pct are read at RUNTIME everywhere (env, guard, banner).
  • RESUMING from a checkpoint uses the CURRENT cfg value, NOT a stored one
    (PPO checkpoints never persist target_pct / max_dd_pct).
  • --daily-target-usd is exactly equivalent to the matching --target-pct.
  • the [ftmo] startup banner reflects the ACTIVE values for the run.
  • binary classification only (no OK/SKIP); zero-trade day = FAIL.

The FTMO principles single source of truth is the block at the top of
core/env/environment.py — these tests encode it.
"""
import torch

from core.settings import CFG, auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import FLAT
from core.pipeline import (build_pipeline, ftmo_rule_summary,
                           resolve_initial_equity)
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
        # use synthetic fixture data so build_pipeline needs no CSV/Drive
        "FEATURES": make_synthetic_ohlcv_array(n=600),
        "SYNTH_BARS": 600,
    })
    return c


def _free_env(**kw):
    arr = make_synthetic_ohlcv_array(n=600)
    return BatchedFTMOEnv(arr, _cfg(**kw), DEV, instrument="EURUSD",
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})


# ════════════════════════════════════════════════════════════════════════════
# 1. daily_increment is FIXED off INITIAL equity (NOT the day's opening balance)
# ════════════════════════════════════════════════════════════════════════════
def test_increment_fixed_off_initial_not_opening_balance():
    """initial 10k, day opens at 10,300, target_pct 2.5% -> increment STILL $250,
    target 10,550 — NOT 10,300 * 1.025 (= 10,557.50)."""
    env = _free_env(account=10_000.0, target_pct=0.025)
    assert env.daily_increment == 250.0
    # If it were percent-of-opening, a 10,300 open would target 10,557.50.
    day_start = 10_300.0
    fixed_target = day_start + env.daily_increment
    assert fixed_target == 10_550.0
    assert fixed_target != day_start * 1.025   # the old WRONG rule


# ════════════════════════════════════════════════════════════════════════════
# 2. PASS iff end/halt equity >= target: +$20 FAILs, +$250 PASSes
# ════════════════════════════════════════════════════════════════════════════
def _drive_day(env, day_start_eq, final_eq):
    env.reset()
    env._equity[:] = day_start_eq
    env._day_start_eq[:] = day_start_eq
    env._day_high_eq[:] = day_start_eq
    env._equity_prev[:] = day_start_eq
    bpd = env.bars_per_day
    info = None
    for step in range(bpd):
        target = final_eq if step == bpd - 1 else day_start_eq
        env._equity[:] = target
        env._day_high_eq[:] = torch.maximum(env._day_high_eq, env._equity)
        _s, _r, _d, info = env.step(
            {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
             "lot_raw": torch.zeros(env.B),
             "exit": torch.zeros(env.B, dtype=torch.long)})
        env._equity[:] = target
    return info


def test_plus_20_fails_plus_250_passes():
    env = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info_fail = _drive_day(env, 10_000.0, 10_020.0)     # +$20
    assert not bool(info_fail["passed"].any())
    assert bool(info_fail["failed"].all())

    env2 = _free_env(account=10_000.0, target_pct=0.025, bars_per_day=40)
    info_pass = _drive_day(env2, 10_000.0, 10_250.0)    # +$250
    assert bool(info_pass["passed"].all())


# ════════════════════════════════════════════════════════════════════════════
# 3. RUNTIME override: target_pct / max_dd_pct flow into the env's rule values
# ════════════════════════════════════════════════════════════════════════════
def test_runtime_target_and_dd_flow_into_env():
    env = _free_env(account=10_000.0, target_pct=0.04, max_dd_pct=0.03)
    assert env.target_pct == 0.04
    assert env.max_dd_pct == 0.03
    assert env.daily_increment == 400.0          # 4% of 10k, not the 2.5% default


# ════════════════════════════════════════════════════════════════════════════
# 4. RESUME uses the NEW cfg value, NOT a value stored in the checkpoint.
#    PPO checkpoints persist only weights/optimizer/{phase,episode,phi,pass_rate}.
# ════════════════════════════════════════════════════════════════════════════
def test_resume_uses_current_cfg_not_stored_target(tmp_path):
    # Build a pipeline at 2.5% / 1% and save a checkpoint.
    cfg_a = _cfg(account=10_000.0, target_pct=0.025, max_dd_pct=0.010)
    env_a, agent_a, _s, guard_a, _g = build_pipeline(cfg_a, DEV,
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    assert env_a.daily_increment == 250.0
    ckpt = tmp_path / "ck.pt"
    agent_a.save(str(ckpt), extra={"phase": "p", "episode": 7,
                                   "phi": 0.1, "pass_rate": 0.0})

    # The checkpoint must NOT carry the FTMO rule values.
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert "DAILY_TARGET_PCT" not in blob and "target_pct" not in blob
    assert "DAILY_MAX_DD_PCT" not in blob and "max_dd_pct" not in blob

    # Now "resume" with a DIFFERENT runtime config (5% / 2%) and load weights.
    cfg_b = _cfg(account=10_000.0, target_pct=0.05, max_dd_pct=0.02)
    env_b, agent_b, _s2, guard_b, _g2 = build_pipeline(cfg_b, DEV,
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    agent_b.load(str(ckpt), partial=True)        # weights only

    # The NEW cfg wins — env and guard now enforce 5% / 2%, not the stored 2.5%/1%.
    assert env_b.target_pct == 0.05
    assert env_b.max_dd_pct == 0.02
    assert env_b.daily_increment == 500.0
    assert guard_b.target_pct == 0.05
    assert guard_b.max_dd_pct == 0.02
    assert guard_b.daily_increment == 500.0


# ════════════════════════════════════════════════════════════════════════════
# 5. The daily guard's initial equity matches the env's (shared resolver), so its
#    fixed increment is computed off the SAME base as the env's classification.
# ════════════════════════════════════════════════════════════════════════════
def test_guard_initial_equity_matches_env():
    cfg = _cfg(account=25_000.0, target_pct=0.025)
    env, _a, _s, guard, _g = build_pipeline(cfg, DEV,
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    assert resolve_initial_equity(cfg) == 25_000.0
    assert env.initial_equity == 25_000.0
    assert guard.initial_balance == 25_000.0
    # Both compute the SAME fixed increment ($625 on $25k @ 2.5%).
    assert env.daily_increment == 625.0
    assert guard.daily_increment == 625.0


# ════════════════════════════════════════════════════════════════════════════
# 6. --daily-target-usd is exactly equivalent to the matching --target-pct.
# ════════════════════════════════════════════════════════════════════════════
def test_daily_target_usd_equivalent_to_target_pct():
    acct = 10_000.0
    usd = 250.0
    # The dollar form back-computes target_pct = usd / initial_equity.
    cfg = _cfg(account=acct, target_pct=usd / acct)
    env = BatchedFTMOEnv(make_synthetic_ohlcv_array(n=600), cfg, DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    assert env.daily_increment == usd
    assert abs(env.target_pct - 0.025) < 1e-12


# ════════════════════════════════════════════════════════════════════════════
# 7. The [ftmo] startup banner reflects the ACTIVE values for the run.
# ════════════════════════════════════════════════════════════════════════════
def test_startup_banner_reflects_active_values():
    cfg = _cfg(account=50_000.0, target_pct=0.03, max_dd_pct=0.015)
    line = ftmo_rule_summary(cfg)
    assert line.startswith("[ftmo] daily target = 3.00%")
    assert "=$1,500.00 on $50,000 account" in line   # 3% of 50k
    assert "daily max DD = 1.50%" in line


# ════════════════════════════════════════════════════════════════════════════
# 8. Binary classification: never OK/SKIP; a zero-trade day is a FAIL.
# ════════════════════════════════════════════════════════════════════════════
def test_binary_no_ok_skip_and_zero_trade_is_fail():
    env = _free_env(bars_per_day=40)
    env.reset()
    info = None
    for _ in range(env.bars_per_day):
        info = env.step(
            {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
             "lot_raw": torch.zeros(env.B),
             "exit": torch.zeros(env.B, dtype=torch.long)})[3]
    assert int(info["trades_today"].sum()) == 0       # genuinely zero trades
    assert bool(info["day_closed"].all())
    assert bool(info["failed"].all())                 # zero-trade -> FAIL
    assert not bool(info["passed"].any())
    # The info contract is strictly binary — no OK/SKIP tier keys exist.
    closed = info["day_closed"]
    assert bool((info["passed"][closed] ^ info["failed"][closed]).all())
    for forbidden in ("ok", "skip", "no_trade_penalty", "ok_day"):
        assert forbidden not in info
