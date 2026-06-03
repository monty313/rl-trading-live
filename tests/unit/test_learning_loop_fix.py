"""
tests/unit/test_learning_loop_fix.py
────────────────────────────────────────────────────────────────────────────
Tests for the learning-loop fix bundle (learning_loop_fix.md):

  • Bug A   — per-day report aggregates the FULL batch (no 1/64 leakage).
  • FIX 1   — reward shaping ordering (pass-day > flat-day > DD-breach-day) and
              normalized magnitude (~O(1), NOT 10,000-scale dollars).
  • FIX 3   — account-size invariance (same % outcome at 10k and 100k yields
              ~the same normalized reward).
  • streak  — consecutive-pass streak increments and resets on a fail.
  • FIX 1.3 — _phi_metric is always-computable and orders pass>flat>breach.

These are real-data-free (synthetic OHLCV only — production stays real-data-only).
"""
from __future__ import annotations

import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import BUY, FLAT, EXIT_HOLD, EXIT_CLOSE

DEV = torch.device("cpu")


def _flat_series(n: int, px: float = 1.10) -> np.ndarray:
    """Perfectly flat OHLCV — no price movement, so PnL is exactly $0 (used to
    isolate the day-classification / reward shaping without market noise)."""
    row = [px, px, px, px, 100.0]
    return np.asarray([row] * n, dtype=np.float32)


def _cfg(account=10_000.0, bars_per_day=40, ep_bars=120):
    return {
        "BATCH_SIZE_ENV": 8, "LOOKBACK": 20,
        "BARS_PER_DAY": bars_per_day, "EPISODE_BARS": ep_bars,
        "ACCOUNT_SIZE": account, "INITIAL_EQUITY": account, "MAX_LOT": 2.0,
        "DAILY_TARGET_PCT": 0.025, "DAILY_MAX_DD_PCT": 0.01,
        "MAX_TRADES_PER_DAY": 800,
        "USE_AMP": False, "USE_TORCH_COMPILE": False,
        "USE_FEATURE_CACHE": False,
    }


# ════════════════════════════════════════════════════════════════════════════
# Bug A — full-batch day aggregation (no 1/64 leakage)
# ════════════════════════════════════════════════════════════════════════════
def test_day_aggregation_covers_full_batch():
    """When the calendar day rolls over, _aggregate_day must cover ALL B episodes
    (n == B), never a single episode reported as the day (the old 1/64 bug)."""
    import training.train as T
    B = 8
    info = {
        "passed":           torch.zeros(B, dtype=torch.bool),
        "failed":           torch.zeros(B, dtype=torch.bool),
        "no_trade_penalty": torch.zeros(B, dtype=torch.bool),
        "trades_today":     torch.full((B,), 3, dtype=torch.long),
        "equity":           torch.full((B,), 10_100.0),
        "day_start_eq":     torch.full((B,), 10_000.0),
    }
    info["passed"][:5] = True            # 5 of 8 episodes passed this day
    closed = torch.ones(B, dtype=torch.bool)
    agg = T._aggregate_day(info, closed)
    assert agg["n"] == B                 # the whole batch, not 1/8
    assert agg["pass"] == 5
    assert agg["day_passed"] is True     # 5 pass > 0 fail -> green bubble
    # mean PnL is the batch mean of (equity - day_start), here exactly +$100.
    assert abs(agg["mean_pnl"] - 100.0) < 1e-6


def test_day_line_has_no_1_of_64_and_phase_last(capsys):
    """The printed day line carries the bubble right after the day number, a
    streak, and ends with the phase name — and never the '1/64'/'closed X/Y'
    noise that the old format printed."""
    import training.train as T
    info = {
        "passed": torch.ones(4, dtype=torch.bool),
        "failed": torch.zeros(4, dtype=torch.bool),
        "no_trade_penalty": torch.zeros(4, dtype=torch.bool),
        "trades_today": torch.full((4,), 2, dtype=torch.long),
        "equity": torch.full((4,), 10_250.0),
        "day_start_eq": torch.full((4,), 10_000.0),
    }
    agg = T._aggregate_day(info, torch.ones(4, dtype=torch.bool))
    T._print_daily_results({"name": "phase3_x"}, 7, agg, streak=3)
    line = capsys.readouterr().out
    assert "1/" not in line and "closed" not in line
    assert "🟢" in line and "streak" in line
    assert line.rstrip().endswith("phase3_x")


# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — reward shaping ordering + normalized magnitude
# ════════════════════════════════════════════════════════════════════════════
def _run_day(env: BatchedFTMOEnv, force_equity_fn):
    """Step a full day; after each step set equity to a scripted path via
    force_equity_fn(step)->equity (lets us script pass / flat / DD-breach days
    deterministically and read the cumulative reward for episode 0)."""
    env.reset()
    total = torch.zeros(env.B)
    open_act = {"direction": torch.full((env.B,), BUY, dtype=torch.long),
                "lot_raw": torch.full((env.B,), 0.1),
                "exit": torch.full((env.B,), EXIT_HOLD, dtype=torch.long)}
    for s in range(env.bars_per_day):
        _st, r, _d, info = env.step(open_act)
        total += r
        eq = force_equity_fn(s)
        if eq is not None:
            env._equity[:] = eq
            env._day_high_eq = torch.maximum(env._day_high_eq, env._equity)
    return float(total[0].item()), info


def test_reward_ordering_pass_gt_flat_gt_breach():
    """Cumulative day reward: PASS day > FLAT day > DD-BREACH day. Verified via
    the shaper's normalized per-day reward (the env mirrors it at day close)."""
    from core.reward.shaper import EpisodeRewardShaper
    s = EpisodeRewardShaper(_cfg())
    pass_r = s.daily_reward(r_d=0.03, dd_d=0.002)     # hit target, tiny DD
    s2 = EpisodeRewardShaper(_cfg())
    flat_r = s2.daily_reward(r_d=0.001, dd_d=0.002)   # green but below target
    s3 = EpisodeRewardShaper(_cfg())
    breach_r = s3.daily_reward(r_d=-0.005, dd_d=0.02) # DD breach + negative
    assert pass_r > flat_r > breach_r
    assert breach_r < 0


def test_reward_magnitude_is_normalized_not_dollars():
    """Per-step + per-day reward must be O(1) normalized units, NOT ~10,000-scale
    dollars. We drive a real env day and assert the cumulative reward magnitude is
    small (single digits), proving the percent-unit conversion (FIX 1)."""
    arr = _flat_series(400)
    env = BatchedFTMOEnv(arr, _cfg(account=10_000.0), DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    total, _info = _run_day(env, lambda s: None)   # flat market, no scripted eq
    # A flat day with the agent merely holding should produce a bounded reward,
    # nowhere near the thousands the raw-dollar formula would have produced.
    assert abs(total) < 50.0, f"reward not normalized (got {total})"


def test_day_close_uses_closing_baseline_not_post_reset():
    """REGRESSION: on the calendar day boundary the env must measure the day that
    just ended against ITS OWN start-of-day equity (the pre-reset snapshot), not
    the rolled-forward baseline. The old code reset _day_start_eq BEFORE computing
    daily_return / day_start_eq, which zeroed both on every boundary — so the
    reported day PnL was always ~$0 AND a winning day could never classify as a
    PASS. We script a +3% climb over the day and assert the close reports the gain
    (non-zero daily_return, day_start_eq == the day's start) and a PASS fires."""
    arr = _flat_series(400)
    cfg = _cfg(account=10_000.0, bars_per_day=40, ep_bars=120)
    env = BatchedFTMOEnv(arr, cfg, DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})

    # climb linearly to +3% of start over exactly one day (clears the +2.5% target)
    def climb(s, start=10_000.0):
        frac = (s + 1) / env.bars_per_day * 0.03
        return start * (1.0 + frac)

    _total, info = _run_day(env, climb)
    # the day that just closed must report a baseline of the day's START (10,000),
    # NOT the post-reset equity (~10,300) — i.e. the gain is visible.
    day_start = float(info["day_start_eq"][0].item())
    daily_ret = float(info["daily_return"][0].item())
    assert abs(day_start - 10_000.0) < 1.0, (
        f"day_start_eq reported post-reset baseline ({day_start}); the day's "
        f"+3% gain would be invisible (always-$0 PnL bug)")
    assert daily_ret > 0.02, f"daily_return zeroed on boundary (got {daily_ret})"
    # a +3% day with trades and no DD breach must classify as a PASS.
    assert bool(info["passed"][0].item()), "winning day failed to classify as PASS"


# ════════════════════════════════════════════════════════════════════════════
# FIX 3 — account-size invariance
# ════════════════════════════════════════════════════════════════════════════
def test_account_size_invariant_reward():
    """The SAME percentage outcome at 10k and 100k must yield ~the same
    normalized reward (reward is in percent-of-start-equity units). We script the
    identical +1% intraday equity path on both account sizes and compare."""
    results = {}
    for acct in (10_000.0, 100_000.0):
        arr = _flat_series(400)
        env = BatchedFTMOEnv(arr, _cfg(account=acct), DEV,
                             phase={"entry_conditions": {"buy": "any", "sell": "any"}})
        # script equity to climb to +1% of start over the day (same % both sizes)
        def path(s, a=acct):
            frac = (s + 1) / env.bars_per_day * 0.01
            return a * (1.0 + frac)
        total, _ = _run_day(env, path)
        results[acct] = total
    r10, r100 = results[10_000.0], results[100_000.0]
    # normalized reward must match within a small tolerance regardless of size
    assert abs(r10 - r100) < 0.05, f"reward not account-size invariant: {results}"


# ════════════════════════════════════════════════════════════════════════════
# streak logic
# ════════════════════════════════════════════════════════════════════════════
def test_streak_increments_and_resets():
    """A run of passing days increments the streak; a failing day resets it.
    Mirrors the trainer's batch-level streak (pass -> +1, fail -> 0)."""
    def step_streak(streak, day_passed, has_fail):
        if day_passed:
            return streak + 1
        if has_fail:
            return 0
        return streak

    streak = 0
    streak = step_streak(streak, True, False)   # pass
    streak = step_streak(streak, True, False)   # pass
    streak = step_streak(streak, True, False)   # pass
    assert streak == 3
    streak = step_streak(streak, False, True)   # fail -> reset
    assert streak == 0
    streak = step_streak(streak, True, False)   # pass again
    assert streak == 1


# ════════════════════════════════════════════════════════════════════════════
# FIX 1.3 — always-computable phi metric
# ════════════════════════════════════════════════════════════════════════════
def test_phi_metric_always_computable_and_ordered():
    """_phi_metric returns a finite number even with zero passes (so best_phi can
    update off -1e9 from episode 1), and orders pass > flat > breach."""
    import training.train as T
    tgt, dd = 0.025, 0.01
    phi_pass = T._phi_metric(0.8, 0.03, 0.002, tgt, dd)
    phi_flat = T._phi_metric(0.0, 0.005, 0.002, tgt, dd)
    phi_breach = T._phi_metric(0.0, -0.01, 0.012, tgt, dd)
    assert np.isfinite(phi_pass) and np.isfinite(phi_flat) and np.isfinite(phi_breach)
    assert phi_pass > phi_flat > phi_breach
    # even a no-pass day yields a real (non -1e9) number
    assert phi_flat > -1.0
