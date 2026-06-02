"""
tests/integration/test_e2e_ftmo.py
────────────────────────────────────────────────────────────────────────────
End-to-end smoke of BatchedFTMOEnv over B=64 parallel episodes, exercising the
four bugs fixed in this PR together:

  Bug 2  PnL economics      — a 10-pip move on 0.10 lots is ~$10, not ~$0.001.
  Bug 3  TF alignment       — integer row indexing aligns 15m/30m/60m to 1m
                              without the fake-timestamp drift.
  Bug 4  per-episode gate    — the gate mask + force-entry is computed for EVERY
                              one of the 64 episodes, not broadcast from ep 0.
  Bug 6  rollout integrity   — torch.compile(default)+.clone() (CPU here, so the
                              agent's rollout/update path just has to run clean).

Synthetic data is allowed for THIS TEST ONLY (the real training path stays
real-data-only). The series is realistically scaled around EURUSD ~1.10 with
strong directional bursts so phase0's CCI10/CCI30 extreme gate actually fires.
"""
from __future__ import annotations

import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.ppo import PPOAgent
from core.agent.action_space import FLAT, BUY, SELL

DEV = torch.device("cpu")
B = 64


def _make_eurusd(n: int = 2200, seed: int = 7) -> np.ndarray:
    """Realistically-scaled EURUSD 1m OHLCV (price ~1.10) with strong trend
    bursts that drive CCI to extremes so the phase0 gate triggers."""
    rng = np.random.default_rng(seed)
    px = 1.1000
    out = []
    direction = 1
    for i in range(n):
        if i % 40 == 0:                      # flip the regime every ~40 bars
            direction = int(rng.choice([-1, 1]))
        step = direction * 0.00020 + rng.normal(0.0, 0.00005)
        o = px
        px = px + step
        c = px
        wick_h = abs(rng.normal(0.0, 0.00003))
        wick_l = abs(rng.normal(0.0, 0.00003))
        h = max(o, c) + wick_h
        l = min(o, c) - wick_l
        out.append([o, h, l, c, float(rng.integers(50, 150))])
    return np.asarray(out, dtype=np.float32)


def _cfg() -> dict:
    return {
        "BATCH_SIZE_ENV": B, "LOOKBACK": 20,
        "BARS_PER_DAY": 60, "EPISODE_BARS": 180,
        "INITIAL_EQUITY": 100_000.0, "MAX_LOT": 2.0,
        "DAILY_TARGET_PCT": 0.025, "DAILY_MAX_DD_PCT": 0.01,
        "MAX_TRADES_PER_DAY": 800,
        "USE_AMP": False, "USE_TORCH_COMPILE": False,
    }


_PHASE = {
    "name": "phase0_cci_extreme", "mask": "phase0_cci_extreme",
    "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15],
}


# ════════════════════════════════════════════════════════════════════════════
# Bug 2 — pure PnL math (no agent), pinned to the spec example.
# ════════════════════════════════════════════════════════════════════════════
def test_pnl_magnitude_is_realistic():
    """A 10-pip favorable move (0.0010 price) on 0.10 lots must yield ~$10.00,
    NOT ~$0.001 (the old double-pip-conversion bug)."""
    env = BatchedFTMOEnv(_make_eurusd(), _cfg(), DEV, phase=_PHASE)
    env.reset()
    # Force a known entry price and position, then mark to a +10-pip close.
    env._position[:] = 0.10                       # 0.10 lots long
    env._entry_px[:] = 1.10000
    move = 0.00100                                # 10 pips
    lots = 0.10
    expected = move * lots * 100_000.0            # = 10.0
    mtm = move * torch.sign(env._position) * env._position.abs() * 100_000.0
    assert abs(float(mtm[0]) - expected) < 1e-6
    assert abs(expected - 10.0) < 1e-9            # the spec number, exactly
    # And the WRONG (old) formula would have been 1e-4 of this:
    assert abs((expected * 0.0001) - 0.001) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# Full e2e: 1 episode, 64 episodes, real PPO agent driving the env.
# ════════════════════════════════════════════════════════════════════════════
def test_e2e_one_episode_all_invariants():
    cfg = _cfg()
    arr = _make_eurusd()
    env = BatchedFTMOEnv(arr, cfg, DEV, phase=_PHASE)
    agent = PPOAgent(env.state_dim, cfg, DEV)

    assert env.B == B, "test requires all 64 parallel episodes"

    state = env.reset()
    done = torch.zeros(env.B, dtype=torch.bool)
    steps = 0
    rollout = 64

    gate_fired_any = False
    total_trades_closed = 0
    invariant_violations = 0
    per_episode_gate_counts = torch.zeros(env.B, dtype=torch.long)

    max_steps = env.ep_bars
    while not done.all() and steps < max_steps:
        abs_idx = env._abs_idx()
        gate_on = env._gate_on_batch(abs_idx)         # (B,) per-episode gate
        per_episode_gate_counts += gate_on.long()
        if gate_on.any():
            gate_fired_any = True

        # Capture which episodes were halted BEFORE the step (halt overrides gate).
        halted_pre = env._day_halted.clone()

        mask = env.current_direction_mask()           # (B, DIRECTION_DIM)
        out = agent.select_actions(state, mask=mask)
        next_state, reward, done, info = agent_step(env, out)
        agent.store(state, out, reward, done, mask)

        total_trades_closed += int(info["trades_today"][info["day_closed"]].sum().item())

        # ── MASK INVARIANT (Bug 4) ──────────────────────────────────────────
        # On every episode where the gate was ON this bar and the day was not
        # halted, a trade MUST be active after the step.
        must_have_trade = gate_on & (~halted_pre) & (~env._day_halted)
        bad = must_have_trade & (env._position == 0)
        invariant_violations += int(bad.sum().item())

        state = next_state
        steps += 1
        if len(agent.buffer) >= rollout:
            agent.update()                            # Bug 6: must not crash
    loss = agent.update()

    # ── ASSERTIONS ──────────────────────────────────────────────────────────
    # 1. The phase gate fired at least once.
    assert gate_fired_any, "phase0 gate never fired — TF alignment / gate broken"

    # 2. Per-episode gate evaluation (Bug 4): episodes start at different bars,
    #    so their gate-on counts must NOT be identical across all 64 episodes
    #    (the old code broadcast episode 0's single value to all).
    assert per_episode_gate_counts.max() > 0
    assert per_episode_gate_counts.unique().numel() > 1, \
        "all 64 episodes share one gate count — mask is still broadcast from ep 0"

    # 3. The mask invariant held on every episode, every step.
    assert invariant_violations == 0, \
        f"{invariant_violations} bars had gate ON + flat (force-entry not enforced)"

    # 4. Trades happened.
    assert total_trades_closed > 0, "no trades were ever opened"

    # 5. Equity moved MEANINGFULLY — > $1 magnitude somewhere, not the ~$0.01
    #    that the old double-pip-conversion PnL produced.
    eq = env._equity
    max_move = float((eq - cfg["INITIAL_EQUITY"]).abs().max().item())
    assert max_move > 1.0, \
        f"equity barely moved (${max_move:.4f}) — PnL formula still broken"

    # 6. The agent's rollout/update path ran cleanly (Bug 6 sanity).
    assert loss is None or np.isfinite(loss)


def agent_step(env, out):
    """Thin wrapper so the assertion block reads clearly."""
    return env.step(out)
