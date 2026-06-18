"""
tests/unit/test_action_parity_s6.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 6 — action-space audit. The action interpretation MUST be identical
in TRAINING and LIVE (zero drift), lot min/max/scaling must be correct, the
curriculum must widen narrow->wide at phase boundaries, the proportional scaler
must track target/DD, the force-gate must prevent going flat (except after a DD
halt), and invalid actions must be rejected LOUDLY (not silently zero-lotted).

These tests would have caught the live<->training lot-mapping DRIFT that existed
before this pass: live_runner used map_lot(raw, max_lot) over the FULL head range
while the training env used the per-phase curriculum window — the same policy
output meant a different lot live vs trained.
"""
from __future__ import annotations

import numpy as np
import torch

from core.agent.action_space import (map_lot, map_lot_curriculum,
                                      MIN_LOT, BUY, SELL, FLAT,
                                      EXIT_HOLD, EXIT_CLOSE)
from core.env.environment import BatchedFTMOEnv
from broker.live_runner import resolve_lot_window
from core.settings import CFG

DEV = torch.device("cpu")
B = 8


def _series(n=600, seed=3):
    rng = np.random.default_rng(seed)
    px = 1.10
    out = []
    for _ in range(n):
        px += rng.normal(0, 0.0002)
        out.append([px, px + 1e-4, px - 1e-4, px, 100.0])
    return np.asarray(out, dtype=np.float32)


def _cfg(**over):
    c = dict(CFG)
    c.update({"BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": 60,
              "EPISODE_BARS": 180, "USE_AMP": False, "USE_TORCH_COMPILE": False})
    c.update(over)
    return c


# ════════════════════════════════════════════════════════════════════════════
# 1. LIVE == TRAINING lot interpretation (the drift fix).
# ════════════════════════════════════════════════════════════════════════════
def test_live_lot_mapping_matches_training_curriculum():
    """For every phase window and a sweep of raw values, the LIVE mapping
    (resolve_lot_window + map_lot_curriculum) must agree with the TRAINING env
    mapping (_map_lot_curriculum) to within one MT5 lot step (0.01).

    They share the IDENTICAL formula (lot_lo + raw*(lot_hi-lot_lo)) and the
    IDENTICAL resolved window. The only permitted difference is the final MT5
    rounding: the env keeps a continuous float32 lot internally (it never places
    a real order), while the live runner rounds to MT5's 0.01 step — so a value
    that lands exactly on a .005 boundary can differ by one step due to float32
    vs float64 representation. That is rounding, not behavioural drift."""
    cfg = _cfg()
    arr = _series()
    for phase_name in cfg["LOT_CURRICULUM"]:
        if phase_name == "_default":
            continue
        phase = {"name": phase_name, "mask": None, "mask_type": "none"}
        env = BatchedFTMOEnv(arr, cfg, DEV, phase=phase)
        env.reset()
        lo, hi = resolve_lot_window(cfg, phase_name, cfg["MAX_LOT"])
        # env window must equal the live-resolved window EXACTLY (zero drift).
        assert abs(env._lot_lo - lo) < 1e-9 and abs(env._lot_hi - hi) < 1e-9, (
            f"window drift for {phase_name}: env=({env._lot_lo},{env._lot_hi}) "
            f"live=({lo},{hi})")
        for raw in (0.0, 0.13, 0.37, 0.5, 0.81, 1.0):
            env_lot = float(env._map_lot_curriculum(
                torch.tensor([raw], dtype=torch.float32))[0].item())
            live_lot = map_lot_curriculum(raw, lo, hi, 1.0)
            assert abs(env_lot - live_lot) <= 0.01 + 1e-9, (
                f"lot drift {phase_name} raw={raw}: env={env_lot} live={live_lot}")


def test_live_no_longer_uses_full_head_map_lot():
    """In a NARROW curriculum phase the old full-head mapping and the correct
    curriculum mapping must DIFFER for a mid raw — proving the fix actually
    changed behaviour (regression guard against silently reverting).

    NOTE: phase1_cci_align's window was widened to [0.01, 1.00] so PPO has the
    headroom to hit the $250 daily target. We pick phase3 here — still narrower
    than the full head ([0.10, 1.25] vs [0.01, 2.00]) — so the regression guard
    still proves curriculum != full-head sizing.
    """
    cfg = _cfg()
    lo, hi = resolve_lot_window(cfg, "phase3", cfg["MAX_LOT"])  # [0.10, 1.25]
    raw = 0.5
    old_full_head = map_lot(raw, cfg["MAX_LOT"])      # ~1.0
    correct = map_lot_curriculum(raw, lo, hi, 1.0)    # 0.10 + 0.5*1.15 = 0.675 → rounded to 0.67
    assert abs(correct - 0.67) < 1e-9
    assert abs(old_full_head - correct) > 0.25, "fix did not change live sizing"


# ════════════════════════════════════════════════════════════════════════════
# 2. Lot min / max / range.
# ════════════════════════════════════════════════════════════════════════════
def test_lot_never_below_mt5_minimum():
    for lo, hi in [(0.10, 0.50), (0.01, 2.0)]:
        assert map_lot_curriculum(0.0, lo, hi, 1.0) >= MIN_LOT
        # even a tiny proportional scale floors at MIN_LOT
        assert map_lot_curriculum(0.0, lo, hi, 0.01) >= MIN_LOT


def test_lot_respects_window_upper_at_raw_one():
    lo, hi = 0.10, 0.50
    assert abs(map_lot_curriculum(1.0, lo, hi, 1.0) - 0.50) < 1e-9
    lo, hi = 0.01, 2.0
    assert abs(map_lot_curriculum(1.0, lo, hi, 1.0) - 2.0) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# 3. Curriculum WIDENS narrow->wide across phase boundaries.
# ════════════════════════════════════════════════════════════════════════════
def test_curriculum_widens_across_phases():
    """Curriculum widens across the LATER strategy phases (phase2 onward).
    phase1_cci_align is the bootstrap learning phase: it gets a wider window
    than phase2 so PPO can actually hit the \$250 daily target while it
    learns direction, then later phases progressively expand toward the full
    [0.01, 2.00] window in live_improve. The monotonic-widening invariant
    therefore applies to phase2 → phase7, not from phase1."""
    cfg = _cfg()
    order = ["phase2", "phase3", "phase4", "phase5", "phase6", "phase7_full_ftmo"]
    his = [resolve_lot_window(cfg, p, cfg["MAX_LOT"])[1] for p in order]
    assert his == sorted(his), f"curriculum upper bound not monotonic: {his}"
    assert his[-1] >= his[0], "final phase must be at least as wide as the first"


def test_phase_change_refreshes_window_in_env():
    cfg = _cfg()
    arr = _series()
    env = BatchedFTMOEnv(arr, cfg, DEV,
                         phase={"name": "phase1_cci_align", "mask": None,
                                "mask_type": "none"})
    env.reset()
    narrow_hi = env._lot_hi
    env.phase = {"name": "phase7_full_ftmo", "mask": None, "mask_type": "none"}
    assert env._lot_hi >= narrow_hi, "phase advance did not widen lot window"


# ════════════════════════════════════════════════════════════════════════════
# 4. Proportional scaler vs target/DD.
# ════════════════════════════════════════════════════════════════════════════
def test_proportional_scaler_scales_lot():
    lo, hi = 0.10, 0.50
    base = map_lot_curriculum(0.5, lo, hi, 1.0)          # 0.30
    up = map_lot_curriculum(0.5, lo, hi, 1.5)            # 0.45
    down = map_lot_curriculum(0.5, lo, hi, 0.5)          # 0.15
    assert up > base > down
    assert abs(up - 0.45) < 1e-9 and abs(down - 0.15) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# 5. Force-gate prevents flat (except after a DD halt).
# ════════════════════════════════════════════════════════════════════════════
def test_force_gate_prevents_flat_when_gate_on_and_not_halted():
    cfg = _cfg(MAX_TRADES_PER_DAY=800)
    arr = _series()
    phase = {"name": "phase0_cci_extreme", "mask": "phase0_cci_extreme",
             "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15]}
    env = BatchedFTMOEnv(arr, cfg, DEV, phase=phase)
    env.reset()
    ones = torch.ones(env.B)
    bad_violations = 0
    for _ in range(120):
        abs_idx = env._abs_idx()
        gate_on = env._gate_on_batch(abs_idx)
        halted_pre = env._day_halted.clone()
        # Force the AGENT to choose FLAT every bar — the env force-entry must
        # override and keep a position open where the gate is on and not halted.
        env.step({"direction": (FLAT * ones).long(),
                  "lot_raw": (0.5 * ones).float(),
                  "exit": (EXIT_CLOSE * ones).long()})
        must_trade = gate_on & (~halted_pre) & (~env._day_halted)
        bad_violations += int((must_trade & (env._position == 0)).sum().item())
    assert bad_violations == 0, (
        f"{bad_violations} bars went flat under an active gate (force-entry broke)")


def test_halted_day_allows_flat():
    """After a DD halt the force-gate must NOT re-enter (halt overrides gate)."""
    cfg = _cfg()
    arr = _series()
    phase = {"name": "phase0_cci_extreme", "mask": "phase0_cci_extreme",
             "mask_type": "force_in_and_gate", "gate_timeframes": [1, 15]}
    env = BatchedFTMOEnv(arr, cfg, DEV, phase=phase)
    env.reset()
    env._day_halted[:] = True                # simulate a DD halt for all episodes
    ones = torch.ones(env.B)
    env.step({"direction": (FLAT * ones).long(),
              "lot_raw": (0.5 * ones).float(),
              "exit": (EXIT_HOLD * ones).long()})
    assert bool((env._position == 0).all().item()), \
        "force-entry fired on a HALTED day (must not re-enter after DD halt)"


# ════════════════════════════════════════════════════════════════════════════
# 6. Invalid actions are rejected LOUDLY, not silently zero-lotted.
# ════════════════════════════════════════════════════════════════════════════
def test_invalid_direction_raises_loudly():
    """An out-of-range direction code must NOT be silently coerced to a flat,
    zero-lot no-op day — the env must raise. Before this pass the env mapped only
    {BUY,SELL} to a side and left any other code as 0 (silent flat)."""
    import pytest
    cfg = _cfg()
    arr = _series()
    env = BatchedFTMOEnv(arr, cfg, DEV,
                         phase={"name": "free", "mask": None, "mask_type": "none"})
    env.reset()
    ones = torch.ones(env.B)
    with pytest.raises(ValueError, match="invalid direction"):
        env.step({"direction": (99 * ones).long(),
                  "lot_raw": (0.5 * ones).float(),
                  "exit": (EXIT_HOLD * ones).long()})


def test_invalid_exit_raises_loudly():
    import pytest
    cfg = _cfg()
    arr = _series()
    env = BatchedFTMOEnv(arr, cfg, DEV,
                         phase={"name": "free", "mask": None, "mask_type": "none"})
    env.reset()
    ones = torch.ones(env.B)
    with pytest.raises(ValueError, match="invalid exit"):
        env.step({"direction": (BUY * ones).long(),
                  "lot_raw": (0.5 * ones).float(),
                  "exit": (7 * ones).long()})


def test_nonfinite_lot_raw_raises_loudly():
    import pytest
    cfg = _cfg()
    arr = _series()
    env = BatchedFTMOEnv(arr, cfg, DEV,
                         phase={"name": "free", "mask": None, "mask_type": "none"})
    env.reset()
    ones = torch.ones(env.B)
    with pytest.raises(ValueError, match="non-finite"):
        env.step({"direction": (BUY * ones).long(),
                  "lot_raw": (float("nan") * ones).float(),
                  "exit": (EXIT_HOLD * ones).long()})
