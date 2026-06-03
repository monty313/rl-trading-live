"""
tests/unit/test_gate_precompute.py
────────────────────────────────────────────────────────────────────────────
PARITY proof for the vectorized phase-gate fast path (the 4-hour stall fix).

The env now derives the per-episode action mask from a PRECOMPUTED length-T gate
signal (core/env/gate_precompute.py) instead of rebuilding pandas row-dicts every
bar. These tests assert the fast path returns BAR-FOR-BAR identical masks to the
original scalar conditions_engine.compute_action_mask path, for every curriculum
phase and for a string-condition phase. If the optimization ever drifts from the
authoritative gate semantics, this fails loudly.
"""
import numpy as np
import torch

from core.settings import CFG, auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.env import conditions_engine as CE
from core.agent.action_space import FLAT
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")

# All named curriculum phases (name -> mask_type + gate timeframes from registry).
NAMED_PHASES = [
    {"name": n, "mask": n, "mask_type": mt, "gate_timeframes": tfs}
    for n, (_fn, mt, tfs) in CE.MASK_REGISTRY.items()
]


def _cfg(**over):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False,
              "EPISODE_BARS": 300, "BARS_PER_DAY": 200})
    c.update(over)
    return c


def _scalar_mask_for_bar(env, abs_i: int, is_flat: bool) -> np.ndarray:
    """Reference mask via the ORIGINAL scalar path: build per-TF row dicts the
    slow way and call compute_action_mask. Returns a (DIRECTION_DIM,) np array."""
    rows = {}
    for tf, df in env._tf_indicators.items():
        recs = df.to_dict("records")
        pos = env._tf_pos(abs_i, tf, len(df))
        rows[tf] = recs[pos]
    m, _me = CE.compute_action_mask(env.phase, rows, DEV, is_flat=is_flat)
    return m.detach().cpu().numpy()


def test_named_gate_parity_all_phases():
    """For every named phase, the precomputed gate_on must reproduce the scalar
    mask exactly, for both flat and in-trade states, across many bars."""
    feats = make_synthetic_ohlcv_array(n=1200, seed=7)
    for phase in NAMED_PHASES:
        env = BatchedFTMOEnv(feats, _cfg(), DEV, phase=phase)
        sig = env._gate_signal
        assert sig is not None and sig["kind"] == "named", phase["name"]
        gate_on = sig["gate_on"].cpu().numpy()
        # sample bars across the series (skip warmup region for indicator validity)
        for abs_i in range(100, len(feats) - 1, 17):
            for is_flat in (True, False):
                ref = _scalar_mask_for_bar(env, abs_i, is_flat)
                # vectorized mask for a single (gate_on, is_flat)
                go = torch.tensor([bool(gate_on[abs_i])])
                fl = torch.tensor([is_flat])
                # build a 1-wide mask via the env helper (B=1 view)
                env.B = 1
                m, _me = env._named_mask_from_gate(go, fl)
                got = m[0].cpu().numpy()
                assert np.array_equal(got, ref), (
                    f"{phase['name']} bar={abs_i} flat={is_flat} "
                    f"got={got} ref={ref} gate_on={bool(gate_on[abs_i])}")


def test_string_gate_parity():
    """String-condition phase: precomputed buy_on/sell_on reproduce the scalar
    mask exactly."""
    feats = make_synthetic_ohlcv_array(n=800, seed=11)
    phase = {"name": "str", "entry_conditions": {
        "buy": "cci10 > 50", "sell": "cci10 < -50"}}
    env = BatchedFTMOEnv(feats, _cfg(), DEV, phase=phase)
    sig = env._gate_signal
    assert sig is not None and sig["kind"] == "string"
    buy_on = sig["buy_on"].cpu().numpy()
    sell_on = sig["sell_on"].cpu().numpy()
    for abs_i in range(50, len(feats) - 1, 13):
        # scalar reference uses the 1m feature row dict
        from core.env.indicators import feature_row_dict
        rows = {1: feature_row_dict(env.features[abs_i])}
        ref, _ = CE.compute_action_mask(phase, rows, DEV, is_flat=True)
        ref = ref.cpu().numpy()
        env.B = 1
        m, _ = env._string_mask_from_triggers(
            torch.tensor([bool(buy_on[abs_i])]),
            torch.tensor([bool(sell_on[abs_i])]))
        assert np.array_equal(m[0].cpu().numpy(), ref), (
            f"bar={abs_i} got={m[0].cpu().numpy()} ref={ref}")


def test_free_phase_no_gating():
    """A free phase (no mask, no string conditions) yields no gate signal and an
    allow-all mask."""
    feats = make_synthetic_ohlcv_array(n=400, seed=2)
    phase = {"name": "free", "mask_type": "free"}
    env = BatchedFTMOEnv(feats, _cfg(), DEV, phase=phase)
    assert env._gate_signal is None
    mask = env.current_direction_mask()
    assert torch.all(mask == 1.0)


def test_phase_setter_refreshes_signal():
    """Reassigning env.phase swaps the cached gate signal (used by train.py's
    run_phase to advance the curriculum)."""
    feats = make_synthetic_ohlcv_array(n=600, seed=5)
    env = BatchedFTMOEnv(feats, _cfg(), DEV, phase=NAMED_PHASES[0])
    first = env._gate_signal
    assert first is not None
    env.phase = {"name": "free", "mask_type": "free"}
    assert env._gate_signal is None
    env.phase = NAMED_PHASES[1]
    assert env._gate_signal is not None and env._gate_signal["kind"] == "named"
