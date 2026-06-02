"""Unit tests for conditions engine: string conditions, named phase masks, semantics."""
import torch
import pytest
from core.env import conditions_engine as CE
from core.agent.action_space import DIRECTION_DIM, BUY, SELL, FLAT

DEV = torch.device("cpu")


def _mask_dirs(mask):
    return {d for d in range(DIRECTION_DIM) if float(mask[d]) > 0.5}


# ── string-condition path ──
def test_any_always_true():
    assert CE.evaluate("any", {}) is True


def test_evaluate_uses_features():
    feats = {"cci30": -150.0, "close": 1.2, "sma_20": 1.1}
    assert CE.evaluate("cci30 < -100 and close > sma_20", feats) is True


def test_unknown_variable_raises_irac():
    with pytest.raises(CE.ConfigError) as e:
        CE.evaluate("rsi_99 > 50", {})
    assert "VARIABLE_REGISTRY" in str(e.value)


def test_string_buy_condition_masks_to_buy():
    phase = {"entry_conditions": {"buy": "cci30 < -100", "sell": "cci30 > 100"}}
    rows = {1: {"cci30": -150.0}}
    mask, must = CE.compute_action_mask(phase, rows, DEV)
    assert _mask_dirs(mask) == {BUY} and must is False


def test_free_allows_all():
    phase = {"mask": None, "mask_type": "free"}
    mask, must = CE.compute_action_mask(phase, {1: {}}, DEV)
    assert _mask_dirs(mask) == {FLAT, BUY, SELL}


# ── named phase mask truth tables ──
def test_phase0_cci_extreme_both_high():
    r = {"cci30": 150.0, "cci100": 120.0}
    assert CE.phase0_cci_extreme(r, r) is True
    r2 = {"cci30": 150.0, "cci100": 50.0}        # cci100 not extreme
    assert CE.phase0_cci_extreme(r2, r2) is False


def test_phase0_opposite_direction_false():
    up = {"cci30": 150.0, "cci100": 120.0}
    dn = {"cci30": -150.0, "cci100": -120.0}
    assert CE.phase0_cci_extreme(up, dn) is False   # TFs disagree


def test_phase1_cci_align():
    r = {"cci30": 10.0, "cci30_sma1_sh8": 5.0,
         "cci100": 8.0, "cci100_sma1_sh8": 3.0}     # both above their SMA
    assert CE.phase1_cci_align(r, r) is True


def test_phase6_atr_expansion():
    r = {"atr14": 0.002, "atr14_sma1_sh8": 0.001,
         "atr45": 0.003, "atr45_sma1_sh8": 0.0025}
    assert CE.phase6_atr_expansion(r, r) is True


# ── mask_type semantics ──
def test_force_in_and_gate_active_masks_flat_always():
    phase = {"mask": "phase0_cci_extreme", "mask_type": "force_in_and_gate",
             "gate_timeframes": [1, 15]}
    extreme = {"cci30": 150.0, "cci100": 120.0}
    rows = {1: extreme, 15: extreme}
    # flat + gate active -> must open (BUY or SELL only, no FLAT)
    mask, must = CE.compute_action_mask(phase, rows, DEV, is_flat=True)
    assert _mask_dirs(mask) == {BUY, SELL} and must is True
    # already in a position + gate active -> can hold (FLAT), flip, or stay — agent decides
    mask2, must2 = CE.compute_action_mask(phase, rows, DEV, is_flat=False)
    assert _mask_dirs(mask2) == {FLAT, BUY, SELL} and must2 is False


def test_force_in_and_gate_blocks_entries_when_condition_false():
    phase = {"mask": "phase0_cci_extreme", "mask_type": "force_in_and_gate",
             "gate_timeframes": [1, 15]}
    calm = {"cci30": 0.0, "cci100": 0.0}
    rows = {1: calm, 15: calm}
    # flat + gate inactive -> can only stay flat (no new entries)
    mask, must = CE.compute_action_mask(phase, rows, DEV, is_flat=True)
    assert _mask_dirs(mask) == {FLAT}
    assert must is False
    # in a trade + gate inactive -> can hold or close, but no new entry flip
    mask2, must2 = CE.compute_action_mask(phase, rows, DEV, is_flat=False)
    assert _mask_dirs(mask2) == {FLAT, BUY, SELL} and must2 is False


def test_open_gate_allows_all_when_true_hold_only_when_false():
    phase = {"mask": "phase1_cci_align", "mask_type": "open_gate",
             "gate_timeframes": [1, 15]}
    aligned = {"cci30": 10.0, "cci30_sma1_sh8": 5.0,
               "cci100": 8.0, "cci100_sma1_sh8": 3.0}
    rows = {1: aligned, 15: aligned}
    mask, must = CE.compute_action_mask(phase, rows, DEV, is_flat=True)
    assert _mask_dirs(mask) == {BUY, SELL}   # active gate -> no FLAT, in any phase
    misaligned = {"cci30": 10.0, "cci30_sma1_sh8": 5.0,
                  "cci100": -8.0, "cci100_sma1_sh8": -3.0}   # disagree
    rows2 = {1: misaligned, 15: misaligned}
    mask2, _ = CE.compute_action_mask(phase, rows2, DEV)
    assert _mask_dirs(mask2) == {FLAT}   # gate closed -> no new entries
