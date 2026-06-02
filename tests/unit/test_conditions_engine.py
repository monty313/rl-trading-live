"""Unit tests for condition parsing + action mask (RULE 12)."""
import torch
import pytest
from core.env import conditions_engine as CE
from core.agent.action_space import NUM_ACTIONS, BUY, SELL, HOLD, decode

DEV = torch.device("cpu")


def _mask_dirs(mask):
    """Return the set of allowed directions given a (NUM_ACTIONS,) mask."""
    allowed = set()
    for a in range(NUM_ACTIONS):
        if mask[a].item() > 0.5:
            allowed.add(decode(a)[0])
    return allowed


def test_any_always_true():
    assert CE.evaluate("any", {}) is True


def test_evaluate_uses_features():
    feats = {"cci_14": -150.0, "close": 1.2, "sma_20": 1.1}
    assert CE.evaluate("cci_14 < -100 and close > sma_20", feats) is True
    assert CE.evaluate("cci_14 > 100", feats) is False


def test_unknown_variable_raises_irac():
    with pytest.raises(CE.ConfigError) as e:
        CE.evaluate("rsi_99 > 50", {})
    assert "VARIABLE_REGISTRY" in str(e.value)


def test_buy_condition_masks_hold_and_sell():
    phase = {"entry_conditions": {"buy": "cci_14 < -100", "sell": "cci_14 > 100"}}
    feats = {"cci_14": -150.0}
    mask = CE.compute_action_mask(phase, feats, DEV)
    assert _mask_dirs(mask) == {BUY}


def test_no_condition_allows_all():
    phase = {"entry_conditions": {"buy": "any", "sell": "any"}}
    mask = CE.compute_action_mask(phase, {}, DEV)
    assert _mask_dirs(mask) == {HOLD, BUY, SELL}


def test_both_false_allows_all():
    phase = {"entry_conditions": {"buy": "cci_14 < -100", "sell": "cci_14 > 100"}}
    feats = {"cci_14": 0.0}
    mask = CE.compute_action_mask(phase, feats, DEV)
    assert _mask_dirs(mask) == {HOLD, BUY, SELL}
