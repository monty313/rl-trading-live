"""Unit tests for BatchedFTMOEnv: shapes, step contract, masking."""
import torch
from core.settings import CFG, auto_tune_batch
from core.env.environment import BatchedFTMOEnv, build_multi_symbol_env
from core.agent.action_space import DIRECTION_DIM, FLAT
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def _cfg():
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"EPISODE_BARS": 120, "BARS_PER_DAY": 60, "LOOKBACK": 20})
    return c


def _env():
    arr = make_synthetic_ohlcv_array(n=400)
    return BatchedFTMOEnv(arr, _cfg(), DEV, instrument="EURUSD",
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})


def test_state_shape():
    env = _env()
    s = env.reset()
    assert s.shape == (env.B, env.state_dim)


def test_step_contract():
    env = _env()
    env.reset()
    out = {"direction": torch.zeros(env.B, dtype=torch.long),
           "lot_raw": torch.zeros(env.B),
           "exit": torch.zeros(env.B, dtype=torch.long)}
    s, r, d, info = env.step(out)
    assert s.shape == (env.B, env.state_dim)
    assert r.shape == (env.B,)
    assert d.dtype == torch.bool and d.shape == (env.B,)
    assert "equity" in info and "passed" in info and "executed_direction" in info


def test_direction_mask_shape():
    env = _env()
    env.reset()
    mask = env.current_direction_mask()
    assert mask.shape == (env.B, DIRECTION_DIM)


def test_episode_terminates():
    env = _env()
    env.reset()
    done_any = False
    flat = {"direction": torch.zeros(env.B, dtype=torch.long),
            "lot_raw": torch.zeros(env.B),
            "exit": torch.zeros(env.B, dtype=torch.long)}
    for _ in range(200):
        s, r, d, info = env.step(flat)
        if d.any():
            done_any = True
            break
    assert done_any


def test_multi_symbol_alignment():
    a = make_synthetic_ohlcv_array(n=400, seed=1)
    b = make_synthetic_ohlcv_array(n=350, seed=2)
    envs = build_multi_symbol_env({"EURUSD": a, "GBPUSD": b}, _cfg(), DEV)
    assert set(envs) == {"EURUSD", "GBPUSD"}
    assert envs["EURUSD"].T == envs["GBPUSD"].T == 350
