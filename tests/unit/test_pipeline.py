"""Unit tests for build_pipeline wiring."""
import torch
from core.pipeline import build_pipeline
from core.settings import CFG
from core.env.environment import BatchedFTMOEnv
from core.agent.ppo import PPOAgent
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def test_returns_five_objects():
    c = dict(CFG)
    c.update({"FEATURES": make_synthetic_ohlcv_array(n=600),
              "EPISODE_BARS": 120, "BARS_PER_DAY": 60,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    env, agent, sizer, guard, gate = build_pipeline(c, DEV)
    assert isinstance(env, BatchedFTMOEnv)
    assert isinstance(agent, PPOAgent)
    assert env.state_dim == agent.state_dim   # state_dim matches agent input dim
    assert guard is not None and gate is not None and sizer is not None
