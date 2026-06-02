"""
Regression tests for prior-build bugs that still apply under PPO.
DQN-specific bugs (epsilon, replay backpatch, 756 transfer) are retired with DQN.
"""
import torch
from core.settings import CFG, auto_tune_batch
from core.agent.ppo import PPOAgent
from core.agent.action_space import DIRECTION_DIM, FLAT
from core.env.environment import BatchedFTMOEnv
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def _cfg(**over):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False,
              "EPISODE_BARS": 120, "BARS_PER_DAY": 60})
    c.update(over)
    return c


def test_checkpoint_roundtrip_with_metadata(tmp_path):
    # weights_only=False path: checkpoints carry non-tensor metadata dicts
    agent = PPOAgent(16, _cfg(), DEV)
    p = str(tmp_path / "c.pt")
    agent.save(p, extra={"phase": "ph0", "meta": {"nested": 1}})
    agent2 = PPOAgent(16, _cfg(), DEV)
    ckpt = agent2.load(p)
    assert ckpt["meta"]["nested"] == 1


def test_env_returns_executed_direction():
    env = BatchedFTMOEnv(make_synthetic_ohlcv_array(n=400), _cfg(), DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    env.reset()
    out = {"direction": torch.zeros(env.B, dtype=torch.long),
           "lot_raw": torch.zeros(env.B), "exit": torch.zeros(env.B, dtype=torch.long)}
    _s, _r, _d, info = env.step(out)
    assert "executed_direction" in info


def test_partial_load_best_effort_on_shape_change(tmp_path):
    # saving a 16-dim agent and loading into a 24-dim agent must not crash (partial)
    a = PPOAgent(16, _cfg(), DEV)
    p = str(tmp_path / "a.pt"); a.save(p)
    b = PPOAgent(24, _cfg(), DEV)
    b.load(p, partial=True)   # mismatched trunk input -> skipped layers, no crash
    out = b.net(torch.randn(2, 24))
    assert out[0].shape == (2, DIRECTION_DIM)
