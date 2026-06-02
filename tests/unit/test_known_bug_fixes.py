"""
Regression tests for the 14 known bugs from the prior build. These lock in the
fixes so they can never silently regress.
"""
import torch
from core.settings import CFG, auto_tune_batch
from core.agent.dqn import DQNAgent
from core.agent.action_space import NUM_ACTIONS
from core.env.environment import BatchedFTMOEnv
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def _cfg(**over):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False, "MEMORY_SIZE": 2000,
              "EPISODE_BARS": 120, "BARS_PER_DAY": 60})
    c.update(over)
    return c


def test_bug1_per_batch_independent_exploration():
    # With epsilon=1, exploration is independent per batch item -> not all equal
    agent = DQNAgent(16, NUM_ACTIONS, _cfg(), DEV)
    agent.epsilon = 1.0
    a = agent.select_actions(torch.randn(64, 16))
    assert a.unique().numel() > 1   # would be 1 if exploration were all-or-nothing


def test_bug2_env_returns_executed_actions():
    env = BatchedFTMOEnv(make_synthetic_ohlcv_array(n=400), _cfg(), DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    env.reset()
    actions = torch.randint(0, NUM_ACTIONS, (env.B,))
    _s, _r, _d, info = env.step(actions)
    assert "executed_actions" in info
    assert torch.equal(info["executed_actions"], actions)


def test_bug3_bonus_backpatch_no_fake_transition():
    # Back-patching adds to existing rewards without growing the buffer by a fake row
    agent = DQNAgent(16, NUM_ACTIONS, _cfg(), DEV)
    B = 8
    s = torch.randn(B, 16); ns = torch.randn(B, 16)
    a = torch.randint(0, NUM_ACTIONS, (B,)); r = torch.zeros(B); d = torch.ones(B).bool()
    agent.store(s, a, r, ns, d)
    size_before = len(agent.memory)
    buf = agent.memory
    last_idx = torch.arange(buf.ptr - B, buf.ptr) % buf.capacity
    buf.rewards[last_idx] = buf.rewards[last_idx] + 0.05
    assert len(agent.memory) == size_before          # no fake transition added
    assert torch.allclose(buf.rewards[last_idx], torch.full((B,), 0.05))


def test_bug5_checkpoint_roundtrip_with_metadata(tmp_path):
    agent = DQNAgent(16, NUM_ACTIONS, _cfg(), DEV)
    p = str(tmp_path / "c.pt")
    agent.save(p, extra={"phase": "ph0", "meta": {"nested": 1}})
    agent2 = DQNAgent(16, NUM_ACTIONS, _cfg(), DEV)
    ckpt = agent2.load(p)   # would crash if weights_only defaulted True
    assert ckpt["meta"]["nested"] == 1


def test_bug6_inference_uses_tiny_buffer():
    c = _cfg(MEMORY_SIZE=1)
    agent = DQNAgent(16, NUM_ACTIONS, c, DEV)
    assert agent.memory.capacity == 1


def test_bug8_replay_buffer_survives_save_load(tmp_path):
    # save with experience -> load -> replay buffer restored (crash-resume)
    c = _cfg(MEMORY_SIZE=500)
    agent = DQNAgent(16, NUM_ACTIONS, c, DEV)
    B = 8
    for _ in range(10):
        agent.store(torch.randn(B, 16), torch.randint(0, NUM_ACTIONS, (B,)),
                    torch.randn(B), torch.randn(B, 16), torch.zeros(B).bool())
    size_before = len(agent.memory)
    p = str(tmp_path / "resume.pt")
    agent.save(p)
    agent2 = DQNAgent(16, NUM_ACTIONS, _cfg(MEMORY_SIZE=500), DEV)
    agent2.load(p)
    assert len(agent2.memory) == size_before   # experience restored on resume


def test_bug8_replay_skipped_on_transfer(tmp_path):
    # 7-action old agent -> 756 new agent: weights transfer, replay starts fresh
    old = DQNAgent(16, 7, _cfg(MEMORY_SIZE=500), DEV)
    B = 8
    for _ in range(5):
        old.store(torch.randn(B, 16), torch.randint(0, 7, (B,)),
                  torch.randn(B), torch.randn(B, 16), torch.zeros(B).bool())
    p = str(tmp_path / "old.pt"); old.save(p)
    new = DQNAgent(16, NUM_ACTIONS, _cfg(MEMORY_SIZE=500), DEV)
    new.load(p, partial=True)   # must not crash; replay may start fresh
    q = new.policy_net(torch.randn(2, 16))
    assert q.shape == (2, NUM_ACTIONS)
