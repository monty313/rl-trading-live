"""Unit tests for the DQN agent: Q-shape, save/load roundtrip, transfer learning."""
import os, tempfile
import torch
from core.settings import CFG, auto_tune_batch
from core.agent.dqn import DQNAgent
from core.agent.action_space import NUM_ACTIONS

DEV = torch.device("cpu")


def _cfg():
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False, "MEMORY_SIZE": 1000})
    return c


def test_q_output_shape():
    agent = DQNAgent(state_dim=32, num_actions=NUM_ACTIONS, cfg=_cfg(), device=DEV)
    state = torch.randn(8, 32)
    a = agent.select_actions(state)
    assert a.shape == (8,)
    assert int(a.max()) < NUM_ACTIONS and int(a.min()) >= 0


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent(state_dim=32, num_actions=NUM_ACTIONS, cfg=_cfg(), device=DEV)
    p = str(tmp_path / "ckpt.pt")
    agent.save(p, extra={"phase": 1, "phi": 0.5})
    agent2 = DQNAgent(state_dim=32, num_actions=NUM_ACTIONS, cfg=_cfg(), device=DEV)
    ckpt = agent2.load(p)
    assert ckpt["phase"] == 1 and ckpt["num_actions"] == NUM_ACTIONS


def test_transfer_learning_reinits_output(tmp_path):
    # save an OLD 7-action agent, load into a 756-action agent with partial=True
    old = DQNAgent(state_dim=32, num_actions=7, cfg=_cfg(), device=DEV)
    p = str(tmp_path / "old.pt")
    old.save(p)
    new = DQNAgent(state_dim=32, num_actions=NUM_ACTIONS, cfg=_cfg(), device=DEV)
    new.load(p, partial=True)
    # output layer must now be 756 wide and produce valid Q-values
    q = new.policy_net(torch.randn(4, 32))
    assert q.shape == (4, NUM_ACTIONS)
    assert new.epsilon == _cfg()["TRANSFER_EPSILON"]


def test_mask_blocks_actions():
    agent = DQNAgent(state_dim=16, num_actions=NUM_ACTIONS, cfg=_cfg(), device=DEV)
    agent.epsilon = 0.0   # greedy
    state = torch.randn(4, 16)
    mask = torch.zeros(4, NUM_ACTIONS)
    mask[:, 5] = 1.0      # only action 5 allowed
    a = agent.select_actions(state, mask=mask)
    assert (a == 5).all()
