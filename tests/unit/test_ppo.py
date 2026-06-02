"""Unit tests for the PPOAgent: action shapes, masking, update, checkpoint."""
import torch
from core.settings import CFG, auto_tune_batch
from core.agent.ppo import PPOAgent
from core.agent.action_space import DIRECTION_DIM, FLAT, BUY, SELL

DEV = torch.device("cpu")


def _cfg():
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False})
    return c


def test_select_actions_shapes():
    agent = PPOAgent(state_dim=32, cfg=_cfg(), device=DEV)
    out = agent.select_actions(torch.randn(8, 32))
    assert out["direction"].shape == (8,)
    assert out["lot_raw"].shape == (8,)
    assert ((out["lot_raw"] >= 0) & (out["lot_raw"] <= 1)).all()
    assert out["direction"].max() < DIRECTION_DIM


def test_direction_mask_blocks_flat():
    agent = PPOAgent(state_dim=16, cfg=_cfg(), device=DEV)
    # mask FLAT entirely: only BUY/SELL allowed (strategy-active rule)
    mask = torch.tensor([[0.0, 1.0, 1.0]]).expand(8, -1)
    out = agent.select_actions(torch.randn(8, 16), mask=mask)
    assert (out["direction"] != FLAT).all()   # FLAT never sampled


def test_update_runs_and_clears_buffer():
    agent = PPOAgent(state_dim=16, cfg=_cfg(), device=DEV)
    B = 4
    for _ in range(10):
        s = torch.randn(B, 16)
        out = agent.select_actions(s)
        agent.store(s, out, torch.randn(B), torch.zeros(B).bool(), None)
    loss = agent.update()
    assert loss is not None
    assert len(agent.buffer) == 0   # buffer cleared after update


def test_save_load_roundtrip(tmp_path):
    agent = PPOAgent(state_dim=16, cfg=_cfg(), device=DEV)
    p = str(tmp_path / "ppo.pt")
    agent.save(p, extra={"phase": "ph0", "phi": 0.5})
    agent2 = PPOAgent(state_dim=16, cfg=_cfg(), device=DEV)
    ckpt = agent2.load(p)
    assert ckpt["agent"] == "ppo" and ckpt["phase"] == "ph0"


def test_select_action_single_obs():
    agent = PPOAgent(state_dim=16, cfg=_cfg(), device=DEV)
    direction, lot_raw, exit_act = agent.select_action(torch.randn(16),
                                                       deterministic=True)
    assert direction in (FLAT, BUY, SELL)
    assert 0.0 <= lot_raw <= 1.0
