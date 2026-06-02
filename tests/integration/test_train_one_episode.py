"""Integration: one full training episode writes a checkpoint."""
import torch
from core.settings import CFG, auto_tune_batch
from core.pipeline import build_pipeline
from training.checkpoint_manager import CheckpointManager
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def test_train_one_episode(tmp_path):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"FEATURES": make_synthetic_ohlcv_array(n=500),
              "EPISODE_BARS": 120, "BARS_PER_DAY": 60,
              "ROLLOUT_STEPS": 32, "USE_AMP": False, "USE_TORCH_COMPILE": False})
    env, agent, *_ = build_pipeline(c, DEV,
        phase={"name": "p", "entry_conditions": {"buy": "any", "sell": "any"}})
    state = env.reset()
    done = torch.zeros(env.B, dtype=torch.bool)
    steps = 0
    while not done.all() and steps < env.ep_bars:
        mask = env.current_direction_mask()
        out = agent.select_actions(state, mask=mask)
        ns, r, done, info = env.step(out)
        agent.store(state, out, r, done, mask)
        if len(agent.buffer) >= 32:
            agent.update()
        state = ns; steps += 1
    agent.update()
    mgr = CheckpointManager(str(tmp_path), str(tmp_path / "manifest.json"))
    path = mgr.save(agent, "p", 1, phi=0.0, pass_rate=0.0)
    assert __import__("os").path.exists(path)
