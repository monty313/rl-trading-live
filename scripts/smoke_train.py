"""
scripts/smoke_train.py
────────────────────────────────────────────────────────────────────────────
Smoke test: run 1 short training episode on synthetic 500-bar data (no MT5, no
Drive) and assert a checkpoint is written to a temp dir. Exits 0 on success.

    python scripts/smoke_train.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402
from training.checkpoint_manager import CheckpointManager  # noqa: E402
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array  # noqa: E402


def main() -> int:
    device = get_device()
    cfg = auto_tune_batch(dict(CFG), device)
    cfg.update({
        "FEATURES": make_synthetic_ohlcv_array(n=500),
        "EPISODE_BARS": 120, "BARS_PER_DAY": 60,
        "USE_AMP": False, "USE_TORCH_COMPILE": False,
        "MEMORY_SIZE": 2000, "BATCH_SIZE_RL": 32,
    })
    phase = {"name": "smoke", "entry_conditions": {"buy": "any", "sell": "any"}}
    env, agent, sizer, guard, gate = build_pipeline(cfg, device, phase=phase)

    print(f"[smoke_train] device={device} state_dim={env.state_dim}", flush=True)
    state = env.reset()
    done = torch.zeros(env.B, dtype=torch.bool, device=device)
    steps = 0
    while not done.all() and steps < env.ep_bars:
        mask = env.current_action_mask()
        actions = agent.select_actions(state, mask=mask)
        next_state, reward, done, info = env.step(actions)
        agent.store(state, actions, reward, next_state, done)
        agent.train_step()
        state = next_state
        steps += 1
    print(f"[smoke_train] episode complete in {steps} steps", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp, os.path.join(tmp, "manifest.json"))
        path = mgr.save(agent, "smoke", episode=1, phi=0.0, pass_rate=0.0)
        assert os.path.exists(path), "checkpoint not written"
        print(f"[smoke_train] checkpoint written -> {path}", flush=True)

    print("SMOKE_TRAIN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
