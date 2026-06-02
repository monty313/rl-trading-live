"""
scripts/smoke_infer.py
Smoke test: build agent, load a checkpoint if present (else skip with WARNING),
run 1 inference action, assert a valid (direction, lot, sl, tp) decode. Exit 0.
    python scripts/smoke_infer.py [--checkpoint PATH]
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402
from core.agent.action_space import FLAT, BUY, SELL  # noqa: E402
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    device = get_device()
    cfg = auto_tune_batch(dict(CFG), device)
    cfg.update({"FEATURES": make_synthetic_ohlcv_array(n=400),
                "EPISODE_BARS": 120, "BARS_PER_DAY": 60,
                "USE_AMP": False, "USE_TORCH_COMPILE": False})
    env, agent, *_ = build_pipeline(cfg, device,
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint, partial=True)
        print(f"[smoke_infer] loaded {args.checkpoint}")
    else:
        print("[smoke_infer] WARNING: no checkpoint — using fresh weights")
    obs = env.reset()[0]
    direction, lot_raw, exit_act = agent.select_action(obs, deterministic=True)
    assert direction in (FLAT, BUY, SELL)
    assert 0.0 <= lot_raw <= 1.0
    print(f"[smoke_infer] dir={direction} lot_raw={lot_raw:.3f} exit={exit_act}")
    print("SMOKE_INFER OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
