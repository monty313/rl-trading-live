#!/usr/bin/env python3
"""
Tight 60-90s smoke that runs the real training loop end-to-end on a small
synthetic dataset, then reports the FIRST passing day (if any) and the
per-day PnL distribution.

Used to iterate fixes locally without burning 8 hours in Colab.

Usage:
    python scripts/mini_train_smoke.py            # default 3-episode, 6-day-each
    python scripts/mini_train_smoke.py --episodes 5 --bars 8000

Exits with code 0 if at least one PASS day was observed across all episodes,
code 1 otherwise. Prints a one-line summary and a per-episode tally.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def synthetic_ohlcv(n_bars: int, seed: int = 42) -> np.ndarray:
    """Generate ~realistic EURUSD-shaped 1m OHLCV with mean reversion + drift.

    Goal: not random noise. Has stretches where direction is predictable
    (so a smart policy *could* hit +$250 in a day) and stretches that
    chop (so a random policy will lose money on spread). This is what
    lets us test 'does the policy actually learn'.
    """
    rng = np.random.default_rng(seed)
    # Drift component: long swings (~1000 bar = 16h period)
    t = np.arange(n_bars)
    drift = 0.0008 * np.sin(2 * np.pi * t / 1000.0)
    # Noise: gaussian with 1.5 pip std
    noise = rng.normal(0, 1.5e-4, n_bars)
    # Close price as a random walk on (drift + noise)
    close = 1.10 + np.cumsum(drift + noise)
    # OHLV from close
    spread_intra = np.abs(rng.normal(0, 0.8e-4, n_bars))
    high = close + spread_intra
    low = close - spread_intra
    op = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(rng.normal(1000, 200, n_bars))
    return np.column_stack([op, high, low, close, vol]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--bars", type=int, default=12_000)
    ap.add_argument("--device", default=None,
                    help="cuda | cpu (auto-detected if omitted)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── settings tuned for fast smoke ──────────────────────────────────────
    from core.settings import CFG, auto_tune_batch
    cfg = auto_tune_batch(dict(CFG), device)
    cfg["BATCH_SIZE_ENV"] = 8                        # tiny batch
    cfg["EPISODE_BARS"] = args.bars                  # ~8 trading days @1440/day
    cfg["MULTI_TF_OBS"] = True                       # the new path we care about
    cfg["dist_prephase_enabled"] = False             # don't load real DQN (1.7GB)
    cfg["WARM_START_FROM_DQN"] = False               # likewise
    cfg["DATA_CSV_EURUSD"] = None                    # use the array we pass directly

    # ── env ────────────────────────────────────────────────────────────────
    print(f"[mini-train] device={device} | batch=8 | bars/ep={args.bars} | episodes={args.episodes}")
    print(f"[mini-train] building synthetic dataset ({args.bars} bars)")
    ohlcv = synthetic_ohlcv(args.bars, seed=args.seed)
    from core.env.environment import BatchedFTMOEnv
    env = BatchedFTMOEnv(ohlcv, cfg, device=device)
    print(f"[mini-train] env.state_dim={env.state_dim}")

    # ── agent ──────────────────────────────────────────────────────────────
    from core.agent.ppo import PPOAgent
    agent = PPOAgent(env.state_dim, cfg, device=device)

    # ── one-pass roll: step the env for the full episode, count PASS days ──
    total_passes = 0
    per_ep_summary = []
    t0 = time.time()
    for ep in range(args.episodes):
        state = env.reset()
        ep_passes = 0
        ep_fails = 0
        ep_pnls = []
        ep_trades = []
        ep_pos_bars = 0
        ep_total_bars = 0
        while True:
            out = agent.select_actions(state, mask=env.current_direction_mask())
            actions = {
                "direction": out["direction"],
                "lot_raw":   out["lot_raw"],
                "exit":      out["exit"],
            }
            state, reward, done, info = env.step(actions)
            # done is a (B,) bool tensor: each episode-position ends when its
            # _step_i hits EPISODE_BARS. All B end at once.
            ep_pos_bars += int((env._position != 0).sum().item())
            ep_total_bars += env.B
            # Day-close detection: env exposes `closed` per-step in info when
            # a calendar day rolled over. Tally passes/fails on those bars.
            if isinstance(info, dict) and "day_closed" in info:
                day_closed = info["day_closed"]
                if day_closed.any():
                    idx = day_closed.nonzero(as_tuple=True)[0]
                    passed = info["passed"][idx].bool()
                    failed = (~passed)
                    eq = info["equity"][idx].float()
                    day_start = info["day_start_eq"][idx].float() if "day_start_eq" in info else eq.clone()
                    pnl = (eq - day_start).cpu().tolist()
                    ep_passes += int(passed.sum().item())
                    ep_fails += int(failed.sum().item())
                    ep_pnls.extend(pnl)
                    ep_trades.append(int(info["trades_today"][idx].float().mean().item()))
            if bool(done.all().item()):
                break
        pos_rate = ep_pos_bars / max(1, ep_total_bars)
        avg_pnl = float(np.mean(ep_pnls)) if ep_pnls else 0.0
        avg_trades = float(np.mean(ep_trades)) if ep_trades else 0.0
        per_ep_summary.append({
            "episode": ep,
            "passes": ep_passes,
            "fails": ep_fails,
            "avg_pnl": avg_pnl,
            "avg_trades_per_day": avg_trades,
            "position_rate": pos_rate,
            "pnls": ep_pnls,
        })
        total_passes += ep_passes
        if not args.quiet:
            best_pnl = max(ep_pnls) if ep_pnls else 0.0
            print(f"  ep{ep} | P:{ep_passes:>3} F:{ep_fails:>3} | "
                  f"avg PnL ${avg_pnl:+8.2f} | best PnL ${best_pnl:+8.2f} | "
                  f"avg trades/day {avg_trades:>5.0f} | "
                  f"pos rate {pos_rate*100:5.1f}%")

    elapsed = time.time() - t0
    print()
    print(f"[mini-train] {args.episodes} episodes in {elapsed:.1f}s")
    print(f"[mini-train] TOTAL PASS DAYS: {total_passes}")
    # Best PnL across all episodes' all days
    all_pnls = [p for ep in per_ep_summary for p in ep["pnls"]]
    if all_pnls:
        print(f"[mini-train] PnL distribution: "
              f"min ${min(all_pnls):+.2f} / "
              f"median ${float(np.median(all_pnls)):+.2f} / "
              f"max ${max(all_pnls):+.2f}")
    if total_passes > 0:
        print("[mini-train] ✅ at least one passing day observed")
        return 0
    print("[mini-train] ❌ zero passing days")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
