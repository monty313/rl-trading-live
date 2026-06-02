"""
training/train.py
────────────────────────────────────────────────────────────────────────────
Main GPU training loop. Ported from gpu_rl_trading/training/train.py (REPO1)
with the spec's changes (STEP 4.14):

  (a) Phases loaded from config/phases.yaml via conditions_engine.
  (b) After all numbered phases complete -> LIVE_IMPROVE infinite phase
      (no episode cap, no advance target; eval every EVAL_EVERY, update best_eval.pt).
  (c) Always resume from checkpoint_manager.find_best_resume() unless --force-fresh.
  (d) CLI: --csv --checkpoint-dir --metrics-dir --manifest --resume
           --start-phase --force-fresh
  (e) Heartbeat file every ~60s: {metrics_dir}/heartbeat_training.txt

Run:
  python -m training.train --csv DATA.csv --checkpoint-dir CK --metrics-dir M \
      --manifest CK/manifest.json --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from core.pipeline import build_pipeline, load_ohlcv_csv  # noqa: E402
from core.reward.shaper import EpisodeRewardShaper  # noqa: E402
from training.checkpoint_manager import CheckpointManager  # noqa: E402
from training.eval_loop import run_eval  # noqa: E402


def _load_phases(repo_root: str) -> list:
    with open(os.path.join(repo_root, "config", "phases.yaml")) as f:
        data = yaml.safe_load(f)
    phases = data.get("phases", []) if data else []
    return sorted(phases, key=lambda p: p.get("order", 0))


def _write_heartbeat(metrics_dir: str, episode: int, phase: str, status="running"):
    os.makedirs(metrics_dir, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "episode": episode, "phase": phase, "status": status}
    with open(os.path.join(metrics_dir, "heartbeat_training.txt"), "w") as f:
        json.dump(payload, f)


def train(args) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = get_device()
    cfg = auto_tune_batch(dict(CFG), device)
    cfg["DATA_CSV_EURUSD"] = args.csv
    cfg["TRADE_LOG"] = os.path.join(args.metrics_dir or "logs", "daily_trade_log.csv")

    phases = _load_phases(repo_root)
    print(f"[train] device={device}  phases={[p['name'] for p in phases]}", flush=True)

    ckpt_mgr = CheckpointManager(args.checkpoint_dir, args.manifest)
    ckpt_mgr.load_manifest()

    # Build the pipeline on the first (or start) phase.
    start_idx = max(0, min(args.start_phase, len(phases) - 1))
    first_phase = phases[start_idx] if phases else {"name": "live_improve",
                                                    "entry_conditions": {"buy": "any", "sell": "any"}}
    env, agent, sizer, guard, gate = build_pipeline(cfg, device, phase=first_phase)
    shaper = EpisodeRewardShaper(cfg)

    # Resume (HARD RULE 9) unless --force-fresh.
    if not args.force_fresh:
        resume = ckpt_mgr.find_best_resume()
        if resume is not None:
            print(f"[train] resuming from {resume}", flush=True)
            agent.load(str(resume), partial=True)   # partial best-effort transfer on shared layers
        else:
            print("[train] no checkpoint found — fresh start", flush=True)

    global_ep = 0
    last_hb = 0.0
    best_phi = -1e9

    def run_phase(phase: dict, infinite: bool):
        nonlocal global_ep, last_hb, best_phi
        env.phase = phase
        max_eps = phase.get("max_episodes", cfg["MAX_EPISODES_PER_PHASE"])
        ep_in_phase = 0
        while True:
            if not infinite and max_eps != -1 and ep_in_phase >= max_eps:
                break
            state = env.reset()
            shaper.global_ep = global_ep
            done = torch.zeros(env.B, dtype=torch.bool, device=device)
            steps = 0
            day_num = 0
            ep_days_passed = ep_days_failed = ep_days_ok = 0
            ep_total_reward = 0.0
            max_steps = env.ep_bars
            rollout = int(cfg.get("ROLLOUT_STEPS", 2048))
            initial_equity = float(cfg.get("INITIAL_EQUITY", 100_000))
            # ── PPO on-policy rollout: collect transitions, update periodically ──
            while not done.all() and steps < max_steps:
                mask = env.current_direction_mask()
                out = agent.select_actions(state, mask=mask)
                next_state, reward, done, info = env.step(out)
                agent.store(state, out, reward, done, mask)
                state = next_state
                steps += 1
                ep_total_reward += float(reward[0].item())

                # ── print a line for each day that closes (env batch item 0) ──
                if info["day_closed"][0].item():
                    day_num += 1
                    eq       = float(info["equity"][0].item())
                    passed   = bool(info["passed"][0].item())
                    halted   = bool(info["day_halted"][0].item())
                    streak   = int(info["pass_streak"][0].item())
                    trades   = int(info["trades_today"][0].item())
                    day_ret  = (eq - initial_equity) / (initial_equity + 1e-9) * 100

                    rw       = cfg.get("REWARD", {}) or {}
                    pass_b   = float(rw.get("pass_day_bonus",   2.0))
                    ok_b     = float(rw.get("ok_day_bonus",     0.5))
                    fail_b   = float(rw.get("fail_day_penalty", -2.0))
                    streak_b = float(rw.get("streak_scale",     0.1)) * streak

                    if passed:
                        ep_days_passed += 1
                        label  = "✅ PASS"
                        base_r = pass_b
                    elif halted or day_ret < 0:
                        ep_days_failed += 1
                        label  = "❌ FAIL"
                        base_r = fail_b
                    else:
                        ep_days_ok += 1
                        label  = "🟡 OK  "
                        base_r = ok_b

                    halt_tag = "  ⛔ DD-HALT" if halted else ""
                    print(
                        f"    Day {day_num:>2}  {label}"
                        f"  │  equity {eq:>10,.2f}  ({day_ret:+.3f}%)"
                        f"  │  trades {trades:>3}"
                        f"  │  streak {streak}"
                        f"  │  reward {base_r+streak_b:+.2f}"
                        f"  (base {base_r:+.1f}  streak {streak_b:+.2f})"
                        f"{halt_tag}",
                        flush=True,
                    )

                if len(agent.buffer) >= rollout:
                    agent.update()        # PPO update, then buffer clears
            loss = agent.update()          # flush any remaining rollout at episode end

            global_ep += 1
            ep_in_phase += 1

            # ── episode summary ───────────────────────────────────────────────
            eq_now   = float(env._equity.mean().item())
            ep_ret   = (eq_now - initial_equity) / (initial_equity + 1e-9) * 100
            loss_str = f"  loss {loss:.4f}" if loss is not None else ""
            print(
                f"  {'─'*70}\n"
                f"  Episode {global_ep:>4}  [{phase['name']}]"
                f"  │  {ep_days_passed}✅ {ep_days_ok}🟡 {ep_days_failed}❌"
                f"  │  equity {eq_now:>10,.2f}  ({ep_ret:+.3f}%)"
                f"  │  reward {ep_total_reward:+.2f}"
                f"{loss_str}  │  best_phi {best_phi:.4f}",
                flush=True,
            )

            if global_ep % cfg["CHECKPOINT_EVERY"] == 0:
                ckpt_mgr.save(agent, phase["name"], global_ep, phi=best_phi,
                              pass_rate=0.0)
            if global_ep % cfg["EVAL_EVERY"] == 0:
                metrics = run_eval(env, agent, cfg, n_days=2)
                if metrics["phi"] > best_phi:
                    best_phi = metrics["phi"]
                    ckpt_mgr.save(agent, phase["name"], global_ep,
                                  phi=best_phi, pass_rate=metrics["pass_rate"],
                                  name="best_eval.pt")
                new_best = metrics["phi"] > best_phi
                print(
                    f"  {'═'*70}\n"
                    f"  🔍 EVAL  pass {metrics['pass_rate']*100:.1f}%"
                    f"  │  phi {metrics['phi']:.4f}"
                    f"  │  avg_ret {metrics['avg_daily_return']*100:+.3f}%"
                    f"  │  avg_dd {metrics['avg_daily_dd']*100:.3f}%"
                    + ("  ⭐ NEW BEST" if new_best else ""),
                    flush=True,
                )
                if infinite:
                    print(f"[LIVE_IMPROVE] ep={global_ep} phi={metrics['phi']:.4f} "
                          f"best_phi={best_phi:.4f}", flush=True)
            now = time.time()
            if now - last_hb >= 60:
                _write_heartbeat(args.metrics_dir or "logs", global_ep, phase["name"])
                last_hb = now

    # Numbered phases (order < 999), then the infinite live_improve phase.
    for phase in phases:
        if phase.get("order", 0) >= 999 or phase.get("max_episodes") == -1:
            continue
        print(f"[train] === PHASE {phase['name']} ===", flush=True)
        run_phase(phase, infinite=False)

    live = next((p for p in phases if p.get("max_episodes") == -1), None)
    if live is not None:
        print("[train] === LIVE_IMPROVE (infinite) ===", flush=True)
        run_phase(live, infinite=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default=None)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-phase", type=int, default=0)
    ap.add_argument("--force-fresh", action="store_true")
    return train(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
