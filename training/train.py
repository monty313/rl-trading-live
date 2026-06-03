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

────────────────────────────────────────────────────────────────────────────
SETTING THE DAILY TARGET & DAILY DRAWDOWN AT RUNTIME (no retraining of the RULES)
────────────────────────────────────────────────────────────────────────────
The FTMO daily profit target and the daily trailing-DD limit are RUNTIME config
inputs. There is exactly ONE obvious place to set them — the CLI flags (which
populate CFG["DAILY_TARGET_PCT"] / CFG["DAILY_MAX_DD_PCT"]):

  # 2.5% daily target, 1% daily max DD on a $10k account  (the defaults)
  python -m training.train --csv DATA.csv --checkpoint-dir CK --metrics-dir M \
      --manifest CK/manifest.json --resume \
      --target-pct 0.025 --max-dd-pct 0.01

  # Same thing expressed as an ABSOLUTE dollar target (equivalent to 2.5% on $10k):
  python -m training.train ... --daily-target-usd 250 --max-dd-pct 0.01

On startup you will see the AUTHORITATIVE banner (printed by build_pipeline, the
single place every entry point emits it — train / backtest / eval / live):

  [ftmo] daily target = 2.50% (=$250 on $10,000 account)  |  daily max DD = 1.00%

CHANGING THESE ON A RESUME WORKS: a checkpoint stores only network weights +
optimizer + {phase,episode,phi,pass_rate}. It does NOT persist target_pct /
max_dd_pct, so a resumed run enforces the CURRENT CLI/cfg values — pass new
--target-pct / --max-dd-pct on resume and the new rules apply immediately.

HONESTY NOTE — "rules are config-driven (instant)" vs "policy is learned":
  • RULE ENFORCEMENT (PASS/FAIL classification + the 1% DD halt) is recomputed at
    RUNTIME from these config values everywhere (env, daily_guard, reward shaper,
    backtest, eval). Change them and enforcement changes on the NEXT bar — no
    retraining needed for the rules.
  • The trained POLICY, however, was OPTIMIZED for the target/risk it trained on.
    Running live at a very different target/DD (e.g. 5%/2% on a policy trained at
    2.5%/1%) will correctly classify days under the new rules, but the agent's
    BEHAVIOUR may be sub-optimal until you retrain at the new values. Small tweaks
    are usually fine; large changes warrant a retrain/fine-tune.
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

# Free A100 tensor-core speedup for fp32 matmuls (learning_loop_fix.md FIX 4.2).
# Harmless on CPU/T4; silences the "set_float32_matmul_precision" warning.
try:
    torch.set_float32_matmul_precision("high")
except Exception:  # pragma: no cover
    pass

from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from core.pipeline import (build_pipeline, load_ohlcv_csv,  # noqa: E402
                           resolve_initial_equity)
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


def _aggregate_day(info: dict, closed: "torch.Tensor") -> dict:
    """
    Aggregate the FULL BATCH of episodes that closed the same calendar day on
    this step into one honest summary (Bug A fix). Because all B episodes share
    the global day boundary (env advances _step_i in lockstep), `closed` here is
    the calendar new-day mask — true for EVERY episode at once — so this line
    always covers the whole batch (never a 1/64 line).

    Returns mean/median day PnL ($), mean equity, mean trades, and the BINARY
    PASS / FAIL counts across the batch (ftmo_rules_fix.md RULE 2).

    Per-episode class (mirrors BatchedFTMOEnv.step — STRICTLY BINARY):
      PASS : final_or_halt_equity >= day_start + daily_increment (info["passed"]).
      FAIL : everything else, including a zero-trade day and a DD-breach day that
             ended under target. There is NO "OK" and NO "SKIP" anymore.
    """
    idx = closed.nonzero(as_tuple=True)[0]
    passed = info["passed"][idx].bool()
    # Binary: a day is PASS or FAIL — fail is simply the complement of pass. We
    # read info["failed"] (the env's binary complement) but fall back to ~passed
    # so the invariant holds even if a caller omits it.
    failed = info["failed"][idx].bool() if "failed" in info else (~passed)
    trades = info["trades_today"][idx].long()

    n = int(idx.numel())
    n_pass = int(passed.sum().item())
    n_fail = int(failed.sum().item())

    eq = info["equity"][idx].float()
    day_start = info.get("day_start_eq", info["equity"])[idx].float() \
        if "day_start_eq" in info else info["equity"][idx].float()
    day_pnl = eq - day_start
    return {
        "n": n, "pass": n_pass, "fail": n_fail,
        "mean_pnl": float(day_pnl.mean().item()) if n else 0.0,
        "median_pnl": float(day_pnl.median().item()) if n else 0.0,
        "mean_eq": float(eq.mean().item()) if n else 0.0,
        "mean_tr": float(trades.float().mean().item()) if n else 0.0,
        # Headline bubble: 🟢 only when the batch PASSED on balance (more passes
        # than fails AND at least one pass), else 🔴 — a glanceable left-edge
        # pass progression. (Per-episode is already strictly binary.)
        "day_passed": (n_pass > n_fail) and (n_pass > 0),
    }


def _print_daily_results(phase: dict, day_num: int, agg: dict, streak: int):
    """
    Print ONE aggregated, column-aligned daily line for the WHOLE batch. Column
    order (learning_loop_fix.md FIX 2), left-to-right, phase LAST:

        DAY <n>  <🟢/🔴>  PnL $<..>  equity <..>  streak <..>  trades <..>  phase <name>

    The colored bubble sits RIGHT AFTER the day number so the left edge scans as
    a green/red pass progression. The class block is now COMPACT BINARY pass/fail
    counts across the 64 episodes (e.g. "P:12 F:52") — no 🟡 (OK) / ⬜ (SKIP)
    indicators anymore (ftmo_rules_fix.md RULE 2). Phase stays LAST on the far
    right. Flushed for live Colab output.
    """
    bubble = "🟢" if agg["day_passed"] else "🔴"
    print(
        f"  DAY {day_num:>4}  {bubble}"
        f"   PnL ${agg['mean_pnl']:>+11,.2f}"
        f"   equity {agg['mean_eq']:>12,.2f}"
        f"   streak {streak:>3}"
        f"   trades {agg['mean_tr']:>5.1f}"
        f"   (P:{agg['pass']:>2} F:{agg['fail']:>2})"
        f"   phase {phase['name']}",
        flush=True,
    )


def _phi_metric(pass_rate: float, avg_ret: float, avg_dd: float,
                target_pct: float, max_dd_pct: float) -> float:
    """ALWAYS-COMPUTABLE best-checkpoint objective (learning_loop_fix.md FIX 1.3).

    A blend of pass-rate, normalized mean daily return, and DD-safety that is a
    meaningful real number even before the first pass (so best_phi updates off
    its -1e9 sentinel from episode 1):

        phi = 0.6 * pass_rate
            + 0.3 * clip(avg_ret / target, -1, 2)
            + 0.1 * (1 - clip(avg_dd / max_dd, 0, 1))     # DD safety: 1=safe,0=at-limit

    Higher is better; it rewards passing days, positive returns, and staying
    clear of the DD limit. Used both for the per-episode running phi and eval.
    """
    ret_n = max(-1.0, min(2.0, avg_ret / (target_pct + 1e-9)))
    dd_safety = 1.0 - max(0.0, min(1.0, avg_dd / (max_dd_pct + 1e-9)))
    return 0.6 * pass_rate + 0.3 * ret_n + 0.1 * dd_safety


def train(args) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = get_device()
    cfg = auto_tune_batch(dict(CFG), device)
    cfg["DATA_CSV_EURUSD"] = args.csv
    cfg["TRADE_LOG"] = os.path.join(args.metrics_dir or "logs", "daily_trade_log.csv")
    cfg["CHECKPOINT_DIR"] = args.checkpoint_dir   # default feature-cache location
    # ── ACCOUNT SIZE (learning_loop_fix.md FIX 3): --account-size overrides the
    # default $10,000. Targets/limits stay percent-based; reward is normalized.
    if getattr(args, "account_size", None):
        cfg["ACCOUNT_SIZE"] = float(args.account_size)
        cfg["INITIAL_EQUITY"] = float(args.account_size)
    # ── FTMO RULE INPUTS (ftmo_rules_fix.md RULE 5): CLI flags override CFG so the
    # user can dial target / DD per live account. NOTHING downstream hardcodes
    # 0.025 / 0.01 / 250 / 100 — env, reward, eval, and guard all read these.
    #
    # RUNTIME-OVERRIDE-ON-RESUME (the important guarantee): these flags are applied
    # to cfg BEFORE the pipeline / checkpoint load runs, and PPOAgent checkpoints
    # store ONLY network weights + optimizer + {phase,episode,phi,pass_rate} — they
    # do NOT persist target_pct / max_dd_pct. So a resumed run ALWAYS enforces the
    # CURRENT cfg/CLI values, never a stale value baked into the checkpoint. Change
    # --target-pct / --max-dd-pct on resume and the new rules take effect at once.
    if getattr(args, "target_pct", None) is not None:
        cfg["DAILY_TARGET_PCT"] = float(args.target_pct)
    if getattr(args, "max_dd_pct", None) is not None:
        cfg["DAILY_MAX_DD_PCT"] = float(args.max_dd_pct)
    # Optional ABSOLUTE-DOLLAR target: if given it OVERRIDES the percentage by
    # back-computing target_pct = usd / initial_equity (so the whole stack stays
    # percent-driven and account-size invariant — only the input form differs).
    _acct = resolve_initial_equity(cfg)
    if getattr(args, "daily_target_usd", None) is not None:
        usd = float(args.daily_target_usd)
        cfg["DAILY_TARGET_PCT"] = usd / (_acct + 1e-9)
        print(f"[train] --daily-target-usd ${usd:,.2f} overrides --target-pct "
              f"-> {cfg['DAILY_TARGET_PCT']*100:.4f}% on ${_acct:,.0f}", flush=True)
    print(f"[train] account size = ${_acct:,.0f}", flush=True)
    # ── RANDOMIZED-TARGET/DD TRAINING (target_aware_policy.md item 2) ─────────
    # --randomize-ftmo turns on per-episode sampling of target_pct/max_dd_pct so
    # ONE policy learns to CONDITION on the FTMO inputs (it OBSERVES them, item 1)
    # and GENERALIZES across target/risk. DEFAULT OFF. When ON, the checkpoint's
    # proportional-scaler baseline is stored as the MIDPOINT of the ranges (see
    # PPOAgent.save). With it OFF the fixed cfg target/DD is used and still appears
    # (constant) in the observation, so inference-time changes still shift behaviour.
    if getattr(args, "randomize_ftmo", False):
        cfg["RANDOMIZE_FTMO_INPUTS"] = True
        cfg["RANDOMIZE_FTMO_ACCOUNT"] = bool(getattr(args, "randomize_ftmo_account", False))
        tlo, thi = cfg.get("RANDOMIZE_TARGET_RANGE", [0.01, 0.05])
        dlo, dhi = cfg.get("RANDOMIZE_DD_RANGE", [0.005, 0.02])
        print(f"[train] --randomize-ftmo ON: per-episode target_pct in "
              f"[{tlo:.3f},{thi:.3f}], max_dd_pct in [{dlo:.4f},{dhi:.4f}]"
              f"{' + account_size' if cfg['RANDOMIZE_FTMO_ACCOUNT'] else ''}. "
              f"Trained baseline (scaler) = midpoint "
              f"({0.5*(tlo+thi)*100:.2f}% / {0.5*(dlo+dhi)*100:.3f}%).", flush=True)
    # NOTE: the authoritative active-rules banner ("[ftmo] daily target = ...")
    # is printed once by build_pipeline() below — the SINGLE place every entry
    # point (train/backtest/eval/live) emits it — so we don't duplicate it here.

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
            # Resume loads WEIGHTS ONLY. The FTMO rules (target_pct / max_dd_pct)
            # in force are the CURRENT cfg/CLI values printed in the [ftmo] banner
            # above — they are NOT restored from the checkpoint (which never stored
            # them). So changing --target-pct / --max-dd-pct on resume takes effect.
            print(f"[train] resuming WEIGHTS from {resume} "
                  f"(FTMO rules come from current CLI/cfg, not the checkpoint)",
                  flush=True)
            agent.load(str(resume), partial=True)   # partial best-effort transfer on shared layers
        else:
            print("[train] no checkpoint found — fresh start", flush=True)

    global_ep = 0
    last_hb = 0.0
    best_phi = -1e9
    # STREAK = consecutive passing DAYS across the batch (resets on a failed day).
    # Persists across episodes within a run — the user wants a running streak.
    pass_streak = 0
    target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
    max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))

    def run_phase(phase: dict, infinite: bool):
        nonlocal global_ep, last_hb, best_phi, pass_streak
        env.phase = phase
        max_eps = phase.get("max_episodes", cfg["MAX_EPISODES_PER_PHASE"])
        ep_in_phase = 0
        # ── HEARTBEAT (learning_loop_fix.md FIX 2): WALL-CLOCK time-based, default
        # every 300s (5 min), configurable via CFG["HEARTBEAT_SECS"]. A one-liner
        # (step, steps/s, elapsed, phase) — NOT every N steps. Keeps Colab visibly
        # alive without flooding the log.
        heartbeat_secs = float(cfg.get("HEARTBEAT_SECS", 300))
        bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
        global_day = 0                     # calendar day counter across the run
        last_heartbeat_t = time.time()
        while True:
            if not infinite and max_eps != -1 and ep_in_phase >= max_eps:
                break
            state = env.reset()
            shaper.global_ep = global_ep
            done = torch.zeros(env.B, dtype=torch.bool, device=device)
            steps = 0
            ep_total_reward = 0.0
            ep_gate_bars = 0
            # running per-episode pass/ret/dd tallies for the always-on phi metric
            ep_day_pass, ep_day_ret, ep_day_dd, ep_day_count = 0, 0.0, 0.0, 0
            max_steps = env.ep_bars
            rollout = int(cfg.get("ROLLOUT_STEPS", 2048))
            ep_t0 = time.time()
            # ── PPO on-policy rollout: collect transitions, update periodically ──
            while not done.all() and steps < max_steps:
                mask = env.current_direction_mask()
                if mask[0, 0].item() == 0.0:          # gate on for ep 0 this bar
                    ep_gate_bars += 1
                out = agent.select_actions(state, mask=mask)
                next_state, reward, done, info = env.step(out)
                agent.store(state, out, reward, done, mask)
                state = next_state
                steps += 1
                ep_total_reward += float(reward.mean().item())

                # ── WALL-CLOCK HEARTBEAT (one-liner, every ~heartbeat_secs) ──
                now = time.time()
                if now - last_heartbeat_t >= heartbeat_secs:
                    elapsed = now - ep_t0
                    sps = steps / max(elapsed, 1e-9)
                    print(
                        f"  ⏱  heartbeat  step {steps:>6}/{max_steps}"
                        f"  {sps:6.1f} steps/s  elapsed {elapsed:6.0f}s"
                        f"  phase {phase['name']}",
                        flush=True,
                    )
                    last_heartbeat_t = now

                # ── ONE HONEST AGGREGATED DAY LINE PER CALENDAR DAY (Bug A) ──
                # The calendar day rolls over for ALL episodes together every
                # bars_per_day steps, so we print exactly once then, aggregating
                # the FULL batch — never a 1/64 line. DD-halts close individual
                # episodes' days early for classification but don't desync the
                # calendar boundary used here.
                if steps % bars_per_day == 0:
                    closed = torch.ones(env.B, dtype=torch.bool, device=device)
                    agg = _aggregate_day(info, closed)
                    # streak: +1 on a batch-passing day, reset to 0 on a fail day.
                    if agg["day_passed"]:
                        pass_streak += 1
                    elif agg["fail"] > 0:
                        pass_streak = 0
                    global_day += 1
                    _print_daily_results(phase, global_day, agg, pass_streak)
                    # accumulate for the always-on phi metric (batch means)
                    ep_day_pass += agg["pass"]
                    ep_day_count += agg["n"]
                    ep_day_ret += float(info["daily_return"].mean().item())
                    ep_day_dd += float(info["daily_dd"].mean().item())

                if len(agent.buffer) >= rollout:
                    agent.update()        # PPO update, then buffer clears
            loss = agent.update()          # flush remaining rollout at episode end

            global_ep += 1
            ep_in_phase += 1

            # ── ALWAYS-COMPUTABLE running phi (best_phi updates from ep 1) ──
            n_days = max(ep_day_count, 1)
            pass_rate = ep_day_pass / n_days
            avg_ret = ep_day_ret / max(global_day, 1)
            avg_dd = ep_day_dd / max(global_day, 1)
            run_phi = _phi_metric(pass_rate, avg_ret, avg_dd, target_pct, max_dd_pct)
            new_best = run_phi > best_phi
            if new_best:
                best_phi = run_phi

            eq_now = float(env._equity.mean().item())
            ep_ret = (eq_now - env.initial_equity) / (env.initial_equity + 1e-9) * 100
            loss_str = f"  loss {loss:.4f}" if loss is not None else ""
            gate_pct = ep_gate_bars / max(steps, 1) * 100
            print(
                f"  {'─'*78}\n"
                f"  Episode {global_ep:>4}  [{phase['name']}]"
                f"  pass {pass_rate*100:4.1f}%  streak {pass_streak}"
                f"  equity {eq_now:>11,.2f} ({ep_ret:+.3f}%)"
                f"  reward {ep_total_reward:+.3f}"
                f"  gate {gate_pct:.1f}%{loss_str}"
                f"  phi {run_phi:+.4f}  best_phi {best_phi:+.4f}"
                + ("  ⭐" if new_best else ""),
                flush=True,
            )

            if global_ep % cfg["CHECKPOINT_EVERY"] == 0:
                ckpt_mgr.save(agent, phase["name"], global_ep, phi=best_phi,
                              pass_rate=pass_rate)
            if global_ep % cfg["EVAL_EVERY"] == 0:
                metrics = run_eval(env, agent, cfg, n_days=2)
                eval_new_best = metrics["phi"] > best_phi
                if eval_new_best:
                    best_phi = metrics["phi"]
                    ckpt_mgr.save(agent, phase["name"], global_ep,
                                  phi=best_phi, pass_rate=metrics["pass_rate"],
                                  name="best_eval.pt")
                print(
                    f"  {'═'*78}\n"
                    f"  🔍 EVAL  pass {metrics['pass_rate']*100:.1f}%"
                    f"  phi {metrics['phi']:+.4f}"
                    f"  avg_ret {metrics['avg_daily_return']*100:+.3f}%"
                    f"  avg_dd {metrics['avg_daily_dd']*100:.3f}%"
                    + ("  ⭐ NEW BEST" if eval_new_best else ""),
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


_KNOWN_ERRORS = [
    (
        "CUDAGraphs that has been overwritten",
        "ISSUE  : torch.compile CUDA Graph overwrote a rollout buffer tensor.\n"
        "RULE   : ActorCritic.forward() must .clone() every output tensor.\n"
        "FIX    : Run Cell 4 (git reset --hard) then re-run Cell 6.\n"
        "         If it persists, add USE_TORCH_COMPILE: false to CFG.",
    ),
    (
        "No such file or directory: 'training/train.py'",
        "ISSUE  : Working directory is not the repo root.\n"
        "RULE   : Cell 6 must run from /content/rl-trading-live.\n"
        "FIX    : Re-run Cell 4 to reset the repo, then re-run Cell 6.",
    ),
    (
        "No module named 'training'",
        "ISSUE  : Python cannot find the training package.\n"
        "RULE   : %cd /content/rl-trading-live must run before training starts.\n"
        "FIX    : Re-run Cell 4, then Cell 6 (it starts with %cd).",
    ),
    (
        "No such file or directory.*requirements.txt",
        "ISSUE  : pip cannot find requirements.txt — wrong working directory.\n"
        "RULE   : Cell 3 must run after Cell 4 has cloned the repo.\n"
        "FIX    : Run Cell 4 first, then re-run Cell 3, then Cell 6.",
    ),
    (
        "ResolutionImpossible",
        "ISSUE  : pip dependency conflict — likely numpy>=2.0 vs faiss-cpu.\n"
        "RULE   : requirements.txt pins numpy>=1.26,<2.0.\n"
        "FIX    : Re-run Cell 3 from the repo root (after Cell 4).",
    ),
    (
        "assert.*vram_gb > 30",
        "ISSUE  : GPU is not an A100 (or no GPU at all).\n"
        "RULE   : Training requires an A100 runtime (>30GB VRAM).\n"
        "FIX    : Runtime → Change runtime type → A100 GPU.",
    ),
    # ── MISSING CSV / UNMOUNTED DRIVE ────────────────────────────────────────
    # FUTURE-LLM NOTE: a FileNotFoundError on the CSV is ALMOST ALWAYS an
    # unmounted Google Drive in a fresh/restarted Colab session — NOT a code bug.
    # Drive unmounts on every runtime restart/timeout, so the path goes empty
    # even though the loader is fine. Do NOT rewrite the data loader. Re-mount
    # Drive (Cell 2, force_remount=True) and `ls` the data folder. The patterns
    # below catch our own friendly DataFileNotFoundError message ("PRIMARY DATA
    # FILE NOT FOUND") AND a raw FileNotFoundError that mentions a .csv path.
    (
        r"PRIMARY DATA FILE NOT FOUND|DataFileNotFoundError|"
        r"FileNotFoundError.*\.csv|No such file or directory.*\.csv",
        "ISSUE  : The training CSV could not be found — Drive is probably not\n"
        "         mounted (it UNMOUNTS on every Colab restart/timeout), or the\n"
        "         file path/name is wrong, or the file isn't in that folder.\n"
        "RULE   : The CSV must exist at the --csv path on a MOUNTED Drive.\n"
        "FIX    : (a) Re-run Cell 2 (MOUNT DRIVE). If it STILL fails, force it:\n"
        "             from google.colab import drive\n"
        "             drive.mount('/content/drive', force_remount=True)\n"
        "         (b) Verify the file exists:\n"
        "             !ls -la /content/drive/MyDrive/RL-Trading-Data/\n"
        "         (c) Confirm the filename EXACTLY matches the --csv argument.\n"
        "         (d) Then re-run Cell 6 (training).\n"
        "         See docs/COLAB_RUNBOOK.md for the full checklist.",
    ),
    (
        "FileNotFoundError.*eurusd_gpu",
        "ISSUE  : Checkpoint manager tried to load a deleted DQN checkpoint.\n"
        "RULE   : Manifest must only list files that exist on Drive.\n"
        "FIX    : Re-run Cell 4b to clean the manifest, then re-run Cell 6.",
    ),
    (
        "weights_only",
        "ISSUE  : torch.load failed loading a checkpoint (PyTorch version mismatch).\n"
        "RULE   : Checkpoints must be loadable with weights_only=False.\n"
        "FIX    : Run Cell 8 (crash recovery) to find the best valid checkpoint.",
    ),
]


def _diagnose(exc: Exception) -> str:
    """Match the exception message against known errors and return a fix."""
    import re
    msg = str(exc)
    tb = ""
    try:
        import traceback
        tb = traceback.format_exc()
    except Exception:
        pass
    full = msg + tb
    for pattern, advice in _KNOWN_ERRORS:
        if re.search(pattern, full, re.IGNORECASE):
            return advice
    return (
        "ISSUE  : Unexpected error (not in known-error list).\n"
        "RULE   : Check the full traceback above for the root cause.\n"
        "FIX    : Run Cell 4 (git reset --hard) to ensure latest code,\n"
        "         then re-run Cell 5 (inspect_system) to find the failure,\n"
        "         then re-run Cell 6.\n"
        "         See docs/COLAB_RUNBOOK.md for the full run order +\n"
        "         a troubleshooting table mapping each error to its fix."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default=None)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-phase", type=int, default=0)
    ap.add_argument("--force-fresh", action="store_true")
    ap.add_argument("--account-size", type=float, default=None,
                    help="Starting equity (10000/25000/50000/100000). "
                         "Default 10000. Rules stay percent-based.")
    # ── FTMO RULE INPUTS (ftmo_rules_fix.md RULE 5) — never hardcoded ──
    ap.add_argument("--target-pct", type=float, default=None,
                    help="Daily profit target as a fraction of INITIAL equity "
                         "(default 0.025 = 2.5%%). The fixed daily increment is "
                         "initial_equity * target_pct (e.g. $250 on a $10k acct).")
    ap.add_argument("--max-dd-pct", type=float, default=None,
                    help="Max trailing intraday drawdown as a fraction "
                         "(default 0.010 = 1%%). Breach halts trading for the day.")
    ap.add_argument("--daily-target-usd", type=float, default=None,
                    help="ABSOLUTE daily profit target in account currency. If "
                         "given, OVERRIDES --target-pct by setting target_pct = "
                         "usd / initial_equity (e.g. 250 on a $10k acct == 2.5%%).")
    # ── RANDOMIZED-TARGET/DD TRAINING (target_aware_policy.md item 2) ──
    ap.add_argument("--randomize-ftmo", action="store_true",
                    help="Domain-randomization: sample target_pct in "
                         "[0.01,0.05] and max_dd_pct in [0.005,0.02] PER EPISODE "
                         "(ranges configurable in CFG). Trains ONE policy that "
                         "GENERALIZES across target/risk so changing --target-pct/"
                         "--max-dd-pct at inference adapts behaviour. DEFAULT OFF.")
    ap.add_argument("--randomize-ftmo-account", action="store_true",
                    help="With --randomize-ftmo, ALSO sample account_size per "
                         "episode from ACCOUNT_SIZE_CHOICES (10k/25k/50k/100k).")
    try:
        return train(ap.parse_args())
    except Exception as exc:
        import traceback
        print("\n" + "═" * 70, flush=True)
        print("  ❌ TRAINING CRASHED", flush=True)
        print("═" * 70, flush=True)
        traceback.print_exc()
        print("\n" + "─" * 70, flush=True)
        print(_diagnose(exc), flush=True)
        print("─" * 70, flush=True)
        print("  📖 Full run order + troubleshooting: docs/COLAB_RUNBOOK.md", flush=True)
        print("─" * 70 + "\n", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
