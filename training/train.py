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
# Post-hoc interpretability + provenance helpers. The action-distribution logger
# is the ONLY one used inside the rollout (lightweight, no SHAP, toggleable); the
# results writer runs once at end-of-training / interrupt. SHAP is never imported
# here — it lives only in core/interpret/shap_explain.py for the post-hoc cell.
from core.interpret import action_logger  # noqa: E402
from core.interpret.results_writer import record_training_results  # noqa: E402


def _run_params_from_cfg(cfg: dict) -> dict:
    """Reconstruct the dashboard PARAMS dict from the EFFECTIVE training cfg so the
    results writer (PART 1) can match this run to its saved params snapshot by the
    SAME md5[:8] hash. Uses the dashboard's widget spec defaults and overlays the
    cfg values the run actually used (FTMO target/DD/account, reward weights, etc.).
    Kept here (not in the hot loop) so it costs nothing during training."""
    from core.interpret.dashboard_utils import build_params, widget_specs
    rw = cfg.get("REWARD", {}) or {}
    values = {}
    for key, spec in widget_specs().items():
        if spec.get("reward"):
            if key in rw:
                values[key] = rw[key]
        elif key in cfg:
            values[key] = cfg[key]
    return build_params(values)


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


class _CompileWatchdog:
    """Stdlib-ONLY liveness ticker for the torch.compile warmup block
    (compile_warmup_visibility fix, item 4).

    WHY: torch.compile(mode="default") compiles LAZILY on the FIRST forward pass,
    which runs INSIDE the rollout step loop and BLOCKS the main thread for
    ~10-15 min on an A100. The wall-clock heartbeat lives further down that same
    loop and therefore can NEVER fire during the block — the Colab cell looks
    frozen. This thread updates the screen WHILE the main thread is blocked.

    HARD CONSTRAINTS (so it is bulletproof and zero training-loop perf cost):
      • Pure stdlib only — time.sleep + print. It NEVER touches torch / CUDA
        (a second thread calling into CUDA during compile could deadlock).
      • daemon thread (won't keep the process alive) and the run loop is wrapped
        so it can NEVER raise into the interpreter.
      • Cleanly stopped + joined right after the first forward returns.
    Guarded by CFG['COMPILE_WATCHDOG_ENABLED'] (default ON); cadence
    CFG['COMPILE_WATCHDOG_SECS'] (default 30s)."""

    def __init__(self, interval_s: float, phase_name: str):
        import threading
        self._interval = max(0.01, float(interval_s))   # tiny floor: no busy-loop
        self._phase = phase_name
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="compile-watchdog")

    def _run(self):
        # Sleep in small slices so stop() is responsive without busy-waiting,
        # printing one "still compiling…" line every self._interval seconds.
        try:
            next_tick = self._interval
            # Wake in slices no longer than the cadence so stop() stays responsive
            # for both the default 30s and the tiny intervals used in tests.
            slice_s = min(0.25, self._interval)
            while not self._stop.is_set():
                self._stop.wait(slice_s)
                if self._stop.is_set():
                    break
                elapsed = time.time() - self._t0
                if elapsed >= next_tick:
                    print(f"  ⏳ still compiling… {elapsed:4.0f}s elapsed "
                          f"(torch.compile warmup, phase {self._phase}) — "
                          f"NORMAL, not a crash", flush=True)
                    next_tick += self._interval
        except Exception:                                  # pragma: no cover
            pass        # a watchdog must NEVER take training down

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:                                  # pragma: no cover
            pass


def _first_forward_warmup(forward_fn, *, compile_on: bool, watchdog_on: bool,
                          watchdog_secs: float, phase_name: str, steps: int,
                          max_steps: int, global_ep: int, metrics_dir: str):
    """Run the VERY FIRST forward pass of the run with full warmup visibility
    (compile_warmup_visibility fix items 1-3, plus the item-4 watchdog).

    `forward_fn` is a zero-arg callable that performs the first forward (this is
    where torch.compile actually compiles and BLOCKS for ~10-15 min on an A100).
    We, in order:
      1. ANNOUNCE the compile (only when compile_on) — flushed, before the block.
      2. Emit an IMMEDIATE step-0 heartbeat: printed AND written to
         heartbeat_training.txt on disk, BEFORE the (blocking) forward, so there
         is a provable liveness signal the instant the rollout loop starts.
      3. Start a stdlib-ONLY watchdog ticker (item 4) when compile_on+watchdog_on,
         run the timed forward, stop+join the watchdog, then print the
         "compile finished in Ns" marker.
    Returns the forward's output. Extracted from the rollout loop so the whole
    startup path is unit-testable with a mock slow forward (no GPU needed)."""
    if compile_on:
        print(
            "[train] 🛠  torch.compile warming up (mode=default) — first step "
            "compiles the model; expect ~10-15 min on A100 with NO further DAY "
            "output. This is NORMAL, not a crash. Disable via the dashboard "
            "COMPILE_MODEL / USE_TORCH_COMPILE toggle to skip warmup.",
            flush=True)
    # IMMEDIATE step-0 heartbeat (item 2): printed + on disk BEFORE the block.
    print(f"  ⏱  heartbeat  step {steps:>6}/{max_steps}"
          f"     0.0 steps/s  elapsed      0s"
          f"  phase {phase_name}  (loop entry)", flush=True)
    _write_heartbeat(metrics_dir, global_ep, phase_name)
    watchdog = (_CompileWatchdog(watchdog_secs, phase_name)
                if (compile_on and watchdog_on) else None)
    if watchdog is not None:
        watchdog.start()
    t0 = time.time()
    try:
        out = forward_fn()                       # ← torch.compile compiles HERE
    finally:
        if watchdog is not None:
            watchdog.stop()                      # always join, even on error
    elapsed = time.time() - t0
    if compile_on:
        print(f"[train] ✅ torch.compile finished in {elapsed:.0f}s — "
              f"training is now running fast.", flush=True)
    return out


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
    # ── COMPILE-WARMUP VISIBILITY (compile_warmup_visibility fix) ────────────
    # torch.compile compiles LAZILY on the FIRST forward pass (inside the rollout
    # loop), blocking the main thread ~10-15 min on an A100. `_warmup` tracks the
    # one-time announce/heartbeat/watchdog/finished-marker sequence so it fires
    # exactly once for the whole run (compile happens once, not per-episode).
    # compile_on: torch.compile is actually engaged (config ON *and* CUDA).
    compile_on = bool(cfg.get("USE_TORCH_COMPILE", True)) and device.type == "cuda"
    watchdog_on = bool(cfg.get("COMPILE_WATCHDOG_ENABLED", True))
    watchdog_secs = float(cfg.get("COMPILE_WATCHDOG_SECS", 30))
    _warmup = {"done": False}                    # flips True after first forward
    # STREAK = consecutive passing DAYS across the batch (resets on a failed day).
    # Persists across episodes within a run — the user wants a running streak.
    pass_streak = 0
    target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
    max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))

    # ── PART 1 RESULTS TRACKING (cheap scalars updated as we go) ─────────────
    # These feed record_training_results() at end-of-training AND on a graceful
    # interrupt (KeyboardInterrupt). They are plain Python numbers updated in the
    # loop so the writer always has the LATEST values — even a partial run yields
    # honest metrics. `run_metrics` is read by main()'s finally/except path.
    run_metrics = {"pass_rate": 0.0, "best_phi": best_phi, "episodes_trained": 0,
                   "final_equity": float(resolve_initial_equity(cfg)),
                   "best_streak": 0, "dd_efficiency_avg": 0.0,
                   "phase_reached": (phases[start_idx]["name"] if phases else "n/a"),
                   "_dd_eff_sum": 0.0, "_dd_eff_n": 0}
    cfg["_RUN_METRICS"] = run_metrics            # so main() can read partial state
    # ── PART 3 ACTION-DISTRIBUTION LOGGER (lightweight, toggleable) ──────────
    log_action_dist = bool(cfg.get("LOG_ACTION_DIST", True))
    action_dist_every = int(cfg.get("LOG_ACTION_DIST_EVERY", 100))
    action_dist_csv = os.path.join(args.metrics_dir or "logs",
                                   "action_distributions.csv")
    episode_summary = bool(cfg.get("LOG_ACTION_DIST_EPISODE_SUMMARY", True))
    _first_dist = {"d": None}                    # first logged dist (for shift line)
    # Section 11 — strategy-phase gate: advance to the next phase once an episode
    # reaches this many consecutive passing DAYS (config-driven; CLI override).
    phase_advance_streak = int(
        getattr(args, "phase_advance_streak", None)
        or cfg.get("PHASE_ADVANCE_STREAK", 10))

    def run_phase(phase: dict, infinite: bool):
        nonlocal global_ep, last_hb, best_phi, pass_streak
        env.phase = phase
        # ── TRAIN/VAL SEPARATION (audit P1) ──────────────────────────────────
        # Confine TRAINING episode starts to the leading [0, EVAL_SPLIT_FRAC)
        # slice so run_eval's held-out tail [EVAL_SPLIT_FRAC, 1.0) is genuinely
        # out-of-sample. Disable (full-range training) by setting EVAL_SPLIT_FRAC
        # >= 1.0 or USE_EVAL_SPLIT=False. The split is config-driven, not hardcoded.
        if bool(cfg.get("USE_EVAL_SPLIT", True)):
            split = float(cfg.get("EVAL_SPLIT_FRAC", 0.8))
            if 0.0 < split < 1.0:
                env.set_start_window(0.0, split)
        max_eps = phase.get("max_episodes", cfg["MAX_EPISODES_PER_PHASE"])
        ep_in_phase = 0
        # ── HEARTBEAT (learning_loop_fix.md FIX 2): WALL-CLOCK time-based, default
        # every 300s (5 min), configurable via CFG["HEARTBEAT_SECS"]. A one-liner
        # (step, steps/s, elapsed, phase) — NOT every N steps. Keeps Colab visibly
        # alive without flooding the log.
        heartbeat_secs = float(cfg.get("HEARTBEAT_SECS", 300))
        bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
        global_day = 0                     # calendar day counter across the run
        # IMMEDIATE FIRST HEARTBEAT (item 2): back-date the wall-clock anchor by a
        # full interval so the very first loop iteration is already "due" — the
        # first heartbeat fires at step 0 instead of 300s in. After that the
        # normal cadence resumes (we reset last_heartbeat_t = now once it fires).
        last_heartbeat_t = time.time() - heartbeat_secs
        while True:
            if not infinite and max_eps != -1 and ep_in_phase >= max_eps:
                break
            state = env.reset()
            shaper.global_ep = global_ep
            # Section 9: anneal the entropy coefficient for THIS episode (high
            # exploration early, stable by ENTROPY_ANNEAL_EPISODES).
            agent.anneal_entropy(global_ep)
            done = torch.zeros(env.B, dtype=torch.bool, device=device)
            steps = 0
            ep_total_reward = 0.0
            ep_gate_bars = 0
            # running per-episode pass/ret/dd tallies for the always-on phi metric
            ep_day_pass, ep_day_ret, ep_day_dd, ep_day_count = 0, 0.0, 0.0, 0
            # Section 4.2: best consecutive-pass streak + mean good-day DD efficiency
            # accrued THIS episode, feeding the composite episode bonus at ep end.
            ep_best_streak = 0
            ep_dd_eff_sum, ep_dd_eff_n = 0.0, 0
            phase_advanced = False
            max_steps = env.ep_bars
            rollout = int(cfg.get("ROLLOUT_STEPS", 2048))
            ep_t0 = time.time()
            # ── PPO on-policy rollout: collect transitions, update periodically ──
            while not done.all() and steps < max_steps:
                mask = env.current_direction_mask()
                if mask[0, 0].item() == 0.0:          # gate on for ep 0 this bar
                    ep_gate_bars += 1
                # ── COMPILE-WARMUP VISIBILITY (items 1-4) — runs ONCE, around the
                # very first forward pass of the whole run, where torch.compile
                # actually compiles and BLOCKS the main thread. Delegated to
                # _first_forward_warmup (announce + immediate heartbeat + watchdog
                # + finished marker). Subsequent forwards take the fast path.
                if not _warmup["done"]:
                    out = _first_forward_warmup(
                        lambda: agent.select_actions(state, mask=mask),
                        compile_on=compile_on, watchdog_on=watchdog_on,
                        watchdog_secs=watchdog_secs, phase_name=phase["name"],
                        steps=steps, max_steps=max_steps, global_ep=global_ep,
                        metrics_dir=args.metrics_dir or "logs")
                    last_heartbeat_t = time.time()       # resume normal cadence
                    _warmup["done"] = True
                else:
                    out = agent.select_actions(state, mask=mask)
                next_state, reward, done, info = env.step(out)
                agent.store(state, out, reward, done, mask)

                # ── PART 3: action-distribution logging (no extra forward pass —
                # we re-read the head logits from the SAME state under no_grad, then
                # write a compact CSV row every action_dist_every steps. Guarded so
                # any failure can't disturb training; toggle via LOG_ACTION_DIST). ──
                if log_action_dist and (steps % action_dist_every == 0):
                    try:
                        with torch.no_grad():
                            dl, el, lm, _v = agent._fwd(state)
                        dist = action_logger.action_distribution(
                            dl, el, torch.sigmoid(lm.squeeze(-1)))
                        if _first_dist["d"] is None:
                            _first_dist["d"] = dist
                        # market-state columns (batch means) the row correlates with
                        eq_mean = float(env._equity.mean().item())
                        dd_budget = float(info["daily_dd"].mean().item()) \
                            if "daily_dd" in info else 0.0
                        cest = time.strftime("%H:%M:%S")
                        action_logger.append_row(
                            action_dist_csv, dist, bar_index=steps, cest_time=cest,
                            equity=eq_mean, streak=pass_streak,
                            dd_budget_remaining=max(0.0, max_dd_pct - dd_budget))
                    except Exception:
                        pass

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
                    # Section 4.2 composite-bonus inputs: track the best per-episode
                    # streak (max over batch) and mean DD efficiency on positive days.
                    ep_best_streak = max(ep_best_streak,
                                         int(info["best_streak"].max().item()))
                    pos = info["daily_return"] > 0
                    if bool(pos.any()):
                        eff = (1.0 - (info["daily_dd"][pos]
                               / (max_dd_pct + 1e-9)).clamp(max=1.0))
                        ep_dd_eff_sum += float(eff.sum().item())
                        ep_dd_eff_n += int(pos.sum().item())
                    # Section 11 — strategy-phase gate: a single episode reaching the
                    # configured consecutive-pass streak advances to the next phase.
                    if (not infinite
                            and int(info["best_streak"].max().item()) >= phase_advance_streak):
                        phase_advanced = True
                        print(f"  ⏩ PHASE ADVANCE: best streak "
                              f"{int(info['best_streak'].max().item())} "
                              f">= {phase_advance_streak} in [{phase['name']}]",
                              flush=True)
                        break

                if len(agent.buffer) >= rollout:
                    # TRUNCATION BOOTSTRAP (PPO correctness): this env only ever sets
                    # done=True on a TIME-LIMIT/trade-cap (truncation) — never on a
                    # terminal "account blown" state — so cutting the rollout here
                    # mid-episode is itself a truncation. GAE must bootstrap the tail
                    # with V(s_T) of the NEXT state (now in `state`), NOT 0. Passing
                    # None made update() bootstrap 0, which biases every advantage at
                    # the rollout boundary toward "the world ends here" and is a silent
                    # value-function bug. We compute V(s_T) under no_grad.
                    last_value = agent.bootstrap_value(state, mask=mask)
                    agent.update(last_value=last_value)   # PPO update, then buffer clears
            # End-of-episode flush. If the loop exited because done.all() (the normal
            # time-limit), the final stored transition already carries done=True so
            # next_nonterminal=0 masks the bootstrap regardless of last_value. If it
            # exited on max_steps/phase-advance WITHOUT done, the tail is a truncation
            # and must bootstrap V(s_T); supplying it is correct in both cases.
            last_value = agent.bootstrap_value(state, mask=env.current_direction_mask())
            loss = agent.update(last_value=last_value)   # flush remaining rollout

            global_ep += 1
            ep_in_phase += 1

            # ── ALWAYS-COMPUTABLE running phi (best_phi updates from ep 1) ──
            n_days = max(ep_day_count, 1)
            pass_rate = ep_day_pass / n_days
            avg_ret = ep_day_ret / max(global_day, 1)
            avg_dd = ep_day_dd / max(global_day, 1)
            # ── Section 4.2/4.3 — composite EPISODE bonus + improvement multiplier.
            # The shaper turns best_streak (priority #1) + pass-rate + DD efficiency
            # into a scalar and amplifies it when the pass rate improved over the
            # previous episode. Folded into the running reward print so it is visible.
            ep_dd_eff = (ep_dd_eff_sum / ep_dd_eff_n) if ep_dd_eff_n else 0.0
            episode_bonus = shaper.episode_bonus(ep_best_streak, pass_rate, ep_dd_eff)
            ep_total_reward += episode_bonus
            run_phi = _phi_metric(pass_rate, avg_ret, avg_dd, target_pct, max_dd_pct)
            new_best = run_phi > best_phi
            if new_best:
                best_phi = run_phi

            eq_now = float(env._equity.mean().item())
            ep_ret = (eq_now - env.initial_equity) / (env.initial_equity + 1e-9) * 100

            # ── PART 1: refresh the live results metrics after each episode so a
            # graceful interrupt always has the latest honest numbers. ──
            run_metrics["episodes_trained"] = global_ep
            run_metrics["pass_rate"] = float(pass_rate)
            run_metrics["best_phi"] = float(best_phi)
            run_metrics["final_equity"] = float(eq_now)
            run_metrics["best_streak"] = max(int(run_metrics["best_streak"]),
                                             int(ep_best_streak), int(pass_streak))
            run_metrics["phase_reached"] = phase["name"]
            if ep_dd_eff_n:
                run_metrics["_dd_eff_sum"] += ep_dd_eff_sum
                run_metrics["_dd_eff_n"] += ep_dd_eff_n
                run_metrics["dd_efficiency_avg"] = (
                    run_metrics["_dd_eff_sum"] / max(run_metrics["_dd_eff_n"], 1))
            # ── PART 3: optional per-episode action-mix SHIFT line ──
            if (episode_summary and log_action_dist and _first_dist["d"] is not None):
                try:
                    with torch.no_grad():
                        dl, el, lm, _v = agent._fwd(state)
                    latest = action_logger.action_distribution(
                        dl, el, torch.sigmoid(lm.squeeze(-1)))
                    print(f"  📊 action mix  {action_logger.format_shift(_first_dist['d'], latest)}",
                          flush=True)
                except Exception:
                    pass

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

            # Section 11: leave this phase early once an episode hit the advance
            # streak (the infinite live_improve phase never advances).
            if phase_advanced:
                ckpt_mgr.save(agent, phase["name"], global_ep, phi=best_phi,
                              pass_rate=pass_rate)
                break

    # ── PART 1: write the training results block to the matching params snapshot
    # at end-of-training AND on a graceful interrupt (KeyboardInterrupt/SIGTERM).
    # The metrics dict is the live `run_metrics` (latest honest values, partial on
    # interrupt). Matching is by params_hash over the run's effective config so the
    # Compare panel can attach these results to the saved snapshot. Crash-safe. ──
    def _finalize_results(interrupted: bool):
        metrics = {
            "pass_rate": run_metrics["pass_rate"],
            "best_phi": run_metrics["best_phi"],
            "episodes_trained": run_metrics["episodes_trained"],
            "final_equity": run_metrics["final_equity"],
            "best_streak": run_metrics["best_streak"],
            "dd_efficiency_avg": run_metrics["dd_efficiency_avg"],
            "phase_reached": run_metrics["phase_reached"],
            "timestamp_completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "interrupted": interrupted,
        }
        snap_dir = getattr(args, "snapshot_dir", None) or cfg.get("SNAPSHOT_DIR")
        record_training_results(cfg, _run_params_from_cfg(cfg), metrics,
                                snapshot_dir=snap_dir)

    try:
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
    except KeyboardInterrupt:
        print("\n[train] ⏹  graceful interrupt — writing partial results.",
              flush=True)
        _finalize_results(interrupted=True)
        return 0
    _finalize_results(interrupted=False)
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
    # ── SECTION 11 — strategy-phase gate (never hardcoded) ──
    ap.add_argument("--phase-advance-streak", type=int, default=None,
                    help="Consecutive passing DAYS within a single episode that "
                         "advance to the next strategy phase (config/phases.yaml). "
                         "Default CFG['PHASE_ADVANCE_STREAK'] (=10).")
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
    ap.add_argument("--snapshot-dir", type=str, default=None,
                    help="Directory of params_snapshot_*.json files to match this "
                         "run against (PART 1 results writer). Defaults to "
                         "CFG['SNAPSHOT_DIR'] when omitted.")
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
