"""
training/estimate_pass_prob.py
────────────────────────────────────────────────────────────────────────────
PASS-PROBABILITY ESTIMATOR (target_aware_policy.md item 7).

After the user sets --target-pct / --max-dd-pct (and --account-size), produce an
EMPIRICAL probability that the agent PASSES a day under those inputs — estimated
by SIMULATION, not fabricated. It:

  1. Builds the pipeline (env + agent) at the GIVEN target/DD/account.
  2. Loads the latest/best PPO checkpoint (find_best_resume, or --checkpoint).
     A v1->v2 obs-schema mismatch is handled by PPOAgent.load() (input-layer
     reinit) — see target_aware_policy.md item 4.
  3. Runs N simulated trading days on the REAL EURUSD data, vectorized across the
     64-batch env (NOT a separate parallel sim — it reuses BatchedFTMOEnv), with a
     DETERMINISTIC-eval policy and the item-6 proportional scaler ACTIVE.
  4. Classifies each simulated day PASS/FAIL under the given target/DD and reports:
       pass probability = passes / N, a 95% Wilson confidence interval, the
       DD-breach rate, mean daily return, and mean trades/day.

Usage:
  python -m training.estimate_pass_prob --target-pct 0.05 --max-dd-pct 0.005 \
      --account-size 10000 --n-days 500 \
      --checkpoint-dir CK --manifest CK/manifest.json [--csv DATA.csv] \
      [--checkpoint path/to/best_eval.pt]

Sample output:
  [estimate] Pass probability at target 5.0% / DD 0.5% on $10k:
             38% (95% CI 34-42%, n=500 days) | DD-breach 21% | mean daily ret +1.1%

HONESTY NOTE: this is an EMPIRICAL estimate from historical EURUSD windows at the
CURRENT policy. It is NOT a guarantee — it assumes future data resembles the
sampled history, and FAR-out-of-distribution target/DD settings (e.g. far outside
the trained range) will have wider, less reliable estimates and may still need a
retrain. Observation-conditioning + --randomize-ftmo training improve
generalization but do not eliminate this caveat.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.settings import CFG, get_device  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402


def wilson_ci(passes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion (handles small n and
    extreme p far better than the normal approximation). Returns (lo, hi) in
    [0,1]. For n==0 returns (0.0, 1.0) — no information."""
    if n <= 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def estimate_pass_prob(env, agent, cfg: dict, n_days: int = 500) -> dict:
    """Run up to n_days simulated days vectorized across the B-batch env with a
    deterministic-eval policy + the item-6 proportional scaler, and return:
        {pass_prob, ci_lo, ci_hi, n_days, dd_breach_rate, mean_daily_return,
         mean_trades_per_day, lot_scale}

    A "day sample" is recorded every time an episode's day CLOSES (calendar
    boundary or DD-halt) — so B episodes contribute B day-samples per closed day,
    which is exactly how the batch vectorizes 500 days fast. The classification
    uses the env's own binary PASS/FAIL (info["passed"]) under the active
    target/DD, identical to training/backtest.
    """
    device = env.device
    bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
    # item-6 scaler for the CURRENT target/DD vs the loaded policy's baseline.
    lot_scale = agent.proportional_scale(env.target_pct, env.max_dd_pct)

    passes = 0
    breaches = 0
    n = 0
    ret_sum = 0.0
    trades_sum = 0.0

    env.reset()
    # Cap total steps generously; loop ends once we have n_days samples or data
    # runs out. With B episodes we gather B samples per closed day.
    max_steps = (n_days // max(env.B, 1) + 2) * bars_per_day + bars_per_day
    for _ in range(max_steps):
        if n >= n_days:
            break
        mask = env.current_direction_mask()
        state = env._get_state()
        out = agent.select_actions_eval(state, mask=mask, lot_scale=lot_scale)
        _s, _r, done, info = env.step(out)
        closed = info["day_closed"]
        if bool(closed.any()):
            idx = closed.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                if n >= n_days:
                    break
                n += 1
                passes += int(bool(info["passed"][i].item()))
                breaches += int(bool(info["dd_breached"][i].item()))
                ret_sum += float(info["daily_return"][i].item())
                trades_sum += float(info["trades_today"][i].item())
        if bool(done.all()):
            env.reset()

    if n == 0:    # degenerate (data too short) — record one partial day
        n = 1
    pass_prob = passes / n
    lo, hi = wilson_ci(passes, n)
    return {
        "pass_prob": pass_prob,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_days": n,
        "dd_breach_rate": breaches / n,
        "mean_daily_return": ret_sum / n,
        "mean_trades_per_day": trades_sum / n,
        "lot_scale": lot_scale,
    }


def _format_line(res: dict, target_pct: float, max_dd_pct: float,
                 account_size: float) -> str:
    acct_k = f"${account_size/1000:.0f}k" if account_size >= 1000 else f"${account_size:.0f}"
    return (f"[estimate] Pass probability at target {target_pct*100:.1f}% / "
            f"DD {max_dd_pct*100:.1f}% on {acct_k}: "
            f"{res['pass_prob']*100:.0f}% "
            f"(95% CI {res['ci_lo']*100:.0f}-{res['ci_hi']*100:.0f}%, "
            f"n={res['n_days']} days) | "
            f"DD-breach {res['dd_breach_rate']*100:.0f}% | "
            f"mean daily ret {res['mean_daily_return']*100:+.1f}%")


def run(args) -> int:
    device = get_device()
    cfg = dict(CFG)
    if args.csv:
        cfg["DATA_CSV_EURUSD"] = args.csv
    if args.account_size is not None:
        cfg["ACCOUNT_SIZE"] = float(args.account_size)
        cfg["INITIAL_EQUITY"] = float(args.account_size)
    if args.target_pct is not None:
        cfg["DAILY_TARGET_PCT"] = float(args.target_pct)
    if args.max_dd_pct is not None:
        cfg["DAILY_MAX_DD_PCT"] = float(args.max_dd_pct)
    # Estimation is an INFERENCE run — the proportional scaler is meant to be ON
    # (item 6) so behaviour tracks the new regime; respect an explicit cfg toggle.

    env, agent, *_ = build_pipeline(
        cfg, device,
        phase={"name": "estimate", "entry_conditions": {"buy": "any", "sell": "any"}})

    # Load the best/latest checkpoint (explicit --checkpoint wins).
    ckpt_path: Optional[Path] = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.checkpoint_dir and args.manifest:
        from training.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(args.checkpoint_dir, args.manifest)
        mgr.load_manifest()
        ckpt_path = mgr.find_best_resume()
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[estimate] loading policy from {ckpt_path}", flush=True)
        agent.load(str(ckpt_path), partial=True)
    else:
        print("[estimate] WARNING: no checkpoint found — estimating with an "
              "UNTRAINED policy (numbers are meaningless until you train).",
              flush=True)

    res = estimate_pass_prob(env, agent, cfg, n_days=int(args.n_days))
    print(_format_line(res, float(cfg["DAILY_TARGET_PCT"]),
                       float(cfg["DAILY_MAX_DD_PCT"]),
                       float(cfg.get("ACCOUNT_SIZE", 10_000.0))), flush=True)
    print("[estimate] HONESTY: empirical estimate from historical EURUSD windows "
          "at the CURRENT policy — not a guarantee; far-out-of-range target/DD is "
          "less reliable and may need a retrain.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Estimate the empirical pass probability under given FTMO "
                    "target/DD/account inputs (item 7 of target_aware_policy.md).")
    ap.add_argument("--target-pct", type=float, default=None,
                    help="Daily profit target as a fraction of initial equity "
                         "(e.g. 0.05 = 5%%).")
    ap.add_argument("--max-dd-pct", type=float, default=None,
                    help="Max trailing intraday drawdown as a fraction "
                         "(e.g. 0.005 = 0.5%%).")
    ap.add_argument("--account-size", type=float, default=None,
                    help="Starting equity (10000/25000/50000/100000).")
    ap.add_argument("--n-days", type=int, default=500,
                    help="Number of simulated day-samples (default 500).")
    ap.add_argument("--csv", default=None,
                    help="EURUSD M1 CSV path (else synthetic fixture / cfg default).")
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit checkpoint .pt to evaluate (overrides best).")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Checkpoint dir for find_best_resume (with --manifest).")
    ap.add_argument("--manifest", default=None,
                    help="Manifest path for find_best_resume.")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
