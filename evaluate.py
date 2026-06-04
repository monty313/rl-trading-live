"""
evaluate.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 8 — HONEST out-of-sample holdout evaluation.

Runs a TRAINED checkpoint deterministically over a holdout CSV the model never
trained on, using the SAME env + indicators + fills + commission + guard as
training (zero divergence — it builds through core.pipeline like every other
entry point). It reports the FULL, honest picture and INCLUDES bad days
(fail / zero-trade / DD-halt) in every aggregate — nothing is silently dropped.

Usage (matches the repo's train.py conventions):

    python evaluate.py --checkpoint gpu/best_eval.pt --data holdout_eurusd.csv \\
        --out-dir eval_out [--seed 0] [--train-pass-rate 0.42] \\
        [--account-size 10000] [--target-pct 0.025] [--max-dd-pct 0.01]

Outputs (both carry the checkpoint SHA-256 so a report is traceable to weights):
    {out_dir}/eval_{stem}.json   — full metrics + per-day rows + config
    {out_dir}/eval_{stem}.csv    — one row per evaluated day

OVERFITTING is made visible: pass --train-pass-rate (the training/eval pass rate
the checkpoint reported) and the JSON includes the train-vs-holdout gap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from core.settings import CFG, auto_tune_batch
from core.pipeline import build_pipeline, load_ohlcv_csv
from core.agent.action_space import DIRECTION_DIM, EXIT_DIM


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def evaluate(checkpoint: str, data_csv: str, cfg: dict, device: torch.device,
             out_dir: str, seed=None, train_pass_rate=None,
             phase_name: str = "live_improve") -> dict:
    """Deterministic holdout replay. Returns the metrics dict (also written to
    disk as JSON + CSV). One B=1 episode rolls the ENTIRE holdout series so every
    calendar day in the file is scored (including zero-trade and halted days)."""
    if seed is not None:
        from core.seeding import set_global_seed
        set_global_seed(int(seed), deterministic=True)

    # Build through the SAME pipeline as training so indicators/fills/commission
    # match bit-for-bit. B=1 and EPISODE_BARS = full length => one pass over the
    # whole holdout file (no train-window confinement; this CSV is the holdout).
    data = load_ohlcv_csv(data_csv)
    cfg = dict(cfg)
    cfg["FEATURES"] = data
    cfg["BATCH_SIZE_ENV"] = 1
    cfg["EPISODE_BARS"] = len(data) + 10        # cover the entire series in one episode
    cfg["USE_TORCH_COMPILE"] = False
    phase = {"name": phase_name, "mask": None, "mask_type": "none"}
    env, agent, _sizer, _guard, _gate = build_pipeline(cfg, device, phase=phase)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    ckpt_meta = agent.load(checkpoint)
    ckpt_hash = _sha256(checkpoint)

    bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
    # item-6 proportional scaler vs the policy's trained baseline (1.0 at baseline).
    lot_scale = agent.proportional_scale(env.target_pct, env.max_dd_pct)

    state = env.reset()
    done = torch.zeros(env.B, dtype=torch.bool)

    # ── per-day + per-trade honest accumulators ──────────────────────────────
    day_rows = []                       # one dict per CLOSED day
    invalid_actions = 0
    lot_sum, lot_n = 0.0, 0
    trade_count = 0
    trade_wins, trade_losses = 0, 0
    gross_profit, gross_loss = 0.0, 0.0   # trade-level, for profit factor
    commission_paid = 0.0

    prev_balance = float(env._balance[0].item())
    prev_position = float(env._position[0].item())

    steps = len(data) - 2
    for _ in range(max(1, steps)):
        mask = env.current_direction_mask()
        out = agent.select_actions_eval(state, mask=mask, lot_scale=lot_scale)
        # honest invalid-action accounting (production select_actions_eval should
        # never emit these; count any that slip through instead of hiding them).
        d = out["direction"]; e = out["exit"]
        invalid_actions += int(((d < 0) | (d >= DIRECTION_DIM)).sum().item())
        invalid_actions += int(((e < 0) | (e >= EXIT_DIM)).sum().item())
        if (d != 0).any():
            lot_sum += float(out["lot_raw"][d != 0].sum().item())
            lot_n += int((d != 0).sum().item())

        _s, _r, done, info = env.step(out)

        # trade-level realization: a position that shrank/closed since last bar
        # realized PnL into the balance. Net balance delta minus the (already
        # deducted) commission is the trade's gross result; sign -> win/loss.
        cur_balance = float(env._balance[0].item())
        cur_position = float(env._position[0].item())
        closed_or_reduced = abs(cur_position) < abs(prev_position) - 1e-12
        if closed_or_reduced:
            realized = cur_balance - prev_balance
            trade_count += 1
            if realized >= 0:
                trade_wins += 1
                gross_profit += realized
            else:
                trade_losses += 1
                gross_loss += -realized
        prev_balance, prev_position = cur_balance, cur_position

        if bool(info["day_closed"][0].item()):
            day_rows.append({
                "day_idx": int(info["day_idx"][0].item()),
                "equity": float(info["equity"][0].item()),
                "day_start_eq": float(info["day_start_eq"][0].item()),
                "daily_return": float(info["daily_return"][0].item()),
                "daily_dd": float(info["daily_dd"][0].item()),
                "passed": bool(info["passed"][0].item()),
                "tier_fail": bool(info["tier_fail"][0].item()),
                "tier_ok": bool(info["tier_ok"][0].item()),
                "tier_pass": bool(info["tier_pass"][0].item()),
                "tier_exceed": bool(info["tier_exceed"][0].item()),
                "dd_breached": bool(info["dd_breached"][0].item()),
                "day_halted": bool(info["day_halted"][0].item()),
                "trades_today": int(info["trades_today"][0].item()),
            })
        if done.all():
            break

    # ── HONEST AGGREGATES (include EVERY day: fail, zero-trade, halted) ───────
    n_days = len(day_rows)
    passes = sum(1 for r in day_rows if r["passed"])
    fails = n_days - passes
    zero_trade_days = sum(1 for r in day_rows if r["trades_today"] == 0)
    halted_days = sum(1 for r in day_rows if r["day_halted"])
    breached_days = sum(1 for r in day_rows if r["dd_breached"])
    rets = [r["daily_return"] for r in day_rows]
    dds = [r["daily_dd"] for r in day_rows]
    win_days = sum(1 for r in day_rows if r["daily_return"] > 0)
    loss_days = sum(1 for r in day_rows if r["daily_return"] < 0)

    final_equity = float(env._equity[0].item())
    initial_equity = float(env.initial_equity)
    net_pnl = final_equity - initial_equity
    gross_pnl = net_pnl + commission_paid   # commission already netted in equity

    metrics = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "data_csv": os.path.abspath(data_csv),
        "bars": int(len(data)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "phase_name": phase_name,
        "lot_scale_applied": float(lot_scale),
        "trained_target_pct": float(agent.trained_target_pct),
        "trained_max_dd_pct": float(agent.trained_max_dd_pct),
        "eval_target_pct": float(env.target_pct),
        "eval_max_dd_pct": float(env.max_dd_pct),
        # ── headline honest metrics ──
        "days_evaluated": n_days,
        "pass_rate": _safe_div(passes, n_days),
        "fail_rate": _safe_div(fails, n_days),
        "passes": passes, "fails": fails,
        "zero_trade_days": zero_trade_days,      # each is a FAIL (RULE 2)
        "halted_days": halted_days,
        "breached_days": breached_days,
        "avg_daily_return": float(np.mean(rets)) if rets else 0.0,
        "max_daily_dd": float(np.max(dds)) if dds else 0.0,
        "avg_daily_dd": float(np.mean(dds)) if dds else 0.0,
        "win_days": win_days, "loss_days": loss_days,
        "day_win_rate": _safe_div(win_days, n_days),
        # ── trade-level ──
        "trade_count": trade_count,
        "trade_wins": trade_wins, "trade_losses": trade_losses,
        "trade_win_rate": _safe_div(trade_wins, trade_count),
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": _safe_div(gross_profit, gross_loss),
        "avg_lot": _safe_div(lot_sum, lot_n),
        "invalid_action_count": invalid_actions,
        # ── PnL ──
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
    }

    # ── OVERFITTING VISIBILITY: train vs holdout pass-rate gap ──
    if train_pass_rate is not None:
        metrics["train_pass_rate"] = float(train_pass_rate)
        metrics["holdout_pass_rate"] = metrics["pass_rate"]
        metrics["overfit_gap"] = float(train_pass_rate) - metrics["pass_rate"]

    # ── write JSON + CSV (both tagged with the checkpoint hash) ──
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(data_csv))[0]
    json_path = os.path.join(out_dir, f"eval_{stem}.json")
    csv_path = os.path.join(out_dir, f"eval_{stem}.csv")
    with open(json_path, "w") as f:
        json.dump({"metrics": metrics, "days": day_rows}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        if day_rows:
            w = csv.DictWriter(f, fieldnames=list(day_rows[0].keys()))
            w.writeheader()
            w.writerows(day_rows)
        else:
            f.write("# no full days evaluated\n")
    metrics["_json_path"] = json_path
    metrics["_csv_path"] = csv_path

    _print_report(metrics)
    return metrics


def _print_report(m: dict):
    print("\n" + "═" * 72, flush=True)
    print("  HONEST HOLDOUT EVALUATION", flush=True)
    print("═" * 72, flush=True)
    print(f"  checkpoint : {m['checkpoint']}", flush=True)
    print(f"  sha256     : {m['checkpoint_sha256'][:16]}…", flush=True)
    print(f"  data       : {m['data_csv']}  ({m['bars']:,} bars)", flush=True)
    print(f"  days       : {m['days_evaluated']}  "
          f"(zero-trade {m['zero_trade_days']}, halted {m['halted_days']}, "
          f"breached {m['breached_days']})", flush=True)
    print(f"  PASS rate  : {m['pass_rate']*100:.1f}%  "
          f"(FAIL {m['fail_rate']*100:.1f}%)", flush=True)
    print(f"  avg ret/dd : {m['avg_daily_return']*100:+.3f}% / "
          f"{m['avg_daily_dd']*100:.3f}%  (max DD {m['max_daily_dd']*100:.3f}%)",
          flush=True)
    print(f"  trades     : {m['trade_count']}  win {m['trade_win_rate']*100:.1f}%  "
          f"PF {m['profit_factor']:.2f}  avg lot {m['avg_lot']:.3f}", flush=True)
    print(f"  PnL        : net ${m['net_pnl']:+,.2f}  "
          f"(gross ${m['gross_pnl']:+,.2f})", flush=True)
    print(f"  invalid    : {m['invalid_action_count']} actions", flush=True)
    if "overfit_gap" in m:
        print(f"  OVERFIT    : train {m['train_pass_rate']*100:.1f}% vs holdout "
              f"{m['holdout_pass_rate']*100:.1f}%  gap "
              f"{m['overfit_gap']*100:+.1f}pp", flush=True)
    print(f"  wrote      : {m['_json_path']}", flush=True)
    print(f"               {m['_csv_path']}", flush=True)
    print("═" * 72 + "\n", flush=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(description="Honest holdout evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="holdout OHLCV CSV (out-of-sample)")
    ap.add_argument("--out-dir", default="eval_out")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-pass-rate", type=float, default=None,
                    help="training/eval pass rate for the overfitting gap line")
    ap.add_argument("--phase-name", default="live_improve",
                    help="curriculum phase whose lot window to size with "
                         "(default live_improve = full range, matching deployment)")
    ap.add_argument("--account-size", type=float, default=None)
    ap.add_argument("--target-pct", type=float, default=None)
    ap.add_argument("--max-dd-pct", type=float, default=None)
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.account_size:
        cfg["ACCOUNT_SIZE"] = float(args.account_size)
        cfg["INITIAL_EQUITY"] = float(args.account_size)
    if args.target_pct is not None:
        cfg["DAILY_TARGET_PCT"] = float(args.target_pct)
    if args.max_dd_pct is not None:
        cfg["DAILY_MAX_DD_PCT"] = float(args.max_dd_pct)

    device = get_device()
    evaluate(args.checkpoint, args.data, cfg, device, args.out_dir,
             seed=args.seed, train_pass_rate=args.train_pass_rate,
             phase_name=args.phase_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
