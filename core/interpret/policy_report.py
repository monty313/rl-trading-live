"""
core/interpret/policy_report.py
────────────────────────────────────────────────────────────────────────────
Plain-English POLICY SUMMARY REPORT (PART 5) — the collaborator deliverable.

Given a checkpoint + a representative batch of observations (and optionally a set
of per-day outcome dicts), run the policy DETERMINISTICALLY, collect statistics,
and emit a human-readable report with these sections:

  TRADING PERSONALITY  direction preference %, exit preference %, avg lot+std
  SESSION BEHAVIOR     most-active session + per-session breakdown %
  FEATURE RELIANCE     top-5 features per BUY/SELL/size (from saliency)
  RISK PROFILE         pass/fail mix, best/worst streak, avg/max DD usage
  CONSISTENCY          FAIL/OK/PASS/EXCEED/SURVIVAL rates via the REAL 5-tier
                       classifier (core/reward/shaper.classify_day)

Saved as BOTH a .txt (the readable report) and a structured .json (the same stats
machine-readable) in the metrics dir. The report leans on saliency.compute_saliency
for FEATURE RELIANCE so it always runs (no SHAP dependency on the fast path).

This module is POST-HOC only; it is never imported by the training loop.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch

from core.agent.action_space import DIRECTION_NAMES, EXIT_NAMES
from core.reward.shaper import classify_day, FAIL, OK, PASS, EXCEED
from core.interpret.saliency import compute_saliency
from core.interpret.dashboard_utils import obs_feature_names


def _policy_action_stats(agent, obs: torch.Tensor, max_lot: float) -> dict:
    """Run the policy deterministically over `obs` and tally direction/exit picks
    + lot mean/std. Uses select_actions_eval (argmax dir/exit, mean lot) so the
    report reflects the policy's ACTUAL deterministic behaviour."""
    device = next(agent.net.parameters()).device
    x = obs.detach().to(device).float()
    out = agent.select_actions_eval(x, mask=None)
    dirs = out["direction"].detach().cpu().numpy()
    exits = out["exit"].detach().cpu().numpy()
    lot_raw = out["lot_raw"].detach().cpu().numpy()
    # Map raw [0,1] lot onto [MIN_LOT, max_lot] for a human-meaningful size.
    from core.agent.action_space import MIN_LOT
    lots = MIN_LOT + lot_raw * (float(max_lot) - MIN_LOT)
    n = max(1, len(dirs))
    dir_counts = Counter(int(d) for d in dirs)
    exit_counts = Counter(int(e) for e in exits)
    return {
        "n": int(n),
        "direction_pref": {DIRECTION_NAMES[k]: dir_counts.get(k, 0) / n
                           for k in DIRECTION_NAMES},
        "exit_pref": {EXIT_NAMES[k]: exit_counts.get(k, 0) / n
                      for k in EXIT_NAMES},
        "avg_lot": float(np.mean(lots)),
        "std_lot": float(np.std(lots)),
    }


def _session_breakdown(obs: torch.Tensor, lkbk: int, n_ind: int) -> dict:
    """Tally how the observation batch distributes across the synthetic CEST
    sessions, using the session_code feature in the v3 obs block. Returns a
    {session_label: fraction} dict + the most-active session name."""
    state_dim = obs.shape[-1]
    names = obs_feature_names(lkbk, n_ind)
    if len(names) != state_dim or "session_code" not in names:
        return {"breakdown": {}, "most_active": "unknown"}
    idx = names.index("session_code")
    codes = obs[:, idx].detach().cpu().numpy()
    # session_code is code/N_SESSIONS in {0, .25, .5, .75, 1.0}; bucket by label.
    label_for = {0.0: "closed", 0.25: "asian", 0.5: "london", 0.75: "ny",
                 1.0: "london_ny_overlap"}
    buckets = Counter()
    for c in codes:
        key = min(label_for, key=lambda k: abs(k - float(c)))
        buckets[label_for[key]] += 1
    total = max(1, len(codes))
    breakdown = {k: v / total for k, v in buckets.items()}
    most = max(breakdown, key=breakdown.get) if breakdown else "unknown"
    return {"breakdown": breakdown, "most_active": most}


def _consistency_from_days(daily_log: Optional[List[dict]], cfg: dict) -> dict:
    """Classify each day in daily_log into the REAL 5 tiers (classify_day) and
    return tier RATES + streak stats. daily_log entries are
    {end_equity, day_start_equity, daily_increment, dd_breached, traded}. When no
    daily_log is supplied we return zeros (the report still renders)."""
    tiers = {FAIL: 0, OK: 0, PASS: 0, EXCEED: 0, "SURVIVAL": 0}
    best_streak = worst_streak = cur = 0
    dd_usages: List[float] = []
    if daily_log:
        for d in daily_log:
            tier = classify_day(
                float(d["end_equity"]), float(d["day_start_equity"]),
                float(d["daily_increment"]), bool(d.get("dd_breached", False)),
                bool(d.get("traded", True)))
            tiers[tier] += 1
            if d.get("traded", True) and not d.get("dd_breached", False):
                tiers["SURVIVAL"] += 1
            passed = tier in (PASS, EXCEED)
            if passed:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            best_streak = max(best_streak, cur)
            worst_streak = min(worst_streak, cur)
            if "dd_used_pct" in d:
                dd_usages.append(float(d["dd_used_pct"]))
    n = max(1, sum(tiers[t] for t in (FAIL, OK, PASS, EXCEED)))
    rates = {t: tiers[t] / n for t in (FAIL, OK, PASS, EXCEED)}
    rates["SURVIVAL"] = tiers["SURVIVAL"] / n
    return {
        "tier_rates": rates,
        "best_streak": int(best_streak),
        "worst_streak": int(worst_streak),
        "avg_dd_usage": float(np.mean(dd_usages)) if dd_usages else 0.0,
        "max_dd_usage": float(np.max(dd_usages)) if dd_usages else 0.0,
    }


def build_report(checkpoint_path: str, obs: torch.Tensor, cfg: dict,
                 daily_log: Optional[List[dict]] = None,
                 device: Optional[torch.device] = None,
                 indicator_columns: Optional[List[str]] = None) -> Dict[str, object]:
    """Collect every section's stats into one structured dict (the .json payload).
    Pure data — render_report_text() turns it into the readable .txt."""
    from core.agent.ppo import PPOAgent
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = int(obs.shape[-1])
    cfg = dict(cfg)
    cfg["STATE_DIM"] = state_dim
    agent = PPOAgent(state_dim, cfg, device)
    agent.load(checkpoint_path, partial=True)

    lkbk = int(cfg.get("LOOKBACK", 20))
    n_ind = max(1, (state_dim - 20) // max(lkbk, 1))
    max_lot = float(cfg.get("MAX_LOT", 2.0))

    personality = _policy_action_stats(agent, obs, max_lot)
    session = _session_breakdown(obs, lkbk, n_ind)
    names = obs_feature_names(lkbk, n_ind, indicator_columns)
    sal = compute_saliency(agent.net, obs,
                           feature_names=names if len(names) == state_dim else None,
                           top_k=5)
    feature_reliance = {head: sal[head]["ranking"][:5] for head in sal}
    consistency = _consistency_from_days(daily_log, cfg)

    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_observations": int(obs.shape[0]),
        "trading_personality": personality,
        "session_behavior": session,
        "feature_reliance": feature_reliance,
        "consistency": consistency,
    }


def render_report_text(report: Dict[str, object]) -> str:
    """Render the structured report dict as the plain-English .txt deliverable."""
    p = report["trading_personality"]
    s = report["session_behavior"]
    fr = report["feature_reliance"]
    c = report["consistency"]
    L: List[str] = []
    L.append("═" * 70)
    L.append("  POLICY SUMMARY REPORT")
    L.append(f"  checkpoint: {report['checkpoint']}    obs: {report['n_observations']}")
    L.append(f"  generated:  {report['timestamp']}")
    L.append("═" * 70)

    L.append("\n── TRADING PERSONALITY ─────────────────────────────────────────")
    dp = ", ".join(f"{k} {v*100:.1f}%" for k, v in p["direction_pref"].items())
    ep = ", ".join(f"{k} {v*100:.1f}%" for k, v in p["exit_pref"].items())
    L.append(f"  Direction preference : {dp}")
    L.append(f"  Exit preference      : {ep}")
    L.append(f"  Avg lot size         : {p['avg_lot']:.3f}  (std {p['std_lot']:.3f})")

    L.append("\n── SESSION BEHAVIOR ────────────────────────────────────────────")
    L.append(f"  Most active session  : {s['most_active']}")
    for name, frac in sorted(s["breakdown"].items(), key=lambda kv: -kv[1]):
        L.append(f"    {name:<20} {frac*100:5.1f}%")

    L.append("\n── FEATURE RELIANCE (top-5 by |saliency| per head) ─────────────")
    head_label = {"direction": "BUY/SELL/FLAT", "exit": "EXIT", "lot": "LOT SIZE"}
    for head, ranking in fr.items():
        L.append(f"  {head_label.get(head, head)}:")
        for fn, val in ranking:
            L.append(f"    {fn:<24} {val:.4f}")

    L.append("\n── RISK PROFILE / CONSISTENCY (real 5-tier classifier) ─────────")
    tr = c["tier_rates"]
    L.append(f"  FAIL {tr.get('FAIL',0)*100:.1f}%   OK {tr.get('OK',0)*100:.1f}%"
             f"   PASS {tr.get('PASS',0)*100:.1f}%   EXCEED {tr.get('EXCEED',0)*100:.1f}%"
             f"   SURVIVAL {tr.get('SURVIVAL',0)*100:.1f}%")
    L.append(f"  Best streak {c['best_streak']}   worst streak {c['worst_streak']}")
    L.append(f"  Avg DD usage {c['avg_dd_usage']*100:.2f}%   "
             f"max DD usage {c['max_dd_usage']*100:.2f}%")
    L.append("═" * 70)
    return "\n".join(L)


def write_report(report: Dict[str, object], metrics_dir: str,
                 prefix: str = "policy_report") -> Dict[str, str]:
    """Write the report .txt + .json to metrics_dir; return {'txt':..., 'json':...}."""
    os.makedirs(metrics_dir, exist_ok=True)
    txt_path = os.path.join(metrics_dir, f"{prefix}.txt")
    json_path = os.path.join(metrics_dir, f"{prefix}.json")
    with open(txt_path, "w") as f:
        f.write(render_report_text(report))
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return {"txt": txt_path, "json": json_path}


def generate_policy_report(checkpoint_path: str, obs: torch.Tensor, cfg: dict,
                           metrics_dir: str, daily_log: Optional[List[dict]] = None,
                           device: Optional[torch.device] = None,
                           indicator_columns: Optional[List[str]] = None,
                           ) -> Dict[str, object]:
    """One-call entry point used by the Colab cell: build + render + write the
    report. Returns {"report": <dict>, "text": <str>, "paths": {txt, json}}."""
    report = build_report(checkpoint_path, obs, cfg, daily_log=daily_log,
                          device=device, indicator_columns=indicator_columns)
    text = render_report_text(report)
    paths = write_report(report, metrics_dir)
    return {"report": report, "text": text, "paths": paths}
