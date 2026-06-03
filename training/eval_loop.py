"""
training/eval_loop.py
────────────────────────────────────────────────────────────────────────────
Runs simulated trading days on the validation window (last 20% of the data,
never used for training) and returns aggregate metrics. Uses the SAME env +
indicators + fills + guard as training (no divergence).

run_eval(env, agent, cfg, n_days=10) -> {
    pass_rate, phi, avg_daily_return, avg_daily_dd
}
"""
from __future__ import annotations

import numpy as np
import torch



def run_eval(env, agent, cfg: dict, n_days: int = 10) -> dict:
    """
    Roll the env forward deterministically (epsilon forced low) for up to
    n_days * bars_per_day steps, collecting per-day PASS/return/DD, and compute Φ.
    """
    device = env.device
    bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
    env.reset()
    daily_returns, daily_dds, daily_pass = [], [], []
    day_start_eq = float(env._equity[0].item())
    day_high = day_start_eq
    steps = n_days * bars_per_day
    # PASS uses the FIXED daily increment off INITIAL equity (ftmo_rules_fix.md
    # RULE 1): daily_target = day_start_eq + daily_increment, NOT a percentage of
    # the day's opening balance.
    daily_increment = float(env.daily_increment)

    for step in range(steps):
        mask = env.current_direction_mask()
        state = env._get_state()
        out = agent.select_actions(state, mask=mask)
        _s, _r, done, info = env.step(out)
        eq = float(info["equity"][0].item())
        day_high = max(day_high, eq)
        if (step + 1) % bars_per_day == 0:
            ret = (eq - day_start_eq) / (day_start_eq + 1e-9)
            dd = (day_high - eq) / (day_high + 1e-9)
            daily_returns.append(ret)
            daily_dds.append(dd)
            daily_pass.append(1.0 if eq >= day_start_eq + daily_increment else 0.0)
            day_start_eq, day_high = eq, eq
        if done.all():
            break

    if not daily_returns:   # episode ended before a full day; record one partial day
        eq = float(env._equity[0].item())
        daily_returns = [(eq - day_start_eq) / (day_start_eq + 1e-9)]
        daily_dds = [(day_high - eq) / (day_high + 1e-9)]
        daily_pass = [1.0 if eq >= day_start_eq + daily_increment else 0.0]

    from core.reward.shaper import EpisodeRewardShaper
    pass_rate = float(np.mean(daily_pass))
    avg_ret = float(np.mean(daily_returns))
    avg_dd = float(np.mean(daily_dds))
    shaper = EpisodeRewardShaper(cfg)
    phi = shaper._phi(pass_rate, avg_ret, avg_dd)

    return {
        "pass_rate": pass_rate,
        "phi": float(phi),
        "avg_daily_return": avg_ret,
        "avg_daily_dd": avg_dd,
    }
