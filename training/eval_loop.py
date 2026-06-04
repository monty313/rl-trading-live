"""
training/eval_loop.py
────────────────────────────────────────────────────────────────────────────
Runs simulated trading days on the VALIDATION window and returns aggregate
metrics. Uses the SAME env + indicators + fills + guard as training (no
divergence). The validation window is the chronological TAIL of the dataset
(fraction [EVAL_SPLIT_FRAC, 1.0), default last 20%); the trainer confines its
own episode starts to the leading [0, EVAL_SPLIT_FRAC) slice, so this eval is
genuinely OUT-OF-SAMPLE — it never overlaps training start bars.

Note: an episode can be up to EPISODE_BARS long, so with a long EPISODE_BARS a
tail-started episode may still roll forward over earlier bars. The split bounds
the START distribution (where evaluation begins), which is what removes the
"eval on the exact same starts as training" leak; it is not a hard wall against
any bar overlap when episodes are long.

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
    # OUT-OF-SAMPLE: evaluate from the held-out chronological TAIL. Default split
    # 0.8 => validation starts sample from the last 20% of the series, disjoint
    # from the trainer's [0, split) window. Restored to full range after eval so
    # we never leave the env in a constrained state for any subsequent caller.
    split = float(cfg.get("EVAL_SPLIT_FRAC", 0.8))
    env.set_start_window(split, 1.0)
    try:
        env.reset()
        return _run_eval_body(env, agent, cfg, n_days, bars_per_day)
    finally:
        env.set_start_window(0.0, 1.0)


def _run_eval_body(env, agent, cfg: dict, n_days: int, bars_per_day: int) -> dict:
    daily_returns, daily_dds, daily_pass = [], [], []
    day_start_eq = float(env._equity[0].item())
    day_high = day_start_eq
    steps = n_days * bars_per_day
    # PASS uses the FIXED daily increment off INITIAL equity (ftmo_rules_fix.md
    # RULE 1): daily_target = day_start_eq + daily_increment, NOT a percentage of
    # the day's opening balance.
    daily_increment = float(env.daily_increment)
    # item-6 proportional scaler: at eval, scale exposure relative to the policy's
    # trained baseline so behaviour tracks the active target/DD (1.0 at baseline).
    lot_scale = agent.proportional_scale(env.target_pct, env.max_dd_pct)

    for step in range(steps):
        mask = env.current_direction_mask()
        state = env._get_state()
        out = agent.select_actions_eval(state, mask=mask, lot_scale=lot_scale)
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
