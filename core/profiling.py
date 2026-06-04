"""
core/profiling.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 11 — HONEST profiling. Measures where wall-clock time actually goes
across a handful of training-shaped steps and reports a bottleneck diagnosis
that does NOT lie: if the run is CPU-bound (env stepping / indicator math on the
host), it says so rather than chasing an 80%-GPU number that is unreachable when
the GPU sits idle waiting for the CPU.

Works on GPU or CPU (the notebook + scripts/profile_training.py both call this),
so the same numbers appear everywhere.

profile_episodes(agent, env, device, n_steps) times, per step:
  • policy forward+sample  (GPU work, when on CUDA)
  • env.step               (the vectorized env — usually the host bottleneck)
  • agent.update           (optional; the PPO optimizer step)
and returns a dict of totals + per-section means + a HONEST verdict string.
"""
from __future__ import annotations

import time
from typing import Dict

import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_forward(agent, env, device, batch_size: int, n_iters: int = 50) -> Dict:
    """Time only the policy forward pass on random observations (the GPU-bound
    part). Useful as a quick 'is the net even the bottleneck' probe."""
    obs = torch.randn(batch_size, env.state_dim, device=device)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            agent._fwd(obs)
    _sync(device)
    dt = time.perf_counter() - t0
    return {"section": "policy_forward", "iters": n_iters,
            "total_s": dt, "per_iter_ms": dt / n_iters * 1e3,
            "device": device.type}


def profile_episodes(agent, env, device, n_steps: int = 200,
                     do_update: bool = False) -> Dict:
    """Profile a realistic rollout: per-step policy inference + env.step, summed
    over n_steps. Returns wall-clock totals per section and a HONEST verdict on
    whether the loop is GPU- or CPU-bound."""
    state = env.reset()
    fwd_s = 0.0
    step_s = 0.0
    _sync(device)
    wall0 = time.perf_counter()
    for _ in range(n_steps):
        _sync(device); t = time.perf_counter()
        out = agent.select_actions(state, mask=env.current_direction_mask()
                                   if hasattr(env, "current_direction_mask") else None)
        _sync(device); fwd_s += time.perf_counter() - t

        t = time.perf_counter()
        state, _r, done, _info = env.step(out)
        _sync(device); step_s += time.perf_counter() - t
        if hasattr(done, "all") and bool(done.all()):
            state = env.reset()
    wall = time.perf_counter() - wall0

    gpu_frac = fwd_s / (wall + 1e-9)
    cpu_frac = step_s / (wall + 1e-9)
    # HONEST verdict: the env.step (host indicator/fill math) dominating means the
    # GPU is starved — raising BATCH_SIZE_ENV won't help util until the host loop
    # is faster (vectorize / move to device). Don't chase 80% GPU if CPU-bound.
    if cpu_frac > gpu_frac:
        verdict = (f"CPU-BOUND: env.step is {cpu_frac:.0%} of wall vs "
                   f"{gpu_frac:.0%} policy-forward. The host loop is the "
                   f"bottleneck — GPU util cannot reach target until env.step is "
                   f"faster. Raising batch size alone will NOT fix utilization.")
    else:
        verdict = (f"GPU-BOUND: policy-forward is {gpu_frac:.0%} of wall vs "
                   f"{cpu_frac:.0%} env.step. Raising BATCH_SIZE_ENV should lift "
                   f"utilization toward the target.")
    return {"n_steps": n_steps, "wall_s": wall,
            "forward_s": fwd_s, "env_step_s": step_s,
            "forward_frac": gpu_frac, "env_step_frac": cpu_frac,
            "device": device.type, "verdict": verdict}


def profile_report(rep: Dict) -> str:
    """Render a profile dict (from either profiler) as a human-readable block."""
    if rep.get("section") == "policy_forward":
        return (f"[profile] policy_forward: {rep['per_iter_ms']:.3f} ms/iter "
                f"over {rep['iters']} iters on {rep['device']}")
    return (
        "\n".join([
            "═" * 60,
            f"  PROFILE ({rep['device']}, {rep['n_steps']} steps, "
            f"{rep['wall_s']:.2f}s wall)",
            "═" * 60,
            f"  policy_forward : {rep['forward_s']:.3f}s  ({rep['forward_frac']:.0%})",
            f"  env.step       : {rep['env_step_s']:.3f}s  ({rep['env_step_frac']:.0%})",
            "─" * 60,
            f"  VERDICT: {rep['verdict']}",
            "═" * 60,
        ]))
