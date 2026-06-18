"""
tests/unit/test_target_aware_policy.py
────────────────────────────────────────────────────────────────────────────
LOCKS the target/risk-AWARE policy work (target_aware_policy.md). Companion to
test_runtime_ftmo_config.py (which proves the RULES are config-driven). Here we
prove the POLICY can CONDITION on the inputs and adapt proportionally:

  • the observation carries the 7 new target/risk-aware features with correct
    values (progress_to_target==1.0 at exact target; dd_headroom→0 at breach);
  • changing target_pct / max_dd_pct changes the observation the agent sees;
  • randomization mode samples in-range and the env's classification uses the
    sampled values that episode;
  • state_dim matches ActorCritic input dim (and the obs schema version);
  • resuming a checkpoint with a mismatched obs schema is handled gracefully
    (input layer reinitialized, no silent wrong-load, no crash);
  • the item-6 proportional scaler is 1.0 at baseline, lower with tighter DD,
    higher with bigger target, always within bounds;
  • the pass-probability estimator returns a prob in [0,1] with a valid CI, runs
    on tiny synthetic data, and an easy target yields a higher prob than a hard one.
"""
import torch

from core.settings import CFG, auto_tune_batch
from core.env.environment import (BatchedFTMOEnv, OBS_SCHEMA_VERSION,
                                   N_POSITION_FEATS, N_FTMO_FEATS,
                                   N_SESSION_FEATS, N_COST_FEATS,
                                   proportional_lot_scale)
from core.agent.ppo import PPOAgent, ActorCritic
from core.agent.action_space import FLAT
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def _cfg(account=10_000.0, target_pct=0.025, max_dd_pct=0.010, bars_per_day=60,
         **extra):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({
        "EPISODE_BARS": bars_per_day * 6,
        "BARS_PER_DAY": bars_per_day,
        "LOOKBACK": 20,
        "ACCOUNT_SIZE": account,
        "INITIAL_EQUITY": account,
        "DAILY_TARGET_PCT": target_pct,
        "DAILY_MAX_DD_PCT": max_dd_pct,
        "FEATURES": make_synthetic_ohlcv_array(n=800),
        "SYNTH_BARS": 800,
    })
    c.update(extra)
    return c


def _env(**kw):
    arr = make_synthetic_ohlcv_array(n=800)
    return BatchedFTMOEnv(arr, _cfg(**kw), DEV, instrument="EURUSD",
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})


# ════════════════════════════════════════════════════════════════════════════
# 1. The observation carries the 7 target/risk-aware features (item 1)
# ════════════════════════════════════════════════════════════════════════════
def test_state_dim_includes_ftmo_feats():
    env = _env()
    # lkbk*F + N_POSITION_FEATS + N_FTMO_FEATS + N_SESSION_FEATS (v3 schema adds
    # the session/streak/commission block AFTER the FTMO block).
    assert env.state_dim == (env.lkbk * env.F + N_POSITION_FEATS
                             + N_FTMO_FEATS + N_SESSION_FEATS
                             + N_COST_FEATS)
    s = env.reset()
    assert s.shape == (env.B, env.state_dim)


def _ftmo_slice(env, state):
    """The target/risk-aware FTMO features in the documented order: target, maxdd,
    difficulty, progress, dd_headroom, day_remaining, account_log. v4 schema:
    state ends with [... FTMO, SESSION, COST]. Slice the FTMO chunk by stepping
    back from the END by (FTMO + SESSION + COST) and forward by (SESSION + COST)."""
    tail = N_SESSION_FEATS + N_COST_FEATS
    return state[:, -(N_FTMO_FEATS + tail):-tail]


def test_obs_target_and_maxdd_features_present():
    env = _env(target_pct=0.03, max_dd_pct=0.015)
    s = env.reset()
    f = _ftmo_slice(env, s)
    assert torch.allclose(f[:, 0], torch.full((env.B,), 0.03), atol=1e-6)   # target_pct
    assert torch.allclose(f[:, 1], torch.full((env.B,), 0.015), atol=1e-6)  # max_dd_pct


def test_progress_to_target_is_one_at_exact_target():
    """progress_to_target == 1.0 EXACTLY when equity - day_start == daily_increment."""
    env = _env(account=10_000.0, target_pct=0.025)   # increment $250
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._equity[:] = 10_250.0                          # exactly +$250
    env._day_high_eq[:] = 10_250.0
    f = _ftmo_slice(env, env._get_state())
    assert torch.allclose(f[:, 3], torch.ones(env.B), atol=1e-6)   # progress


def test_progress_clamped_below_target():
    env = _env(account=10_000.0, target_pct=0.025)   # increment $250
    env.reset()
    env._day_start_eq[:] = 10_000.0
    env._equity[:] = 10_125.0                          # +$125 = half the target
    env._day_high_eq[:] = 10_125.0
    f = _ftmo_slice(env, env._get_state())
    assert torch.allclose(f[:, 3], torch.full((env.B,), 0.5), atol=1e-6)


def test_dd_headroom_goes_to_zero_at_breach():
    """dd_headroom → 0 as equity approaches the breach floor peak*(1-max_dd)."""
    env = _env(account=10_000.0, max_dd_pct=0.01)
    env.reset()
    env._day_high_eq[:] = 10_000.0                     # peak
    # full room when equity == peak
    env._equity[:] = 10_000.0
    f_full = _ftmo_slice(env, env._get_state())
    assert torch.allclose(f_full[:, 4], torch.ones(env.B), atol=1e-6)
    # at the breach floor (peak*(1-0.01) = 9900) headroom == 0
    env._equity[:] = 9_900.0
    f_breach = _ftmo_slice(env, env._get_state())
    assert torch.allclose(f_breach[:, 4], torch.zeros(env.B), atol=1e-6)


def test_fraction_of_day_remaining_decreases():
    env = _env(bars_per_day=60)
    env.reset()
    f0 = _ftmo_slice(env, env._get_state())[:, 5]
    assert torch.allclose(f0, torch.ones(env.B), atol=1e-6)   # day start = full
    # advance some bars
    out = {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
           "lot_raw": torch.zeros(env.B),
           "exit": torch.zeros(env.B, dtype=torch.long)}
    for _ in range(30):
        env.step(out)
    f_mid = _ftmo_slice(env, env._get_state())[:, 5]
    assert bool((f_mid < f0).all())
    assert bool((f_mid >= 0.0).all())


# ════════════════════════════════════════════════════════════════════════════
# 2. Changing target_pct / max_dd_pct changes the observation
# ════════════════════════════════════════════════════════════════════════════
def test_changing_target_dd_changes_observation():
    env_a = _env(target_pct=0.02, max_dd_pct=0.01)
    env_b = _env(target_pct=0.05, max_dd_pct=0.02)
    fa = _ftmo_slice(env_a, env_a.reset())
    fb = _ftmo_slice(env_b, env_b.reset())
    # the target & maxdd columns differ -> the agent literally sees a different obs
    assert not torch.allclose(fa[:, 0], fb[:, 0])
    assert not torch.allclose(fa[:, 1], fb[:, 1])


# ════════════════════════════════════════════════════════════════════════════
# 3. Randomization mode samples in-range and the env uses the sampled values
# ════════════════════════════════════════════════════════════════════════════
def test_randomize_ftmo_samples_in_range():
    torch.manual_seed(0)
    env = _env(RANDOMIZE_FTMO_INPUTS=True,
               RANDOMIZE_TARGET_RANGE=[0.01, 0.05],
               RANDOMIZE_DD_RANGE=[0.005, 0.02])
    env.reset()
    assert bool((env._target_pct_t >= 0.01).all() and (env._target_pct_t <= 0.05).all())
    assert bool((env._max_dd_pct_t >= 0.005).all() and (env._max_dd_pct_t <= 0.02).all())
    # the sampled values are surfaced in the observation (item 1)
    f = _ftmo_slice(env, env._get_state())
    assert torch.allclose(f[:, 0], env._target_pct_t, atol=1e-6)
    assert torch.allclose(f[:, 1], env._max_dd_pct_t, atol=1e-6)


def test_randomize_ftmo_classification_uses_sampled_values():
    """The env's PASS test uses the PER-EPISODE sampled increment, not the scalar."""
    torch.manual_seed(1)
    env = _env(account=10_000.0, RANDOMIZE_FTMO_INPUTS=True,
               RANDOMIZE_TARGET_RANGE=[0.04, 0.04],   # pin target to 4% for the test
               RANDOMIZE_DD_RANGE=[0.01, 0.01], bars_per_day=40)
    env.reset()
    # pinned range -> every episode sampled exactly 4% -> increment $400
    assert torch.allclose(env._daily_increment_t, torch.full((env.B,), 400.0), atol=1e-3)
    # drive a +$300 day: FAILS under the sampled 4% (target $400), would PASS at 2.5%
    env._equity[:] = 10_000.0
    env._day_start_eq[:] = 10_000.0
    env._day_high_eq[:] = 10_000.0
    env._equity_prev[:] = 10_000.0
    info = None
    bpd = env.bars_per_day
    out = {"direction": torch.full((env.B,), FLAT, dtype=torch.long),
           "lot_raw": torch.zeros(env.B),
           "exit": torch.zeros(env.B, dtype=torch.long)}
    for step in range(bpd):
        tgt = 10_300.0 if step == bpd - 1 else 10_000.0
        env._equity[:] = tgt
        env._day_high_eq[:] = torch.maximum(env._day_high_eq, env._equity)
        info = env.step(out)[3]
        env._equity[:] = tgt
    assert bool(info["failed"].all())          # +$300 < $400 sampled target -> FAIL


def test_off_mode_obs_constant_equals_cfg():
    """With randomization OFF the obs still carries the (constant) cfg target/DD."""
    env = _env(target_pct=0.035, max_dd_pct=0.012)
    assert env.randomize_ftmo is False
    f = _ftmo_slice(env, env.reset())
    assert torch.allclose(f[:, 0], torch.full((env.B,), 0.035), atol=1e-6)
    assert torch.allclose(f[:, 1], torch.full((env.B,), 0.012), atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# 4. state_dim matches ActorCritic input dim
# ════════════════════════════════════════════════════════════════════════════
def test_state_dim_matches_actorcritic_input_dim():
    env = _env()
    ac = ActorCritic(env.state_dim)
    # first trunk layer's in_features must equal state_dim
    assert ac.trunk[0].in_features == env.state_dim
    agent = PPOAgent(env.state_dim, _cfg(), DEV)
    assert agent.net.trunk[0].in_features == env.state_dim
    # a forward pass on a real state works end-to-end
    s = env.reset()
    out = agent.select_actions(s, mask=env.current_direction_mask())
    assert out["direction"].shape == (env.B,)


def test_obs_schema_version_is_v4():
    env = _env()
    assert env.obs_schema_version == OBS_SCHEMA_VERSION == 4


# ════════════════════════════════════════════════════════════════════════════
# 5. Mismatched-schema resume handled gracefully (item 4)
# ════════════════════════════════════════════════════════════════════════════
def test_mismatched_obs_schema_resume_handled(tmp_path, capsys):
    """Save a v1-shaped checkpoint (smaller input layer / no schema tag) and load
    it into a v2 agent: the input layer is reinitialized, other layers transfer,
    no crash, and a LOUD mismatch message is printed."""
    cfg = _cfg()
    env = _env()
    agent_v2 = PPOAgent(env.state_dim, cfg, DEV)

    # Build an "old" agent with a SMALLER input dim (simulates v1 obs schema).
    old_dim = env.state_dim - N_FTMO_FEATS    # the v1 layout (before the 7 feats)
    agent_v1 = PPOAgent(old_dim, cfg, DEV)
    ckpt = tmp_path / "v1.pt"
    # Save WITHOUT a schema tag to also exercise the input-dim fallback path.
    payload = {"net": agent_v1.net.state_dict(),
               "optimizer": agent_v1.optimizer.state_dict(),
               "state_dim": old_dim, "agent": "ppo",
               "phase": "p", "episode": 3, "phi": 0.0, "pass_rate": 0.0}
    torch.save(payload, str(ckpt))

    # Loading must NOT crash and must reinit the input layer (kept v2 width).
    agent_v2.load(str(ckpt), partial=True)
    assert agent_v2.net.trunk[0].in_features == env.state_dim
    out = capsys.readouterr().out
    assert "OBSERVATION-SCHEMA MISMATCH" in out
    # the agent still runs forward on a v2 state
    s = env.reset()
    o = agent_v2.select_actions(s, mask=env.current_direction_mask())
    assert o["direction"].shape == (env.B,)


def test_matching_schema_loads_full(tmp_path):
    """A same-schema checkpoint loads with no mismatch warning and restores the
    proportional-scaler baseline metadata."""
    cfg = _cfg()
    env = _env()
    a = PPOAgent(env.state_dim, cfg, DEV)
    ckpt = tmp_path / "v2.pt"
    a.save(str(ckpt), extra={"phase": "p", "episode": 5, "phi": 0.1, "pass_rate": 0.0})
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert blob["obs_schema_version"] == OBS_SCHEMA_VERSION
    assert abs(blob["trained_target_pct"] - 0.025) < 1e-9
    assert abs(blob["trained_max_dd_pct"] - 0.010) < 1e-9
    b = PPOAgent(env.state_dim, cfg, DEV)
    b.load(str(ckpt), partial=True)
    assert abs(b.trained_target_pct - 0.025) < 1e-9


def test_randomized_checkpoint_baseline_is_midpoint(tmp_path):
    """When trained with --randomize-ftmo the persisted scaler baseline is the
    MIDPOINT of the ranges (item 6)."""
    cfg = _cfg(RANDOMIZE_FTMO_INPUTS=True,
               RANDOMIZE_TARGET_RANGE=[0.01, 0.05],
               RANDOMIZE_DD_RANGE=[0.005, 0.02])
    env = _env(RANDOMIZE_FTMO_INPUTS=True)
    a = PPOAgent(env.state_dim, cfg, DEV)
    ckpt = tmp_path / "rnd.pt"
    a.save(str(ckpt), extra={"phase": "p", "episode": 1, "phi": 0.0, "pass_rate": 0.0})
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert abs(blob["trained_target_pct"] - 0.03) < 1e-9     # midpoint of [.01,.05]
    assert abs(blob["trained_max_dd_pct"] - 0.0125) < 1e-9   # midpoint of [.005,.02]


# ════════════════════════════════════════════════════════════════════════════
# 6. Proportional scaler: 1.0 at baseline, lower with tighter DD, higher target,
#    always in bounds (item 6)
# ════════════════════════════════════════════════════════════════════════════
def test_scaler_is_one_at_baseline():
    s = proportional_lot_scale(0.025, 0.010, 0.025, 0.010)
    assert abs(s - 1.0) < 1e-9


def test_scaler_lower_with_tighter_dd():
    base = proportional_lot_scale(0.025, 0.010, 0.025, 0.010)
    tighter = proportional_lot_scale(0.025, 0.005, 0.025, 0.010)   # half the DD room
    assert tighter < base
    assert abs(tighter - 0.5) < 1e-6      # dd_ratio 0.5 * f(1.0)=1.0 -> 0.5


def test_scaler_higher_with_bigger_target():
    base = proportional_lot_scale(0.025, 0.010, 0.025, 0.010)
    bigger = proportional_lot_scale(0.05, 0.010, 0.025, 0.010)     # double the target
    assert bigger > base


def test_scaler_always_within_bounds():
    # extreme inputs both directions must clamp to [0.25, 2.0]
    lo = proportional_lot_scale(0.001, 0.0001, 0.025, 0.010, lo=0.25, hi=2.0)
    hi = proportional_lot_scale(0.5, 0.5, 0.025, 0.010, lo=0.25, hi=2.0)
    assert lo == 0.25
    assert hi == 2.0
    for t in (0.005, 0.01, 0.025, 0.05, 0.1):
        for d in (0.002, 0.005, 0.01, 0.02, 0.05):
            s = proportional_lot_scale(t, d, 0.025, 0.010)
            assert 0.25 <= s <= 2.0


def test_agent_scaler_toggle_off_returns_one():
    cfg = _cfg(PROPORTIONAL_SCALER=False)
    env = _env()
    a = PPOAgent(env.state_dim, cfg, DEV)
    a.trained_target_pct, a.trained_max_dd_pct = 0.025, 0.010
    # even with very different current values, OFF -> 1.0
    assert a.proportional_scale(0.05, 0.005) == 1.0


# ════════════════════════════════════════════════════════════════════════════
# 7. Pass-probability estimator (item 7)
# ════════════════════════════════════════════════════════════════════════════
def test_estimator_returns_valid_prob_and_ci():
    from training.estimate_pass_prob import estimate_pass_prob, wilson_ci
    cfg = _cfg(bars_per_day=60)
    env = _env(bars_per_day=60)
    a = PPOAgent(env.state_dim, cfg, DEV)
    res = estimate_pass_prob(env, a, cfg, n_days=30)
    assert 0.0 <= res["pass_prob"] <= 1.0
    assert 0.0 <= res["ci_lo"] <= res["ci_hi"] <= 1.0
    assert res["n_days"] > 0
    assert 0.0 <= res["dd_breach_rate"] <= 1.0
    # Wilson CI sanity
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_estimator_easy_target_beats_hard_target():
    """An easy target (0.1%) yields a HIGHER pass probability than a hard one (5%)
    on the SAME tiny synthetic data + same untrained policy (item 7 direction)."""
    from training.estimate_pass_prob import estimate_pass_prob
    torch.manual_seed(7)
    cfg_easy = _cfg(target_pct=0.001, max_dd_pct=0.02, bars_per_day=60)
    env_easy = _env(target_pct=0.001, max_dd_pct=0.02, bars_per_day=60)
    a = PPOAgent(env_easy.state_dim, cfg_easy, DEV)
    easy = estimate_pass_prob(env_easy, a, cfg_easy, n_days=40)

    cfg_hard = _cfg(target_pct=0.05, max_dd_pct=0.02, bars_per_day=60)
    env_hard = _env(target_pct=0.05, max_dd_pct=0.02, bars_per_day=60)
    a2 = PPOAgent(env_hard.state_dim, cfg_hard, DEV)
    hard = estimate_pass_prob(env_hard, a2, cfg_hard, n_days=40)

    assert easy["pass_prob"] >= hard["pass_prob"]
