# [DIST PRE-PHASE] Colab Cells — TEMPORARY, REMOVE AT GRADUATION

Paste these cells into `rl_trading_colab.ipynb` after the standard setup
cells. Each cell is bookended with `[DIST PRE-PHASE START / END]` so
removal at graduation is a literal `grep -r "DIST PRE-PHASE" .` exercise.

## Cell 1 — "[DIST] Checkpoint Probe"

```python
# [DIST PRE-PHASE START — REMOVE AT GRADUATION]
import os, sys
sys.path.insert(0, "/content/rl-trading-live")
os.chdir("/content/rl-trading-live")
%run scripts/dist_checkpoint_probe.py --ckpt /content/drive/MyDrive/checkpoints/eurusd_gpu_ph0_ep0120.pt
# [DIST PRE-PHASE END]
```

## Cell 2 — "[DIST] Initialize Pre-Phase"

```python
# [DIST PRE-PHASE START — REMOVE AT GRADUATION]
import torch
from core.settings import CFG, get_device, auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.dist_teacher import DistDQNTeacher, DistPrePhaseWrapper
from core.dist_teacher.dist_obs_adapter import build_adapter_if_needed
from core.dist_phase import DistPhaseManager, DistPhase

# Flip kill switch ON for this run.
CFG["dist_prephase_enabled"] = True
device = get_device()
cfg = auto_tune_batch(dict(CFG), device)

# Build the base env with whatever data path you normally use.
# (Existing pipeline unchanged — only the wrap site is new.)
base_env = BatchedFTMOEnv(cfg)  # ← pass your usual constructor args

# Configure DQN risk window for the pre-phase via the phase manager.
dist_phase_mgr = DistPhaseManager(CFG, start_phase=DistPhase.PRE_PHASE)

# Build the obs adapter only if the probe reported a mismatch.
ppo_obs_dim = base_env.state_dim
dqn_input_dim = ppo_obs_dim   # replace with the value printed by Cell 1
dist_adapter = build_adapter_if_needed(ppo_obs_dim, dqn_input_dim)
dist_phase_mgr.record_adapter_info(
    used=dist_adapter is not None,
    dqn_input_dim=dqn_input_dim,
    ppo_obs_dim=ppo_obs_dim,
)

# Load the frozen DQN.
dist_teacher = DistDQNTeacher(
    checkpoint_path=CFG["dist_teacher"]["checkpoint_path"],
    device=device,
    action_order=CFG["dist_teacher"]["action_order"],
    obs_adapter=dist_adapter,
    temperature=CFG["dist_teacher"]["temperature"],
)

# Wrap the env.
env = DistPrePhaseWrapper(
    base_env,
    teacher=dist_teacher,
    dist_phase_manager=dist_phase_mgr,
    confidence_threshold=CFG["dist_teacher"]["confidence_threshold"],
    masking_enabled=CFG["dist_masking_enabled"],
)

# Pre-phase opening banner (Section 11).
print("╔" + "═"*58 + "╗")
print("║  [DIST] DIST_PRE_PHASE STARTED" + " "*27 + "║")
print(f"║  DQN teacher: ON | weight={dist_phase_mgr.get_distillation_weight():.2f}" + " "*23 + "║")
print(f"║  Checkpoint: eurusd_gpu_ph0_ep0120.pt" + " "*21 + "║")
print(f"║  TEMPORARY DD: {dist_phase_mgr.get_dist_max_daily_dd()*100:.0f}% max | "
      f"{dist_phase_mgr.get_dist_daily_target()*100:.0f}% daily target" + " "*18 + "║")
print("║  Lot size: fixed 0.01 (no sizing during pre-phase)" + " "*9 + "║")
print("║  NORMAL RULES RESUME at DIST_PHASE_1" + " "*22 + "║")
print("╚" + "═"*58 + "╝")
print(f"   Obs dim:        base={ppo_obs_dim} + 3 DQN slots = {env.state_dim}")
print(f"   Obs adapter:    {'YES (DQN_dim=' + str(dqn_input_dim) + ')' if dist_adapter else 'NO — dims match'}")
print(f"   Teacher frozen: {dist_teacher.is_frozen}")
# [DIST PRE-PHASE END]
```

## Day-end hook (call after every closed day)

```python
# [DIST PRE-PHASE START — REMOVE AT GRADUATION]
from core.dist_phase.dist_phase_manager import DistDailyMetrics

# In your daily eval/close handler, build a DistDailyMetrics object from the
# day's stats. The wrapper already accumulates the agreement counters per day
# in env.daily_entry_steps / env.daily_agreement_hits — snapshot them here.
diag = env.reset_daily_diagnostics()
metrics = DistDailyMetrics(
    day=current_day_idx,
    date=current_date_str,
    pnl_usd=day_pnl_usd,
    win_rate=day_win_rate,
    profit_factor=day_profit_factor,
    expectancy_pips_net=day_expectancy_pips,
    trades=day_trades,
    max_dd_pct=day_max_dd_pct,
    dd_breached=day_dd_breached,
    agreement_count=diag["agreement_count"],
    confident_entry_steps=diag["entry_step_count"],
)
summary = dist_phase_mgr.on_dist_day_end(metrics)
# Emit the [DIST] daily diagnostics line per Section 11.
# [DIST PRE-PHASE END]
```
