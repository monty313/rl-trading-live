# AUDIT.md — Source Repo Audit for `rl-trading-live`

**Date:** 2026-06-01
**Auditor:** Perplexity Computer (read-only clone audit)
**Source repos (read-only, never modified):**
- REPO1 — `monty313/deep-reinforcement-learning-trading` (the working GPU DQN system)
- REPO2 — `mmayes313/jordan-rl-trading-bot` (Streamlit + Jordan UI + reward ideas)

---

## Executive Summary

- **REPO1's `gpu_rl_trading/` package is authoritative.** It is a clean, GPU-centric, batched DQN
  trading system that is already running and producing checkpoints. We port it as the core of
  `rl-trading-live`, applying: 756-action space, `torch.compile`, AMP, `BATCH_SIZE_RL=2048`,
  `TRAIN_EVERY=2`, `MEMORY_SIZE=500_000`, vectorized day-boundary logic, `intrabar_fills`,
  `conditions_engine`, `phases.yaml`, `checkpoint_manager` + manifest, multi-symbol loading.
- **REPO1 root-level `train.py`, `env/`, `training/`, `monitoring/` are an OLDER DUPLICATE layer.**
  The authoritative loop is `gpu_rl_trading/training/train.py`. We port from the `gpu_rl_trading/`
  package only; root duplicates are **DEAD** for our purposes.
- **REPO2's `ppo_model.py` is a MOCK** (`class MockPPO`), not a functioning PPO. There is therefore
  no real PPO to port. The spec is DQN-based with a 756-action discrete space, which is the correct
  and only working algorithm across both repos. PPO hyperparameters from
  `data/models/best_hyperparams.json` are preserved in `config/` comments for any future PPO branch,
  but PPO is **not** wired into the live system. (Resolves the user's "may need PPO" note: PPO was
  never actually implemented — DQN is the proven path.)
- **REPO2's Streamlit + Jordan code** informs the new `dashboard/` and `jordan/` modules, but is
  rebuilt fresh against the spec (multi-page `st.Page` API, dark Orbitron theme, IRAC engine,
  consent flow, vitals daemon) rather than ported line-for-line.

### ALGORITHM_CONFLICT (resolved)
REPO2 assumes PPO (`stable-baselines3`, continuous-ish policy) while REPO1 uses a discrete DQN.
Because REPO2's PPO is a non-functional mock, there is **no real conflict to reconcile**. The system
standardizes on **DQN with a 756-action discrete space** (3 direction × 7 lot × 6 SL × 6 TP).

---

## REPO1 — File-by-File Categorization

| File | Disposition | Notes |
|------|-------------|-------|
| `gpu_rl_trading/agent/dqn.py` | **KEEP_PORT** | `DQNAgent` + `QNetwork` + `load_partial` transfer learning. Port to `core/agent/dqn.py`; set `NUM_ACTIONS=756` (import from `action_space.py`), add `torch.compile` + AMP autocast. |
| `gpu_rl_trading/agent/replay.py` | **KEEP_PORT** | `GPUReplayBuffer`. Port as-is into `core/agent/dqn.py` (or kept inline); bump capacity to 500k via cfg. |
| `gpu_rl_trading/env/environment.py` | **KEEP_PORT** (heavy edit) | `BatchedFTMOEnv`. Port to `core/env/environment.py`; wire `intrabar_fills`, add `(B,756)` action mask, multi-symbol, new PASS/FAIL rule, vectorize day-boundary loop (bottleneck #1). |
| `gpu_rl_trading/env/indicators.py` | **MERGE** | numpy indicators. Port to `core/env/indicators.py` as **GPU torch ops** with `device` arg; emit named feature columns required by `VARIABLE_REGISTRY` (sma_20, ema_20, cci_14, atr_14, atr_14_ma, rolling_high_20, rolling_low_20). |
| `gpu_rl_trading/training/train.py` | **KEEP_PORT** (heavy edit) | Main loop + `EpisodeRewardShaper`. Port to `training/train.py`; add YAML phases via `conditions_engine`, LIVE_IMPROVE infinite phase, CLI args, heartbeat, `checkpoint_manager` resume. |
| `gpu_rl_trading/config/settings.py` | **KEEP_PORT** | `CFG` dict. Port to `core/agent/`/`config`-level settings; apply A100 upgrades (BATCH_SIZE_RL 256→2048, TRAIN_EVERY 4→2, MEMORY_SIZE→500k). |
| `gpu_rl_trading/eval/backtest.py` | **MERGE** | Folds into `backtest/engine.py` with parity (md5) assertion. |
| `gpu_rl_trading/eval/mt5_backtest.py` | **MERGE** | Concepts fold into `backtest/engine.py`. |
| `gpu_rl_trading/live/live_agent.py` | **KEEP_PORT** (concepts) | Informs `broker/live_runner.py` per-bar inference loop. |
| `gpu_rl_trading/live/FTMO_DQN.mq5` | **DEAD** | MQL5 EA — out of scope for the Python repo. |
| `gpu_rl_trading/metrics/episode_rewards_EURUSD_gpu.csv` | **DEAD** (data) | Sample metrics — not code; not ported. |
| `gpu_rl_trading/notebooks/gpu_train_eurusd.ipynb` | **MERGE** | Superseded by new `rl_trading_colab.ipynb` (9 cells). |
| `gpu_rl_trading/CRASH_RECOVERY.md`, `TODO.md` | **MERGE** | Concepts fold into `README.md` + `scripts/crash_recovery.py`. |
| root `train.py` | **DEAD** | Older duplicate of `gpu_rl_trading/training/train.py`. Not authoritative. |
| root `env/`, `training/`, `monitoring/` | **DEAD** | Older duplicate package layer superseded by `gpu_rl_trading/`. `monitoring/` ideas (slippage, regime, tail risk) noted for future; not ported now. |
| root `smoke_test.py`, `test_run.py`, `tests/` | **MERGE** | Rebuilt as `scripts/smoke_*.py` + `tests/` against the new structure. |
| root `agents/`, `data/`, `execution/`, `config/` | **DEAD/MERGE** | Older layer; `execution/mt5_bridge.py` concepts inform `broker/mt5_adapter.py`. |

## REPO2 — File-by-File Categorization

| File | Disposition | Notes |
|------|-------------|-------|
| `src/models/ppo_model.py` | **DEAD** (mock) | `MockPPO` — not a real model. PPO not ported. |
| `src/rewards/reward_system.py` | **MERGE** (ideas) | Reward shaping ideas (consistency, frequency, drawdown) inform `core/reward/shaper.py`; the Φ-potential design from REPO1 wins. |
| `data/models/best_hyperparams.json` | **KEEP** (reference) | PPO hyperparams preserved as comments in `config/` for any future PPO branch. |
| `src/indicators/*` | **MERGE** (ideas) | Indicator math cross-checked against REPO1's; REPO1 GPU port wins. |
| `src/masks/trading_masks.py` | **MERGE** (ideas) | Action-mask concept folds into `core/env/conditions_engine.py`. |
| `streamlit/app.py`, `streamlit/pages/*`, `streamlit/components/jordan_personality.py` | **MERGE** (rebuild) | Inform `dashboard/` + `jordan/persona.py`; rebuilt fresh per spec (dark Orbitron, 5 pages, consent flow). |
| `src/environment/*` (multiple env variants) | **DEAD** | Superseded by REPO1's `BatchedFTMOEnv`. |
| `data/raw/*` (many CSVs) | **DEAD** (data) | Sample data; real data lives on Google Drive. |
| everything else (`scripts/`, `notebooks/`, `tests/`, `debug_*`) | **DEAD/MERGE** | Dev scratch; rebuilt against new spec where needed. |

---

## Authoritative-source decisions (the user's STEP 1 questions)

1. **Is root `train.py` the same as `gpu_rl_trading/training/train.py`?**
   No — root `train.py` is an older duplicate. **`gpu_rl_trading/training/train.py` is authoritative.**
2. **What is `NUM_ACTIONS`? Where defined?**
   `NUM_ACTIONS = 7` in both `gpu_rl_trading/config/settings.py` and `environment.py`.
   **We expand to 756** in `core/agent/action_space.py`; transfer learning bridges old→new output layer.
3. **`BatchedFTMOEnv.step()` signature:** returns `(state, reward, done, info)` — ported and extended
   to apply the `(B,756)` action mask before action selection.
4. **`EpisodeRewardShaper.compute_bonus()` accepts** a `daily_log` list of per-day dicts
   (`pass`, `ret`, `dd`). Ported; PASS/FAIL rule updated to the spec's 2.5% rule + weekly bonus.
5. **REPO1 `live/`** has `live_agent.py` (per-bar inference) + an `.mq5` EA. The Python runner concept
   is ported into `broker/live_runner.py`; the `.mq5` is out of scope.
