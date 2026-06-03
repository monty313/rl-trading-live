# INTERFACES.md — Module Contracts (internal build reference)

This file pins the exact function/class signatures every module must expose so
the pieces connect. It is the contract all builders follow. (Kept in-repo; also
serves future LLMs maintaining the code.)

## Environment & device
- `core/settings.py`: `CFG` dict, `get_device()->torch.device`, `auto_tune_batch(cfg,device)->cfg`.
- Device is auto-detected. All code must run on CPU (dev/CI) and CUDA (Colab) identically.
- `cfg["device"]` is set by `build_pipeline` before constructing objects.

## core/agent/action_space.py  (PPO)
- `DIRECTION_DIM=3` (FLAT/BUY/SELL), `EXIT_DIM=3` (HOLD/REDUCE/CLOSE), `LOT_DIM=1`.
- `map_lot(raw,max_lot)->float` (raw in [0,1] -> [MIN_LOT,max_lot]). `decode((dir,lot_raw,exit))->dict`.

## core/env/indicators.py
- `build_feature_matrix(open,high,low,close,volume,device)->torch.Tensor (N,F)`.
  Accepts np arrays or tensors; returns float32 tensor on `device`; no NaNs (warmup rows filled).
- Must expose NAMED columns via `FEATURE_COLUMNS: list[str]` and `COL: dict[name->idx]` including:
  `close, open, high, low, volume, sma_20, ema_20, cci_14, atr_14, atr_14_ma,
   rolling_high_20, rolling_low_20`.
- `feature_row_dict(features_row)->dict[name->float]` helper for conditions_engine.

## core/env/conditions_engine.py
- `VARIABLE_REGISTRY: set[str]` = the named columns above.
- `evaluate(condition_str, bar_features_dict)->bool`. `"any"`->True. Safe eval (no builtins).
- Unknown variable -> `raise ConfigError("... not in VARIABLE_REGISTRY ...")`.
- `compute_action_mask(phase, rows_by_tf, device, is_flat=True)->(dir_mask (DIRECTION_DIM,), must_enter)`
  returns 1.0 allowed / 0.0 masked per RULE 12: if buy True -> mask all HOLD+SELL; if sell True ->
  mask HOLD+BUY; both False -> all allowed; `"any"` -> all allowed.
- `class ConfigError(Exception)`.

## core/env/intrabar_fills.py
- `compute_fill(bar:dict, direction:int, sl_pips:int, tp_pips:int, instrument:str,
  policy:dict, atr_14:float=None)->dict` returns `{entry, sl, tp, spread_cost}`.
  Implements the exact BUY/SELL fill rules from STEP 4.2; clamps SL to <= atr_14*3.

## core/env/environment.py
- `class BatchedFTMOEnv`: `__init__(features:np.ndarray|tensor, cfg, device, instrument="EURUSD")`.
  `.state_dim:int`, `.reset()->state (B,state_dim)`, `.step(actions:(B,))->(state,reward,done,info)`.
  Applies action mask each step; PASS/FAIL per RULE 7; preloads features to device.
- Multi-symbol: `build_multi_symbol_env(instruments:list, cfg, device)` aligns by datetime.

## core/agent/ppo.py  (the single live agent; DQN deprecated -> legacy/)
- `class PPOAgent(state_dim, cfg, device)`: `select_actions(state, mask=None)->dict`
  (direction,exit,lot_raw,logp,value); `select_action(obs,deterministic,mask)->(dir,lot_raw,exit)`;
  `store(state,out,reward,done,dir_mask)`; `update(last_value=None)->float|None`;
  `save(path,extra)`; `load(path,partial=False)->dict`.
- mask is (B,DIRECTION_DIM): masked directions get -1e9 before sampling. AMP/compile on CUDA.

## core/reward/shaper.py
- `class EpisodeRewardShaper(cfg)`: `compute_bonus(daily_log:list[dict])->float`,
  `weekly_consistency_bonus()->float`, tracks 14-day deque. PASS rule = final-or-halt
  equity >= day_start + daily_increment (daily_increment = initial_equity*target_pct,
  a fixed $ amount; NOT start*1.025).

## core/risk/position_sizer.py
- `class PositionSizer(cfg)`: `size(lot_idx, max_lot, balance)->float` (clamp 0.01..max_lot, risk warn).

## core/risk/daily_guard.py
- `class DailyGuard(mode, initial_balance, cfg)`: `update(equity, trade_count)`, `force_halt()`,
  `reset_day()`, `get_status()->dict{mode,equity,peak,baseline,dd_pct,trades_today,halted,pass_fail}`.
  PASS/FAIL per RULE 7. FTMO trailing from prior-day close; Beast from peak.

## core/risk/trade_gate.py
- `class TradeGate(daily_guard, log_path="logs/daily_trade_log.csv")`: `approve(order:dict)->bool`,
  `set_consent(bool)`. Logs BLOCKED orders. False when guard halted or consent unresolved.

## core/pipeline.py
- `build_pipeline(cfg, device)->(env, agent, sizer, guard, gate)`. Single import point.

## training/checkpoint_manager.py
- `class CheckpointManager(checkpoint_dir, manifest_path)`: `bootstrap()`, `save(agent,phase,episode,phi,pass_rate,name=None)`,
  `find_best_resume(phase=None)->Path|None`, `verify_all()->dict`, `parity_hashes(set)`.
  Protected: latest.pt,best_eval.pt,live_trading.pt,transfer_start.pt. Keep<=5 per phase (auto-tune by free space).

## training/eval_loop.py
- `run_eval(env, agent, cfg, n_days=10)->dict{pass_rate,phi,avg_daily_return,avg_daily_dd}`.

## training/train.py
- CLI: `--csv --checkpoint-dir --metrics-dir --manifest --resume --start-phase --force-fresh`.
- Loads phases.yaml, runs curriculum then infinite LIVE_IMPROVE, heartbeat every 60s, resume always.

## backtest/engine.py
- `run_backtest(csv, cfg, device, manifest_path=None)->dict{daily_returns,pass_fail,phi,total_pass_days,total_fail_days,max_drawdown_pct}`.
- Parity: md5 of indicators.py & intrabar_fills.py vs manifest "parity_hashes" -> `ParityError`.

## broker/*  — base ABC, mt5_adapter (import-guarded), live_runner, account_manager, stubs.
## jordan/*  — vitals_daemon, irac_engine.generate_irac(event_type,event_data)->md, policy_inspector,
##             persona.get_response(context,msg)->str (Grok + fallback), consent_flow, repo_indexer.
## dashboard/* — app.py + 5 pages. monitoring/* — flatline_detector, alert_dispatcher.
## scripts/* — crash_recovery, validate_phases_yaml, smoke_train, smoke_backtest, smoke_infer.
## inspect_system.py — runs every check, IRAC on fail, exit 0/1.
