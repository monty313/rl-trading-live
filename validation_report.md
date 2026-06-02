# Validation Report — rl-trading-live (PURE PPO)

**Date**: 2026-06-02 06:06  
**Environment**: Local CPU (Python 3.12.8, RL_ALLOW_NUMPY_INDICATORS=1). talib/GPU/Drive/MT5 paths verified on Colab.

## Agent & strategy (authoritative)

- **Pure PPO**: one actor-critic outputs direction (FLAT/BUY/SELL), continuous lot size, and exit (hold/reduce/close). DQN deprecated to legacy/.

- **Forced-entry rule**: when any strategy gate is active, FLAT is masked in every phase — agent must hold a position; code never picks the side.

- **8-phase curriculum** (config/training_config.yaml) + multi-timeframe talib indicators (CCI30/100+shifted SMA, BB20/200, ATR14/45, SMA stacks). CCI300/900 excluded.

- **PASS/FAIL**: r_d>=2.5% AND dd<=1% -> pass; DD>1% (day ends immediately) or EOD<2.5% -> fail. Progressive day reward (pass/ok/fail + streak + low-DD) + Φ shaping. Episode = 30 days. FTMO CEST end-of-day.

## Commands Executed

| # | Command | Result |
|---|---------|--------|
| 1 | `python scripts/validate_phases_yaml.py` | exit 0 — 9 phases PASS |
| 2 | `python scripts/smoke_train.py` | exit 0 — SMOKE_TRAIN OK (PPO rollout+update) |
| 3 | `python scripts/smoke_backtest.py` | exit 0 — SMOKE_BACKTEST OK |
| 4 | `python scripts/smoke_infer.py` | exit 0 — SMOKE_INFER OK (dir,lot_raw,exit) |
| 5 | `python tests/run_all_tests.py` | exit 0 — 83 tests pass |
| 6 | `python inspect_system.py` | exit 0 — 0 failed, 1 skip (GPU) |

## Files (Python)

| File | Lines |
|------|-------|
| inspect_system.py | 220 |
| core/__init__.py | 1 |
| core/pipeline.py | 92 |
| core/settings.py | 105 |
| core/env/__init__.py | 1 |
| core/env/conditions_engine.py | 309 |
| core/env/environment.py | 388 |
| core/env/indicators.py | 296 |
| core/env/intrabar_fills.py | 105 |
| core/agent/__init__.py | 1 |
| core/agent/action_space.py | 82 |
| core/agent/ppo.py | 258 |
| core/reward/__init__.py | 1 |
| core/reward/shaper.py | 135 |
| core/risk/__init__.py | 1 |
| core/risk/daily_guard.py | 122 |
| core/risk/position_sizer.py | 37 |
| core/risk/trade_gate.py | 56 |
| training/__init__.py | 1 |
| training/checkpoint_manager.py | 203 |
| training/eval_loop.py | 67 |
| training/train.py | 164 |
| backtest/__init__.py | 1 |
| backtest/engine.py | 131 |
| broker/__init__.py | 1 |
| broker/account_manager.py | 33 |
| broker/broker_base.py | 22 |
| broker/live_runner.py | 96 |
| broker/mt5_adapter.py | 96 |
| broker/schwab_stub.py | 18 |
| broker/tradovate_stub.py | 18 |
| jordan/__init__.py | 1 |
| jordan/consent_flow.py | 60 |
| jordan/irac_engine.py | 73 |
| jordan/persona.py | 81 |
| jordan/policy_inspector.py | 57 |
| jordan/repo_indexer.py | 46 |
| jordan/vitals_daemon.py | 83 |
| dashboard/__init__.py | 1 |
| dashboard/app.py | 89 |
| dashboard/pages/__init__.py | 1 |
| dashboard/pages/beast_mode.py | 25 |
| dashboard/pages/jordan_chat.py | 51 |
| dashboard/pages/live_dashboard.py | 49 |
| dashboard/pages/model_control.py | 48 |
| dashboard/pages/performance.py | 25 |
| monitoring/__init__.py | 1 |
| monitoring/alert_dispatcher.py | 63 |
| monitoring/flatline_detector.py | 43 |
| scripts/crash_recovery.py | 40 |
| scripts/smoke_backtest.py | 28 |
| scripts/smoke_infer.py | 40 |
| scripts/smoke_train.py | 64 |
| scripts/validate_phases_yaml.py | 117 |
| tests/__init__.py | 1 |
| tests/conftest.py | 9 |
| tests/run_all_tests.py | 170 |
| tests/unit/__init__.py | 1 |
| tests/unit/test_action_space.py | 29 |
| tests/unit/test_checkpoint_manager.py | 72 |
| tests/unit/test_conditions_engine.py | 105 |
| tests/unit/test_daily_guard.py | 46 |
| tests/unit/test_environment.py | 69 |
| tests/unit/test_eval_loop.py | 20 |
| tests/unit/test_flatline_detector.py | 42 |
| tests/unit/test_indicators.py | 33 |
| tests/unit/test_indicators_multitf.py | 49 |
| tests/unit/test_intrabar_fills.py | 33 |
| tests/unit/test_jordan.py | 44 |
| tests/unit/test_known_bug_fixes.py | 50 |
| tests/unit/test_mt5_adapter.py | 41 |
| tests/unit/test_pipeline.py | 21 |
| tests/unit/test_position_sizer.py | 20 |
| tests/unit/test_ppo.py | 59 |
| tests/unit/test_reward_shaper.py | 66 |
| tests/unit/test_trade_gate.py | 31 |
| tests/integration/__init__.py | 1 |
| tests/integration/test_backtest_one_day.py | 15 |
| tests/integration/test_eval_loop.py | 17 |
| tests/integration/test_pipeline_parity.py | 22 |
| tests/integration/test_train_one_episode.py | 32 |
| tests/fixtures/__init__.py | 1 |
| tests/fixtures/sample_candles.py | 80 |
| tests/fixtures/sample_configs.py | 74 |
| tests/mocks/__init__.py | 1 |
| tests/mocks/mock_mt5.py | 55 |
| tests/mocks/mock_telegram.py | 18 |
| legacy/__init__.py | 7 |
| legacy/dqn.py | 305 |

## Tests (83) — all ✅ PASS

| File | Test |
|------|------|
| tests/integration/test_backtest_one_day.py | test_backtest_one_day |
| tests/integration/test_eval_loop.py | test_eval_loop_integration |
| tests/integration/test_pipeline_parity.py | test_indicators_bit_identical |
| tests/integration/test_pipeline_parity.py | test_parity_hashes_present |
| tests/integration/test_train_one_episode.py | test_train_one_episode |
| tests/unit/test_action_space.py | test_direction_and_exit_dims |
| tests/unit/test_action_space.py | test_map_lot_range_and_clamp |
| tests/unit/test_action_space.py | test_decode_structured_action |
| tests/unit/test_action_space.py | test_describe |
| tests/unit/test_checkpoint_manager.py | test_rolling_deletion_keeps_five_lowest_phi_removed |
| tests/unit/test_checkpoint_manager.py | test_protected_never_deleted |
| tests/unit/test_checkpoint_manager.py | test_bootstrap_from_existing |
| tests/unit/test_checkpoint_manager.py | test_find_best_resume |
| tests/unit/test_checkpoint_manager.py | test_verify_all_detects_corrupt |
| tests/unit/test_conditions_engine.py | test_any_always_true |
| tests/unit/test_conditions_engine.py | test_evaluate_uses_features |
| tests/unit/test_conditions_engine.py | test_unknown_variable_raises_irac |
| tests/unit/test_conditions_engine.py | test_string_buy_condition_masks_to_buy |
| tests/unit/test_conditions_engine.py | test_free_allows_all |
| tests/unit/test_conditions_engine.py | test_phase0_cci_extreme_both_high |
| tests/unit/test_conditions_engine.py | test_phase0_opposite_direction_false |
| tests/unit/test_conditions_engine.py | test_phase1_cci_align |
| tests/unit/test_conditions_engine.py | test_phase6_atr_expansion |
| tests/unit/test_conditions_engine.py | test_force_in_and_gate_active_masks_flat_always |
| tests/unit/test_conditions_engine.py | test_force_in_and_gate_blocks_entries_when_condition_false |
| tests/unit/test_conditions_engine.py | test_open_gate_allows_all_when_true_hold_only_when_false |
| tests/unit/test_daily_guard.py | test_ftmo_halts_at_dd_threshold |
| tests/unit/test_daily_guard.py | test_ftmo_no_halt_within_limit |
| tests/unit/test_daily_guard.py | test_pass_recorded_when_target_hit |
| tests/unit/test_daily_guard.py | test_force_halt |
| tests/unit/test_daily_guard.py | test_beast_trailing_from_peak |
| tests/unit/test_daily_guard.py | test_trade_cap_halts |
| tests/unit/test_environment.py | test_state_shape |
| tests/unit/test_environment.py | test_step_contract |
| tests/unit/test_environment.py | test_direction_mask_shape |
| tests/unit/test_environment.py | test_episode_terminates |
| tests/unit/test_environment.py | test_multi_symbol_alignment |
| tests/unit/test_eval_loop.py | test_eval_returns_four_keys |
| tests/unit/test_flatline_detector.py | test_flatline_fires_on_flat_series |
| tests/unit/test_flatline_detector.py | test_no_flatline_when_improving |
| tests/unit/test_flatline_detector.py | test_alert_appends_to_session_and_telegram |
| tests/unit/test_flatline_detector.py | test_alert_never_raises_when_all_channels_fail |
| tests/unit/test_indicators.py | test_shape_and_finite |
| tests/unit/test_indicators.py | test_named_columns_present |
| tests/unit/test_indicators.py | test_rolling_high_ge_close |
| tests/unit/test_indicators.py | test_device_placement_cpu |
| tests/unit/test_indicators_multitf.py | test_compute_indicators_has_authoritative_columns |
| tests/unit/test_indicators_multitf.py | test_cci300_and_cci900_removed |
| tests/unit/test_indicators_multitf.py | test_resample_to_15m |
| tests/unit/test_indicators_multitf.py | test_feature_matrix_shape_and_columns |
| tests/unit/test_intrabar_fills.py | test_buy_fill_above_open |
| tests/unit/test_intrabar_fills.py | test_sell_fill_below_open |
| tests/unit/test_intrabar_fills.py | test_hold_returns_open |
| tests/unit/test_intrabar_fills.py | test_atr_caps_sl_width |
| tests/unit/test_jordan.py | test_irac_has_four_sections |
| tests/unit/test_jordan.py | test_persona_fallback_without_key |
| tests/unit/test_jordan.py | test_consent_requires_two_steps |
| tests/unit/test_jordan.py | test_jordan_can_read_test_results |
| tests/unit/test_known_bug_fixes.py | test_checkpoint_roundtrip_with_metadata |
| tests/unit/test_known_bug_fixes.py | test_env_returns_executed_direction |
| tests/unit/test_known_bug_fixes.py | test_partial_load_best_effort_on_shape_change |
| tests/unit/test_mt5_adapter.py | test_symbol_alias_resolves |
| tests/unit/test_mt5_adapter.py | test_send_order_calls_gate_and_blocks |
| tests/unit/test_mt5_adapter.py | test_send_order_fills_when_approved |
| tests/unit/test_pipeline.py | test_returns_five_objects |
| tests/unit/test_position_sizer.py | test_clamped_to_max_lot |
| tests/unit/test_position_sizer.py | test_minimum_lot_floor |
| tests/unit/test_position_sizer.py | test_no_crash_on_any_bucket |
| tests/unit/test_ppo.py | test_select_actions_shapes |
| tests/unit/test_ppo.py | test_direction_mask_blocks_flat |
| tests/unit/test_ppo.py | test_update_runs_and_clears_buffer |
| tests/unit/test_ppo.py | test_save_load_roundtrip |
| tests/unit/test_ppo.py | test_select_action_single_obs |
| tests/unit/test_reward_shaper.py | test_warmup_returns_zero |
| tests/unit/test_reward_shaper.py | test_weekly_bonus_fires_when_rate_improves |
| tests/unit/test_reward_shaper.py | test_no_weekly_bonus_when_flat |
| tests/unit/test_reward_shaper.py | test_compute_bonus_runs_after_warmup |
| tests/unit/test_reward_shaper.py | test_daily_reward_pass_and_streak |
| tests/unit/test_reward_shaper.py | test_daily_reward_fail_resets_streak_and_penalizes |
| tests/unit/test_reward_shaper.py | test_daily_reward_low_dd_bonus |
| tests/unit/test_trade_gate.py | test_approve_true_when_not_halted |
| tests/unit/test_trade_gate.py | test_approve_false_when_halted |
| tests/unit/test_trade_gate.py | test_consent_blocks |

## Unverified (Colab A100 / Drive / Windows / talib)

- ⚠️ TA-Lib (required in prod) — CPU CI uses numpy fallback via RL_ALLOW_NUMPY_INDICATORS=1
- ⚠️ torch.cuda / A100, torch.compile + AMP (device-guarded)
- ⚠️ Google Drive multi-symbol data; real MT5 live execution

## Failed Checks
- None.