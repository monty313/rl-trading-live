# Validation Report — rl-trading-live

**Date**: 2026-06-02 04:49  
**Environment**: Local CPU validation (Python 3.12.8). GPU/Drive/MT5 paths device-guarded, verified on Colab.

## Authoritative curriculum

8-phase curriculum ported from `config/training_config.yaml` (phase0..phase7) into `config/phases.yaml` with named masks in `conditions_engine.MASK_REGISTRY`. Multi-timeframe indicators (CCI30/100 + shifted SMA, BB20/200, ATR14/45, SMA stacks, high/low SMA bands); CCI300/900 excluded. Mask semantics: force_in_and_gate / open_gate / free. Progressive cross-day reward (pass/ok/fail + streak + low-DD).

## Commands Executed

| # | Command | Result |
|---|---------|--------|
| 1 | `python scripts/validate_phases_yaml.py` | exit 0 — 9 phases PASS |
| 2 | `python scripts/smoke_train.py` | exit 0 — SMOKE_TRAIN OK |
| 3 | `python scripts/smoke_backtest.py` | exit 0 — SMOKE_BACKTEST OK |
| 4 | `python scripts/smoke_infer.py` | exit 0 — SMOKE_INFER OK |
| 5 | `python tests/run_all_tests.py` | exit 0 — 88 tests pass |
| 6 | `python inspect_system.py` | exit 0 — 0 failed, 1 skip (GPU) |

## Files Created

| File | Lines |
|------|-------|
| inspect_system.py | 217 |
| core/__init__.py | 1 |
| core/pipeline.py | 93 |
| core/settings.py | 108 |
| core/env/__init__.py | 1 |
| core/env/conditions_engine.py | 308 |
| core/env/environment.py | 347 |
| core/env/indicators.py | 282 |
| core/env/intrabar_fills.py | 105 |
| core/agent/__init__.py | 1 |
| core/agent/action_space.py | 121 |
| core/agent/dqn.py | 300 |
| core/reward/__init__.py | 1 |
| core/reward/shaper.py | 135 |
| core/risk/__init__.py | 1 |
| core/risk/daily_guard.py | 122 |
| core/risk/position_sizer.py | 48 |
| core/risk/trade_gate.py | 56 |
| training/__init__.py | 1 |
| training/checkpoint_manager.py | 203 |
| training/eval_loop.py | 73 |
| training/train.py | 177 |
| backtest/__init__.py | 1 |
| backtest/engine.py | 136 |
| broker/__init__.py | 1 |
| broker/account_manager.py | 33 |
| broker/broker_base.py | 22 |
| broker/live_runner.py | 92 |
| broker/mt5_adapter.py | 96 |
| broker/schwab_stub.py | 18 |
| broker/tradovate_stub.py | 18 |
| jordan/__init__.py | 1 |
| jordan/consent_flow.py | 60 |
| jordan/irac_engine.py | 73 |
| jordan/persona.py | 81 |
| jordan/policy_inspector.py | 56 |
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
| scripts/smoke_infer.py | 41 |
| scripts/smoke_train.py | 62 |
| scripts/validate_phases_yaml.py | 117 |
| tests/__init__.py | 1 |
| tests/conftest.py | 3 |
| tests/run_all_tests.py | 170 |
| tests/unit/__init__.py | 1 |
| tests/unit/test_action_space.py | 44 |
| tests/unit/test_checkpoint_manager.py | 73 |
| tests/unit/test_conditions_engine.py | 102 |
| tests/unit/test_daily_guard.py | 46 |
| tests/unit/test_dqn.py | 54 |
| tests/unit/test_environment.py | 63 |
| tests/unit/test_eval_loop.py | 20 |
| tests/unit/test_flatline_detector.py | 42 |
| tests/unit/test_indicators.py | 33 |
| tests/unit/test_indicators_multitf.py | 49 |
| tests/unit/test_intrabar_fills.py | 33 |
| tests/unit/test_jordan.py | 44 |
| tests/unit/test_known_bug_fixes.py | 98 |
| tests/unit/test_mt5_adapter.py | 41 |
| tests/unit/test_pipeline.py | 21 |
| tests/unit/test_position_sizer.py | 20 |
| tests/unit/test_reward_shaper.py | 66 |
| tests/unit/test_trade_gate.py | 31 |
| tests/integration/__init__.py | 1 |
| tests/integration/test_backtest_one_day.py | 15 |
| tests/integration/test_eval_loop.py | 17 |
| tests/integration/test_pipeline_parity.py | 22 |
| tests/integration/test_train_one_episode.py | 29 |
| tests/fixtures/__init__.py | 1 |
| tests/fixtures/sample_candles.py | 80 |
| tests/fixtures/sample_configs.py | 74 |
| tests/mocks/__init__.py | 1 |
| tests/mocks/mock_mt5.py | 55 |
| tests/mocks/mock_telegram.py | 18 |
| config/phases.yaml | 90 |
| config/training_config.yaml | 155 |
| config/trading_policy.yaml | 57 |
| config/jordan_sources.yaml | 18 |
| requirements.txt | 26 |
| .env.example | 15 |
| rl_trading_colab.ipynb | 201 |
| README.md | 184 |
| AUDIT.md | 88 |
| SPEC_STRATEGY.md | 94 |
| INTERFACES.md | 92 |

## Tests (88 functions) — all ✅ PASS

| Test File | Test |
|-----------|------|
| tests/integration/test_backtest_one_day.py | test_backtest_one_day |
| tests/integration/test_eval_loop.py | test_eval_loop_integration |
| tests/integration/test_pipeline_parity.py | test_indicators_bit_identical |
| tests/integration/test_pipeline_parity.py | test_parity_hashes_present |
| tests/integration/test_train_one_episode.py | test_train_one_episode |
| tests/unit/test_action_space.py | test_num_actions_is_756 |
| tests/unit/test_action_space.py | test_encode_decode_roundtrip_all_756 |
| tests/unit/test_action_space.py | test_decode_encode_roundtrip_all_ids |
| tests/unit/test_action_space.py | test_lot_resolution_and_clamp |
| tests/unit/test_action_space.py | test_sl_tp_tables |
| tests/unit/test_action_space.py | test_out_of_range_raises |
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
| tests/unit/test_conditions_engine.py | test_force_in_and_gate_forces_entry_when_flat |
| tests/unit/test_conditions_engine.py | test_force_in_and_gate_blocks_entries_when_condition_false |
| tests/unit/test_conditions_engine.py | test_open_gate_allows_all_when_true_hold_only_when_false |
| tests/unit/test_daily_guard.py | test_ftmo_halts_at_dd_threshold |
| tests/unit/test_daily_guard.py | test_ftmo_no_halt_within_limit |
| tests/unit/test_daily_guard.py | test_pass_recorded_when_target_hit |
| tests/unit/test_daily_guard.py | test_force_halt |
| tests/unit/test_daily_guard.py | test_beast_trailing_from_peak |
| tests/unit/test_daily_guard.py | test_trade_cap_halts |
| tests/unit/test_dqn.py | test_q_output_shape |
| tests/unit/test_dqn.py | test_save_load_roundtrip |
| tests/unit/test_dqn.py | test_transfer_learning_reinits_output |
| tests/unit/test_dqn.py | test_mask_blocks_actions |
| tests/unit/test_environment.py | test_state_shape |
| tests/unit/test_environment.py | test_step_contract |
| tests/unit/test_environment.py | test_action_mask_shape |
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
| tests/unit/test_known_bug_fixes.py | test_bug1_per_batch_independent_exploration |
| tests/unit/test_known_bug_fixes.py | test_bug2_env_returns_executed_actions |
| tests/unit/test_known_bug_fixes.py | test_bug3_bonus_backpatch_no_fake_transition |
| tests/unit/test_known_bug_fixes.py | test_bug5_checkpoint_roundtrip_with_metadata |
| tests/unit/test_known_bug_fixes.py | test_bug6_inference_uses_tiny_buffer |
| tests/unit/test_known_bug_fixes.py | test_bug8_replay_buffer_survives_save_load |
| tests/unit/test_known_bug_fixes.py | test_bug8_replay_skipped_on_transfer |
| tests/unit/test_mt5_adapter.py | test_symbol_alias_resolves |
| tests/unit/test_mt5_adapter.py | test_send_order_calls_gate_and_blocks |
| tests/unit/test_mt5_adapter.py | test_send_order_fills_when_approved |
| tests/unit/test_pipeline.py | test_returns_five_objects |
| tests/unit/test_position_sizer.py | test_clamped_to_max_lot |
| tests/unit/test_position_sizer.py | test_minimum_lot_floor |
| tests/unit/test_position_sizer.py | test_no_crash_on_any_bucket |
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

## Unverified (require Colab A100 / Drive / Windows MT5)

- ⚠️ torch.cuda / A100 detection (marked SKIP on CPU)
- ⚠️ Google Drive mount + real multi-symbol data files
- ⚠️ torch.compile + AMP speedups (CUDA-only; code device-guarded)
- ⚠️ talib (optional) — numpy fallback used when absent; same columns
- ⚠️ Real MT5 connection + live order execution (Windows)

## Failed Checks
- None.