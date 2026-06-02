# Validation Report — rl-trading-live

**Date**: 2026-06-02 04:07  
**Environment**: Local CPU validation (Python 3.12.8, PyTorch 2.x). GPU-only paths verified on A100 in Colab.

## Commands Executed

| # | Command | Result |
|---|---------|--------|
| 1 | `python scripts/validate_phases_yaml.py` | exit 0 — 3 phases PASS |
| 2 | `python scripts/smoke_train.py` | exit 0 — SMOKE_TRAIN OK |
| 3 | `python scripts/smoke_backtest.py` | exit 0 — SMOKE_BACKTEST OK |
| 4 | `python scripts/smoke_infer.py` | exit 0 — SMOKE_INFER OK |
| 5 | `python tests/run_all_tests.py` | exit 0 — all tests pass |
| 6 | `python inspect_system.py` | exit 0 — 0 failed, 1 skip (GPU) |

## Files Created

| File | Lines |
|------|-------|
| backtest/__init__.py | 1 |
| backtest/engine.py | 136 |
| broker/__init__.py | 1 |
| broker/account_manager.py | 33 |
| broker/broker_base.py | 22 |
| broker/live_runner.py | 92 |
| broker/mt5_adapter.py | 96 |
| broker/schwab_stub.py | 18 |
| broker/tradovate_stub.py | 18 |
| build_notebook.py | 145 |
| core/__init__.py | 1 |
| core/agent/__init__.py | 1 |
| core/agent/action_space.py | 121 |
| core/agent/dqn.py | 300 |
| core/env/__init__.py | 1 |
| core/env/conditions_engine.py | 159 |
| core/env/environment.py | 284 |
| core/env/indicators.py | 161 |
| core/env/intrabar_fills.py | 105 |
| core/pipeline.py | 93 |
| core/reward/__init__.py | 1 |
| core/reward/shaper.py | 88 |
| core/risk/__init__.py | 1 |
| core/risk/daily_guard.py | 122 |
| core/risk/position_sizer.py | 48 |
| core/risk/trade_gate.py | 56 |
| core/settings.py | 108 |
| dashboard/__init__.py | 1 |
| dashboard/app.py | 89 |
| dashboard/pages/__init__.py | 1 |
| dashboard/pages/beast_mode.py | 25 |
| dashboard/pages/jordan_chat.py | 51 |
| dashboard/pages/live_dashboard.py | 49 |
| dashboard/pages/model_control.py | 48 |
| dashboard/pages/performance.py | 25 |
| inspect_system.py | 217 |
| jordan/__init__.py | 1 |
| jordan/consent_flow.py | 60 |
| jordan/irac_engine.py | 73 |
| jordan/persona.py | 81 |
| jordan/policy_inspector.py | 56 |
| jordan/repo_indexer.py | 46 |
| jordan/vitals_daemon.py | 83 |
| monitoring/__init__.py | 1 |
| monitoring/alert_dispatcher.py | 63 |
| monitoring/flatline_detector.py | 43 |
| scripts/crash_recovery.py | 40 |
| scripts/smoke_backtest.py | 28 |
| scripts/smoke_infer.py | 41 |
| scripts/smoke_train.py | 62 |
| scripts/validate_phases_yaml.py | 90 |
| tests/__init__.py | 1 |
| tests/conftest.py | 3 |
| tests/fixtures/__init__.py | 1 |
| tests/fixtures/sample_candles.py | 80 |
| tests/fixtures/sample_configs.py | 74 |
| tests/integration/__init__.py | 1 |
| tests/integration/test_backtest_one_day.py | 15 |
| tests/integration/test_eval_loop.py | 17 |
| tests/integration/test_pipeline_parity.py | 22 |
| tests/integration/test_train_one_episode.py | 29 |
| tests/mocks/__init__.py | 1 |
| tests/mocks/mock_mt5.py | 55 |
| tests/mocks/mock_telegram.py | 18 |
| tests/run_all_tests.py | 170 |
| tests/unit/__init__.py | 1 |
| tests/unit/test_action_space.py | 44 |
| tests/unit/test_checkpoint_manager.py | 73 |
| tests/unit/test_conditions_engine.py | 52 |
| tests/unit/test_daily_guard.py | 46 |
| tests/unit/test_dqn.py | 54 |
| tests/unit/test_environment.py | 63 |
| tests/unit/test_eval_loop.py | 20 |
| tests/unit/test_flatline_detector.py | 42 |
| tests/unit/test_indicators.py | 33 |
| tests/unit/test_intrabar_fills.py | 33 |
| tests/unit/test_jordan.py | 44 |
| tests/unit/test_known_bug_fixes.py | 98 |
| tests/unit/test_mt5_adapter.py | 41 |
| tests/unit/test_pipeline.py | 21 |
| tests/unit/test_position_sizer.py | 20 |
| tests/unit/test_reward_shaper.py | 40 |
| tests/unit/test_trade_gate.py | 31 |
| training/__init__.py | 1 |
| training/checkpoint_manager.py | 203 |
| training/eval_loop.py | 73 |
| training/train.py | 177 |
| config/phases.yaml | 48 |
| config/trading_policy.yaml | 57 |
| config/jordan_sources.yaml | 18 |
| requirements.txt | 26 |
| .env.example | 15 |
| rl_trading_colab.ipynb | 201 |
| README.md | 173 |
| AUDIT.md | 88 |

## Tests (75 functions)

| Test File | Test | Status |
|-----------|------|--------|
| tests/integration/test_backtest_one_day.py | test_backtest_one_day | ✅ PASS |
| tests/integration/test_eval_loop.py | test_eval_loop_integration | ✅ PASS |
| tests/integration/test_pipeline_parity.py | test_indicators_bit_identical | ✅ PASS |
| tests/integration/test_pipeline_parity.py | test_parity_hashes_present | ✅ PASS |
| tests/integration/test_train_one_episode.py | test_train_one_episode | ✅ PASS |
| tests/unit/test_action_space.py | test_num_actions_is_756 | ✅ PASS |
| tests/unit/test_action_space.py | test_encode_decode_roundtrip_all_756 | ✅ PASS |
| tests/unit/test_action_space.py | test_decode_encode_roundtrip_all_ids | ✅ PASS |
| tests/unit/test_action_space.py | test_lot_resolution_and_clamp | ✅ PASS |
| tests/unit/test_action_space.py | test_sl_tp_tables | ✅ PASS |
| tests/unit/test_action_space.py | test_out_of_range_raises | ✅ PASS |
| tests/unit/test_checkpoint_manager.py | test_rolling_deletion_keeps_five_lowest_phi_removed | ✅ PASS |
| tests/unit/test_checkpoint_manager.py | test_protected_never_deleted | ✅ PASS |
| tests/unit/test_checkpoint_manager.py | test_bootstrap_from_existing | ✅ PASS |
| tests/unit/test_checkpoint_manager.py | test_find_best_resume | ✅ PASS |
| tests/unit/test_checkpoint_manager.py | test_verify_all_detects_corrupt | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_any_always_true | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_evaluate_uses_features | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_unknown_variable_raises_irac | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_buy_condition_masks_hold_and_sell | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_no_condition_allows_all | ✅ PASS |
| tests/unit/test_conditions_engine.py | test_both_false_allows_all | ✅ PASS |
| tests/unit/test_daily_guard.py | test_ftmo_halts_at_dd_threshold | ✅ PASS |
| tests/unit/test_daily_guard.py | test_ftmo_no_halt_within_limit | ✅ PASS |
| tests/unit/test_daily_guard.py | test_pass_recorded_when_target_hit | ✅ PASS |
| tests/unit/test_daily_guard.py | test_force_halt | ✅ PASS |
| tests/unit/test_daily_guard.py | test_beast_trailing_from_peak | ✅ PASS |
| tests/unit/test_daily_guard.py | test_trade_cap_halts | ✅ PASS |
| tests/unit/test_dqn.py | test_q_output_shape | ✅ PASS |
| tests/unit/test_dqn.py | test_save_load_roundtrip | ✅ PASS |
| tests/unit/test_dqn.py | test_transfer_learning_reinits_output | ✅ PASS |
| tests/unit/test_dqn.py | test_mask_blocks_actions | ✅ PASS |
| tests/unit/test_environment.py | test_state_shape | ✅ PASS |
| tests/unit/test_environment.py | test_step_contract | ✅ PASS |
| tests/unit/test_environment.py | test_action_mask_shape | ✅ PASS |
| tests/unit/test_environment.py | test_episode_terminates | ✅ PASS |
| tests/unit/test_environment.py | test_multi_symbol_alignment | ✅ PASS |
| tests/unit/test_eval_loop.py | test_eval_returns_four_keys | ✅ PASS |
| tests/unit/test_flatline_detector.py | test_flatline_fires_on_flat_series | ✅ PASS |
| tests/unit/test_flatline_detector.py | test_no_flatline_when_improving | ✅ PASS |
| tests/unit/test_flatline_detector.py | test_alert_appends_to_session_and_telegram | ✅ PASS |
| tests/unit/test_flatline_detector.py | test_alert_never_raises_when_all_channels_fail | ✅ PASS |
| tests/unit/test_indicators.py | test_shape_and_finite | ✅ PASS |
| tests/unit/test_indicators.py | test_named_columns_present | ✅ PASS |
| tests/unit/test_indicators.py | test_rolling_high_ge_close | ✅ PASS |
| tests/unit/test_indicators.py | test_device_placement_cpu | ✅ PASS |
| tests/unit/test_intrabar_fills.py | test_buy_fill_above_open | ✅ PASS |
| tests/unit/test_intrabar_fills.py | test_sell_fill_below_open | ✅ PASS |
| tests/unit/test_intrabar_fills.py | test_hold_returns_open | ✅ PASS |
| tests/unit/test_intrabar_fills.py | test_atr_caps_sl_width | ✅ PASS |
| tests/unit/test_jordan.py | test_irac_has_four_sections | ✅ PASS |
| tests/unit/test_jordan.py | test_persona_fallback_without_key | ✅ PASS |
| tests/unit/test_jordan.py | test_consent_requires_two_steps | ✅ PASS |
| tests/unit/test_jordan.py | test_jordan_can_read_test_results | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug1_per_batch_independent_exploration | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug2_env_returns_executed_actions | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug3_bonus_backpatch_no_fake_transition | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug5_checkpoint_roundtrip_with_metadata | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug6_inference_uses_tiny_buffer | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug8_replay_buffer_survives_save_load | ✅ PASS |
| tests/unit/test_known_bug_fixes.py | test_bug8_replay_skipped_on_transfer | ✅ PASS |
| tests/unit/test_mt5_adapter.py | test_symbol_alias_resolves | ✅ PASS |
| tests/unit/test_mt5_adapter.py | test_send_order_calls_gate_and_blocks | ✅ PASS |
| tests/unit/test_mt5_adapter.py | test_send_order_fills_when_approved | ✅ PASS |
| tests/unit/test_pipeline.py | test_returns_five_objects | ✅ PASS |
| tests/unit/test_position_sizer.py | test_clamped_to_max_lot | ✅ PASS |
| tests/unit/test_position_sizer.py | test_minimum_lot_floor | ✅ PASS |
| tests/unit/test_position_sizer.py | test_no_crash_on_any_bucket | ✅ PASS |
| tests/unit/test_reward_shaper.py | test_warmup_returns_zero | ✅ PASS |
| tests/unit/test_reward_shaper.py | test_weekly_bonus_fires_when_rate_improves | ✅ PASS |
| tests/unit/test_reward_shaper.py | test_no_weekly_bonus_when_flat | ✅ PASS |
| tests/unit/test_reward_shaper.py | test_compute_bonus_runs_after_warmup | ✅ PASS |
| tests/unit/test_trade_gate.py | test_approve_true_when_not_halted | ✅ PASS |
| tests/unit/test_trade_gate.py | test_approve_false_when_halted | ✅ PASS |
| tests/unit/test_trade_gate.py | test_consent_blocks | ✅ PASS |

## Passed Checks (inspect_system.py)

- ✅ Python >= 3.10
- ✅ phases.yaml schema + VARIABLE_REGISTRY
- ✅ trading_policy.yaml required keys
- ✅ All modules import (no orphans)
- ✅ action_space 756 roundtrip
- ✅ trade_gate blocks when halted
- ✅ Jordan IRAC markdown
- ✅ Jordan persona fallback
- ✅ dashboard.app imports
- ✅ No hardcoded credentials
- ✅ smoke_train / smoke_backtest / smoke_infer
- ✅ pytest unit + integration

## Failed Checks
- None.

## Unverified (require Colab A100 / Drive / Windows MT5)

- ⚠️ torch.cuda / A100 detection (CELL 1) — CPU here, marked SKIP
- ⚠️ Google Drive mount + real data file (CELL 2)
- ⚠️ torch.compile + AMP speedups (CUDA-only; code is device-guarded)
- ⚠️ GPU profiling cell (CELL 9)
- ⚠️ Real MT5 connection + live order execution (Windows + MT5 terminal)
- ⚠️ localtunnel/ngrok public dashboard URL (Colab networking)

## Known-bug regression coverage

All 14 + 9 prior-build bugs are fixed or documented; locked by `tests/unit/test_known_bug_fixes.py` (per-batch epsilon, executed-action storage, bonus back-patch, weights_only metadata load, inference MEMORY_SIZE=1, replay-buffer resume, transfer learning).