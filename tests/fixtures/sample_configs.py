"""
tests/fixtures/sample_configs.py
────────────────────────────────────────────────────────────────────────────
Minimal valid phases.yaml + trading_policy.yaml content as Python dicts, for
tests that need configs without reading the real files. Mirrors the schema of
config/phases.yaml and config/trading_policy.yaml exactly.
"""
from __future__ import annotations


def minimal_phases() -> dict:
    """A minimal valid phases config: one gated phase + the infinite live phase."""
    return {
        "phases": [
            {
                "name": "ftmo_baseline",
                "order": 1,
                "instruments": ["EURUSD"],
                "entry_conditions": {
                    "buy": "cci_14 < -100 and close > sma_20",
                    "sell": "cci_14 > 100 and close < sma_20",
                },
                "max_episodes": 5,
                "advance_criteria": {"consecutive_pass_days": 2},
            },
            {
                "name": "live_improve",
                "order": 999,
                "instruments": ["EURUSD"],
                "entry_conditions": {"buy": "any", "sell": "any"},
                "max_episodes": -1,
                "advance_criteria": None,
            },
        ]
    }


def minimal_trading_policy() -> dict:
    """A minimal valid trading_policy config covering all required keys."""
    return {
        "mode": "ftmo",
        "ftmo_settings": {
            "daily_target_pct": 2.5,
            "daily_max_dd_pct": 1.0,
            "max_trades_per_day": 800,
        },
        "beast_settings": {
            "trailing_dd_from_peak_pct": 5.0,
            "profit_target_pct": 10.0,
            "max_trades_per_day": None,
        },
        "accounts": [
            {
                "id": "ftmo_primary",
                "login": "YOUR_MT5_LOGIN_HERE",
                "password": "YOUR_MT5_PASSWORD_HERE",
                "server": "YOUR_MT5_SERVER_HERE",
                "max_lot": 2.0,
                "mode": "ftmo",
                "instruments": ["EURUSD"],
                "symbol_aliases": {"EURUSD": ["EURUSD", "EURUSDm"]},
            }
        ],
        "instrument_settings": {
            "EURUSD": {"pip_value": 0.0001, "spread_pips": 1.0,
                       "slippage_pips": 0.5, "sl_buffer_pips": 2.0},
            "GBPUSD": {"pip_value": 0.0001, "spread_pips": 1.5,
                       "slippage_pips": 0.5, "sl_buffer_pips": 2.0},
            "XAUUSD": {"pip_value": 0.01, "spread_pips": 3.0,
                       "slippage_pips": 1.0, "sl_buffer_pips": 5.0},
            "US30": {"pip_value": 1.0, "spread_pips": 2.0,
                     "slippage_pips": 1.0, "sl_buffer_pips": 10.0},
        },
    }
