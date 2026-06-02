"""
tests/fixtures/sample_candles.py
────────────────────────────────────────────────────────────────────────────
Synthetic M1 OHLCV candle data for tests, smoke runs, and any time Google Drive
is not mounted. Deterministic (seeded) so tests are reproducible.

Used by:
  - scripts/smoke_train.py / smoke_backtest.py / smoke_infer.py
  - tests/unit/* and tests/integration/*
  - core/pipeline.py fallback when Drive data is unavailable

The generator produces a gently trending random walk with realistic intrabar
high/low spread, so indicators (SMA/EMA/CCI/ATR) and breakout conditions all
have something meaningful to compute on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_candles(n: int = 500, seed: int = 42, start_price: float = 1.1750
                           ) -> pd.DataFrame:
    """
    Return a DataFrame with columns: time, open, high, low, close,
    tick_volume, spread, real_volume — matching the user's EURUSD_features.csv
    schema (OHLC + tick_volume) so the loader path is identical.

    Args:
        n           : number of 1-minute bars
        seed        : RNG seed for reproducibility
        start_price : starting mid price
    """
    rng = np.random.default_rng(seed)
    # Random-walk close with a mild drift and occasional regime shifts.
    steps = rng.normal(0.0, 0.00015, size=n)
    # inject a few trend bursts so breakout conditions can trigger
    for burst_start in range(50, n, 120):
        steps[burst_start:burst_start + 20] += 0.00010
    close = start_price + np.cumsum(steps)

    # Build OHLC around the close path.
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 0.00012, size=n))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    tick_volume = rng.integers(20, 200, size=n)
    spread = np.full(n, 12, dtype=int)        # points (matches sample data)
    real_volume = np.zeros(n, dtype=int)

    times = pd.date_range("2025-01-01 00:00:00", periods=n, freq="1min")

    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": tick_volume,
        "spread": spread,
        "real_volume": real_volume,
    })


def make_synthetic_ohlcv_array(n: int = 500, seed: int = 42) -> np.ndarray:
    """
    Return an (n, 5) float32 array of [open, high, low, close, volume] — the
    raw matrix the environment and indicators consume directly.
    """
    df = make_synthetic_candles(n=n, seed=seed)
    return df[["open", "high", "low", "close", "tick_volume"]].to_numpy(dtype=np.float32)


def write_synthetic_csv(path: str, n: int = 500, seed: int = 42) -> str:
    """Write a synthetic candle CSV to `path` (used by smoke_train). Returns path."""
    df = make_synthetic_candles(n=n, seed=seed)
    df.to_csv(path, index=False)
    return path
