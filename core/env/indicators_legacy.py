"""
core/env/indicators_legacy.py
──────────────────────────────────────────────────────────────────────────────
VERBATIM port of the predecessor repo's feature builder used by the DQN that
produced eurusd_gpu_ph0_ep0120.pt. Provenance:

  monty313/deep-reinforcement-learning-trading
  gpu_rl_trading/env/indicators.py

This file is the SINGLE SOURCE OF TRUTH for the 27-feature schema the DQN
checkpoint expects. Do NOT replace these implementations with talib variants
— numerical differences (Wilder vs simple smoothing, etc.) would break the
DQN's input distribution and corrupt the teacher signal.

Kept pure-numpy on purpose for byte-identical parity with the DQN training.
"""
from __future__ import annotations
import numpy as np


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    cs  = np.cumsum(x)
    out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float32)
    if n <= 0:
        return out
    if len(x) < n:
        return out
    try:
        # numpy>=1.20 has sliding_window_view which enables a vectorized window
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(x, window_shape=n)
        stds = windows.std(axis=1)
        out[n-1:] = stds.astype(np.float32)
        return out
    except Exception:
        # Fallback for older numpy versions: safe Python loop
        for i in range(n-1, len(x)):
            out[i] = x[i-n+1:i+1].std()
        return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling_mean(x, n)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float32)
    k   = 2.0 / (n + 1)
    # find first valid
    start = n - 1
    if start >= len(x):
        return out
    out[start] = x[:start+1].mean()
    for i in range(start+1, len(x)):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return ema(tr.astype(np.float32), n)


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0).astype(np.float32)
    loss  = np.where(delta < 0, -delta, 0.0).astype(np.float32)
    avg_gain = ema(gain, n)
    avg_loss = ema(loss, n)
    rs  = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    return 100.0 - (100.0 / (1.0 + rs))


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 20) -> np.ndarray:
    tp   = (high + low + close) / 3.0
    sma_ = _rolling_mean(tp, n)
    out  = np.full_like(tp, np.nan)
    for i in range(n-1, len(tp)):
        window = tp[i-n+1:i+1]
        md     = np.mean(np.abs(window - window.mean()))
        out[i] = (tp[i] - sma_[i]) / (0.015 * md + 1e-10)
    return out.astype(np.float32)


def bb_bands(close: np.ndarray, n: int = 20, nbdev: float = 1.0):
    """Returns (upper, middle, lower) as float32 arrays."""
    mid   = _rolling_mean(close, n)
    std   = _rolling_std(close, n)
    upper = (mid + nbdev * std).astype(np.float32)
    lower = (mid - nbdev * std).astype(np.float32)
    return upper, mid.astype(np.float32), lower


def build_feature_matrix(
    open_: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Build a (T, F) float32 feature matrix from raw OHLCV arrays.
    Columns (27 features):
      0  open, 1  high, 2  low, 3  close, 4  volume
      5  atr14, 6  atr45
      7  rsi7,  8  rsi14
      9  cci14, 10 cci30, 11 cci100
      12 sma20, 13 sma50, 14 sma200
      15 bb20_upper, 16 bb20_mid, 17 bb20_lower
      18 bb200_upper,19 bb200_mid,20 bb200_lower
      21 high_sma4_sh8 (SMA(4,shift=8) of high)
      22 low_sma4_sh8  (SMA(4,shift=8) of low)
      23 sma2_sh0, 24 sma2_sh1, 25 sma2_sh2
      26 atr14_sma1_sh8
    """
    c  = close.astype(np.float32)
    h  = high.astype(np.float32)
    l  = low.astype(np.float32)
    o  = open_.astype(np.float32)
    v  = volume.astype(np.float32)

    atr14  = atr(h, l, c, 14)
    atr45  = atr(h, l, c, 45)
    rsi7_  = rsi(c, 7)
    rsi14_ = rsi(c, 14)
    cci14_ = cci(h, l, c, 14)
    cci30_ = cci(h, l, c, 30)
    cci100_= cci(h, l, c, 100)
    sma20_ = sma(c, 20)
    sma50_ = sma(c, 50)
    sma200_= sma(c, 200)
    bb20u, bb20m, bb20l     = bb_bands(c, 20, 1.0)
    bb200u, bb200m, bb200l  = bb_bands(c, 200, 1.0)

    def _sma_shift(arr, n, shift):
        s = sma(arr, n)
        return np.roll(s, shift)

    h_sma4_sh8  = _sma_shift(h, 4, 8)
    l_sma4_sh8  = _sma_shift(l, 4, 8)
    sma2_sh0    = sma(c, 2)
    sma2_sh1    = np.roll(sma2_sh0, 1)
    sma2_sh2    = np.roll(sma2_sh0, 2)
    atr14_sh8   = np.roll(atr14, 8)

    mat = np.column_stack([
        o, h, l, c, v,
        atr14, atr45,
        rsi7_, rsi14_,
        cci14_, cci30_, cci100_,
        sma20_, sma50_, sma200_,
        bb20u, bb20m, bb20l,
        bb200u, bb200m, bb200l,
        h_sma4_sh8, l_sma4_sh8,
        sma2_sh0, sma2_sh1, sma2_sh2,
        atr14_sh8,
    ]).astype(np.float32)

    # NaN -> 0 (warm-up rows)
    np.nan_to_num(mat, copy=False, nan=0.0)
    return mat
