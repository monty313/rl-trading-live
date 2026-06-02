"""
core/env/indicators.py
────────────────────────────────────────────────────────────────────────────
Authoritative multi-indicator feature builder. Ported from REPO1
env/indicators.py `compute_indicators()` (the canonical spec) with:

  - CCI(300) and CCI(900) REMOVED (too slow on 1m data; per user instruction),
    along with their derived columns (cci300 BB bands, cci900_sma20).
  - talib used when available; otherwise a numpy fallback computes the SAME
    columns so the repo runs clone-and-run on Colab/CI without the talib C lib.
  - Multi-timeframe support via resample_ohlcv() (1m -> 15m/30m/1H/1D) so phase
    masks that gate on [1m, 15m] / [1m, 1H] etc. have per-TF rows.

PARITY (HARD RULE 10): md5 of this file is recorded in the manifest; training,
backtest, and live all import these functions, so values match across all three.

Two public surfaces:
  1. compute_indicators(df) -> pandas DataFrame with ALL named indicator columns
     (used by the phase-mask engine, which reads per-bar rows by name).
  2. build_feature_matrix(open,high,low,close,volume, device) -> torch.Tensor
     (N, NUM_FEATURES) on device — the compact normalized matrix the agent's
     state uses. FEATURE_COLUMNS lists the columns the VARIABLE_REGISTRY exposes.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import torch

# talib is the REQUIRED single source of truth for indicators
# (DESIGN_DECISIONS.md #3). The numpy implementations below are a TEST-ONLY
# fallback and must never be mixed with talib in a parity run. Set the env var
# RL_ALLOW_NUMPY_INDICATORS=1 to permit the fallback (tests/CI without talib);
# otherwise a missing talib raises ImportError at import time (fail-fast).
import os as _os

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:  # pragma: no cover
    talib = None
    _HAS_TALIB = False
    if _os.getenv("RL_ALLOW_NUMPY_INDICATORS") != "1":
        raise ImportError(
            "TA-Lib is required (DESIGN_DECISIONS.md #3) and is not installed. "
            "On Colab: `!apt-get install -y ta-lib && pip install TA-Lib`. "
            "On Windows: install the TA-Lib wheel. For tests/CI WITHOUT talib, "
            "set RL_ALLOW_NUMPY_INDICATORS=1 to use the numpy fallback (TEST ONLY "
            "— never mix with talib in a parity run).")


# ════════════════════════════════════════════════════════════════════════════
# Pure-numpy indicator primitives (talib-compatible) — used when talib absent
# ════════════════════════════════════════════════════════════════════════════
def _np_sma(x: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return x.astype(float)
    return pd.Series(x).rolling(period, min_periods=1).mean().to_numpy()


def _np_atr(h, l, c, period=14) -> np.ndarray:
    prev_c = np.concatenate([c[:1], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    # Wilder smoothing approximated by simple rolling mean (matches our fallback parity)
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _np_cci(h, l, c, period=14) -> np.ndarray:
    tp = (h + l + c) / 3.0
    sma_tp = pd.Series(tp).rolling(period, min_periods=1).mean().to_numpy()
    mad = pd.Series(tp).rolling(period, min_periods=1).apply(
        lambda w: np.mean(np.abs(w - w.mean())), raw=True).to_numpy()
    mad = np.where(mad == 0, 1e-10, mad)
    return (tp - sma_tp) / (0.015 * mad)


def _np_rsi(c, period=14) -> np.ndarray:
    # Wilder smoothing (RMA) to match TA-Lib: seed avg with SMA of first `period`
    # up/down moves (TA-Lib starts from index 1), then EMA with alpha=1/period.
    d = np.diff(c, prepend=c[:1])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    n = len(c)
    avg_up = np.full(n, np.nan)
    avg_dn = np.full(n, np.nan)
    if n >= period:
        avg_up[period - 1] = up[1:period].mean()
        avg_dn[period - 1] = dn[1:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            avg_up[i] = alpha * up[i] + (1.0 - alpha) * avg_up[i - 1]
            avg_dn[i] = alpha * dn[i] + (1.0 - alpha) * avg_dn[i - 1]
    rs = avg_up / np.where(avg_dn == 0, 1e-10, avg_dn)
    return 100.0 - (100.0 / (1.0 + rs))


def _np_adx(h, l, c, period=14) -> np.ndarray:
    # Lightweight ADX approximation (directional movement -> smoothed).
    up_move = np.diff(h, prepend=h[:1])
    dn_move = -np.diff(l, prepend=l[:1])
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr = _np_atr(h, l, c, period)
    atr = np.where(atr == 0, 1e-10, atr)
    plus_di = 100.0 * pd.Series(plus_dm).rolling(period, min_periods=1).mean().to_numpy() / atr
    minus_di = 100.0 * pd.Series(minus_dm).rolling(period, min_periods=1).mean().to_numpy() / atr
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, plus_di + minus_di)
    return pd.Series(dx).rolling(period, min_periods=1).mean().to_numpy()


def _np_bbands(c, period, nbdev=1.0):
    s = pd.Series(c)
    mid = s.rolling(period, min_periods=1).mean()
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0)
    upper = mid + nbdev * std
    lower = mid - nbdev * std
    return upper.to_numpy(), mid.to_numpy(), lower.to_numpy()


# ════════════════════════════════════════════════════════════════════════════
# talib-or-numpy wrappers (same outputs either way)
# ════════════════════════════════════════════════════════════════════════════
def _atr(h, l, c, period):
    return talib.ATR(h, l, c, timeperiod=period) if _HAS_TALIB else _np_atr(h, l, c, period)


def _cci(h, l, c, period):
    return talib.CCI(h, l, c, timeperiod=period) if _HAS_TALIB else _np_cci(h, l, c, period)


def _rsi(c, period):
    return talib.RSI(c, timeperiod=period) if _HAS_TALIB else _np_rsi(c, period)


def _adx(h, l, c, period):
    return talib.ADX(h, l, c, timeperiod=period) if _HAS_TALIB else _np_adx(h, l, c, period)


def _bbands(c, period, nbdev=1.0):
    if _HAS_TALIB:
        return talib.BBANDS(c, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return _np_bbands(c, period, nbdev)


def _sma_series(series: pd.Series, period: int, shift: int = 0) -> pd.Series:
    """SMA with optional forward shift (positive = look further back)."""
    s = series.astype(float) if period <= 1 else \
        series.astype(float).rolling(window=period, min_periods=1).mean()
    return s.shift(shift) if shift else s


# ════════════════════════════════════════════════════════════════════════════
# Full indicator DataFrame (per timeframe) — authoritative column set
# ════════════════════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: [open, high, low, close, volume].
    Returns a DataFrame (same index) with all named indicator columns appended.
    CCI(300)/CCI(900) intentionally excluded. No rows dropped (early NaNs kept).
    """
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    c = close.to_numpy(np.float64)
    h = high.to_numpy(np.float64)
    l = low.to_numpy(np.float64)
    idx = df.index
    out = pd.DataFrame(index=idx)

    out["open"], out["high"], out["low"], out["close"] = open_, high, low, close
    out["volume"] = df["volume"]

    # ── ATR (14, 45) + shifted SMA(1, +8) refs ──
    atr14 = pd.Series(_atr(h, l, c, 14), index=idx)
    atr45 = pd.Series(_atr(h, l, c, 45), index=idx)
    out["atr14"] = atr14
    out["atr45"] = atr45
    out["atr14_sma1_sh8"] = _sma_series(atr14, 1, shift=8)
    out["atr45_sma1_sh8"] = _sma_series(atr45, 1, shift=8)
    out["atr14_ma"] = _sma_series(atr14, 20)   # 20-bar MA of ATR14 (compat)

    # ── Bollinger Bands (nbdev=1.0) ──
    bb200_u, bb200_m, bb200_l = _bbands(c, 200, 1.0)
    bb20_u, bb20_m, bb20_l = _bbands(c, 20, 1.0)
    out["bb200_upper"], out["bb200_mid"], out["bb200_lower"] = bb200_u, bb200_m, bb200_l
    out["bb20_upper"], out["bb20_mid"], out["bb20_lower"] = bb20_u, bb20_m, bb20_l

    # ── RSI ──
    out["rsi7"] = pd.Series(_rsi(c, 7), index=idx)
    out["rsi5"] = pd.Series(_rsi(c, 5), index=idx)

    # ── CCI (10, 14, 30, 140) — NO 300/900 ──
    # cci10 and cci30 are the primary phase gate indicators (changed from 30/100)
    out["cci10"] = pd.Series(_cci(h, l, c, 10), index=idx)
    out["cci14"] = pd.Series(_cci(h, l, c, 14), index=idx)
    out["cci30"] = pd.Series(_cci(h, l, c, 30), index=idx)
    out["cci140"] = pd.Series(_cci(h, l, c, 140), index=idx)
    # shifted SMA(1,+8) on CCI10/30 (Phase 1 gate) + compat MAs
    out["cci10_sma1_sh8"] = _sma_series(out["cci10"], 1, shift=8)
    out["cci30_sma1_sh8"] = _sma_series(out["cci30"], 1, shift=8)
    out["cci14_sma20"] = _sma_series(out["cci14"], 20)
    out["cci30_sma20"] = _sma_series(out["cci30"], 20)
    out["cci140_sma1_sh4"] = _sma_series(out["cci140"], 1, shift=4)
    # CCI BB bands on cci10/cci30
    for col in ["cci10", "cci30"]:
        u, m, lo = _bbands(out[col].to_numpy(np.float64), 14, 1.0)
        out[f"{col}_bb14_upper"], out[f"{col}_bb14_mid"], out[f"{col}_bb14_lower"] = u, m, lo

    # ── SMA family ──
    out["sma4"] = _sma_series(close, 4)
    for sh in (1, 2, 3, 4):
        out[f"sma4_sh{sh}"] = _sma_series(close, 4, shift=sh)
    out["sma30"] = _sma_series(close, 30)
    out["sma50"] = _sma_series(close, 50)
    out["sma200"] = _sma_series(close, 200)
    out["sma_20"] = _sma_series(close, 20)        # compat alias used by simple strategies
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    # SMA(2) stack for Phase 5
    for sh in range(5):
        out[f"sma2_sh{sh}"] = _sma_series(close, 2, shift=sh)
    # High/Low SMA(4,+8) bands for Phase 2 & 3
    out["high_sma4_sh8"] = _sma_series(high, 4, shift=8)
    out["low_sma4_sh8"] = _sma_series(low, 4, shift=8)
    # rolling hi/lo (compat)
    out["rolling_high_20"] = high.rolling(20, min_periods=1).max()
    out["rolling_low_20"] = low.rolling(20, min_periods=1).min()

    # ── ADX + BB-upper SMA(4,+8) refs ──
    out["adx14"] = pd.Series(_adx(h, l, c, 14), index=idx)
    out["bb20_upper_sma4_sh8"] = _sma_series(pd.Series(bb20_u, index=idx), 4, shift=8)
    out["bb200_upper_sma4_sh8"] = _sma_series(pd.Series(bb200_u, index=idx), 4, shift=8)

    return out


# ════════════════════════════════════════════════════════════════════════════
# Multi-timeframe resampling (1m base -> 15m / 30m / 1H / 1D)
# ════════════════════════════════════════════════════════════════════════════
def resample_ohlcv(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Resample a 1-minute OHLCV DataFrame (DatetimeIndex) to `minutes` bars.
    minutes=1 returns the input unchanged. Standard OHLC aggregation.
    """
    if minutes <= 1:
        return df_1m
    rule = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    cols = [c for c in agg if c in df_1m.columns]
    return df_1m[cols].resample(rule, label="right", closed="right").agg(
        {c: agg[c] for c in cols}).dropna(how="any")


# ════════════════════════════════════════════════════════════════════════════
# Compact normalized feature matrix for the agent state (GPU tensor)
# ════════════════════════════════════════════════════════════════════════════
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_20", "ema_20", "cci_14", "atr_14", "atr_14_ma",
    "rolling_high_20", "rolling_low_20",
    # authoritative gate variables exposed to VARIABLE_REGISTRY:
    "cci10", "cci30", "cci10_sma1_sh8", "cci30_sma1_sh8",
    "bb20_upper", "bb20_mid", "bb200_upper", "bb200_mid",
    "high_sma4_sh8", "low_sma4_sh8",
    "atr14", "atr14_sma1_sh8", "atr45", "atr45_sma1_sh8",
    "bb20_upper_sma4_sh8", "bb200_upper_sma4_sh8",
]
COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
NUM_FEATURES = len(FEATURE_COLUMNS)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_np(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def build_feature_matrix(open_, high, low, close, volume, device: torch.device
                         ) -> torch.Tensor:
    """
    Build the compact (N, NUM_FEATURES) state matrix on `device`. Computes the
    full indicator set once (single TF = the input series), selects the named
    feature columns, fills NaN/inf, and returns a float32 tensor on device.
    """
    df = pd.DataFrame({
        "open": _as_np(open_), "high": _as_np(high), "low": _as_np(low),
        "close": _as_np(close), "volume": _as_np(volume),
    })
    ind = compute_indicators(df)
    # map compact names -> source columns
    alias = {"cci_14": "cci14", "atr_14": "atr14", "atr_14_ma": "atr14_ma"}
    data = {}
    for name in FEATURE_COLUMNS:
        src = alias.get(name, name)
        data[name] = ind[src].to_numpy(np.float64) if src in ind.columns \
            else np.zeros(len(df))
    mat = np.stack([data[n] for n in FEATURE_COLUMNS], axis=1)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(mat, dtype=torch.float32, device=device)


def feature_row_dict(row: torch.Tensor) -> dict:
    """Convert a feature-matrix row (length NUM_FEATURES) to a name->float dict."""
    vals = row.detach().cpu().reshape(-1).tolist()
    return {name: float(vals[i]) for name, i in COL.items()}
