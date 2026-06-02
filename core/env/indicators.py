"""
core/env/indicators.py
────────────────────────────────────────────────────────────────────────────
GPU-tensor feature-matrix builder. Ported from gpu_rl_trading/env/indicators.py
(REPO1, numpy) and rewritten to run as torch ops on the active device so the
SAME code path runs on CPU (dev/CI) and CUDA (Colab A100).

PARITY (HARD RULE 10): this file's md5 is recorded in the checkpoint manifest.
Training, backtest, and live_runner all import THIS function, so feature values
are bit-identical across all three. Do not fork the math anywhere else.

OUTPUT: build_feature_matrix(...) -> torch.Tensor of shape (N, F) on `device`,
with named columns exposed via FEATURE_COLUMNS / COL. The named columns are the
ones the phases.yaml VARIABLE_REGISTRY can reference:

    close, open, high, low, volume,
    sma_20, ema_20, cci_14, atr_14, atr_14_ma, rolling_high_20, rolling_low_20

All warmup rows (before an indicator is defined) are back-filled with the first
valid value so the matrix contains no NaN/inf (asserted in unit tests).
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

# ── Named feature columns (order defines the matrix layout) ──────────────────
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_20", "ema_20", "cci_14", "atr_14", "atr_14_ma",
    "rolling_high_20", "rolling_low_20",
]
COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
NUM_FEATURES = len(FEATURE_COLUMNS)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(x: ArrayLike, device: torch.device) -> torch.Tensor:
    """Coerce np array or tensor to a 1-D float32 tensor on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).reshape(-1)


def _rolling_mean(x: torch.Tensor, n: int) -> torch.Tensor:
    """Causal rolling mean over window n. Warmup rows = first valid value."""
    if n <= 1:
        return x.clone()
    csum = torch.cumsum(x, dim=0)
    out = torch.empty_like(x)
    out[n - 1:] = (csum[n - 1:] - torch.cat([csum.new_zeros(1), csum[:-n]])) / n
    out[:n - 1] = out[n - 1]                      # back-fill warmup
    return out


def _rolling_max(x: torch.Tensor, n: int) -> torch.Tensor:
    """Causal rolling max over window n (via unfold). Warmup back-filled."""
    if n <= 1:
        return x.clone()
    pad = x[:1].expand(n - 1)
    xp = torch.cat([pad, x])                       # left-pad so output aligns
    return xp.unfold(0, n, 1).max(dim=1).values


def _rolling_min(x: torch.Tensor, n: int) -> torch.Tensor:
    """Causal rolling min over window n (via unfold). Warmup back-filled."""
    if n <= 1:
        return x.clone()
    pad = x[:1].expand(n - 1)
    xp = torch.cat([pad, x])
    return xp.unfold(0, n, 1).min(dim=1).values


def _ema(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Exponential moving average. Seeded with the SMA of the first n values, then
    recursively smoothed. A short Python loop over time is unavoidable for EMA;
    it runs once at env init (not in the hot training loop), so it is not a
    GPU bottleneck.
    """
    out = torch.empty_like(x)
    k = 2.0 / (n + 1)
    seed = x[:n].mean()
    out[:n] = seed
    prev = seed
    for i in range(n, x.shape[0]):
        prev = x[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def _cci(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, n: int = 14
         ) -> torch.Tensor:
    """
    Commodity Channel Index. CCI = (TP - SMA(TP)) / (0.015 * mean_dev(TP)).
    TP = typical price = (H+L+C)/3. Matches the sample data's CCI scaling.
    """
    tp = (high + low + close) / 3.0
    sma_tp = _rolling_mean(tp, n)
    # mean absolute deviation over the window
    pad = tp[:1].expand(n - 1)
    tpp = torch.cat([pad, tp])
    windows = tpp.unfold(0, n, 1)                  # (N, n)
    mean_dev = (windows - windows.mean(dim=1, keepdim=True)).abs().mean(dim=1)
    mean_dev = mean_dev.clamp(min=1e-10)
    return (tp - sma_tp) / (0.015 * mean_dev)


def _atr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, n: int = 14
         ) -> torch.Tensor:
    """Average True Range over window n. TR = max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = torch.cat([close[:1], close[:-1]])
    tr = torch.maximum(high - low,
                       torch.maximum((high - prev_close).abs(),
                                     (low - prev_close).abs()))
    return _rolling_mean(tr, n)


def build_feature_matrix(open_, high, low, close, volume, device: torch.device
                         ) -> torch.Tensor:
    """
    Build the (N, NUM_FEATURES) feature tensor on `device`.

    Args accept np arrays or tensors (1-D, equal length). Returns a float32
    tensor on `device` with columns ordered per FEATURE_COLUMNS, free of NaN/inf.
    """
    o = _to_tensor(open_, device)
    h = _to_tensor(high, device)
    l = _to_tensor(low, device)
    c = _to_tensor(close, device)
    v = _to_tensor(volume, device)

    sma_20 = _rolling_mean(c, 20)
    ema_20 = _ema(c, 20)
    cci_14 = _cci(h, l, c, 14)
    atr_14 = _atr(h, l, c, 14)
    atr_14_ma = _rolling_mean(atr_14, 20)          # 20-bar MA of ATR_14
    roll_hi_20 = _rolling_max(h, 20)
    roll_lo_20 = _rolling_min(l, 20)

    mat = torch.stack([
        o, h, l, c, v,
        sma_20, ema_20, cci_14, atr_14, atr_14_ma,
        roll_hi_20, roll_lo_20,
    ], dim=1)

    # Guard: replace any residual NaN/inf with column-safe zeros (defensive).
    mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat.to(device=device, dtype=torch.float32)


def feature_row_dict(row: torch.Tensor) -> dict:
    """
    Convert a single feature row (length NUM_FEATURES tensor) to a name->float
    dict for conditions_engine.evaluate(). Moves to CPU once (display/eval only).
    """
    vals = row.detach().cpu().reshape(-1).tolist()
    return {name: float(vals[i]) for name, i in COL.items()}
