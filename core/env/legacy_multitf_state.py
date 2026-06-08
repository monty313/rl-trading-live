"""
core/env/legacy_multitf_state.py
──────────────────────────────────────────────────────────────────────────────
Multi-timeframe state provider that PRODUCES THE EXACT 2166-DIM OBSERVATION
THE DQN CHECKPOINT EXPECTS. Single source of truth for the legacy obs schema:

    state_dim = LOOKBACK (20) × NUM_FEATURES (27) × len(TF_FACTORS [1,15,60,1440])
              + 6 trailing position/FTMO features
              = 2160 + 6
              = 2166

Provenance: monty313/deep-reinforcement-learning-trading
            gpu_rl_trading/env/environment.py::_get_state()

This module exists so:
  1. PPO can SEE multi-timeframe context (1m + 15m + 1h + 1d) — solving the
     single-TF momentum blindness that the current 620-dim obs has.
  2. The 1.74 GB DQN checkpoint (eurusd_gpu_ph0_ep0120.pt) can be loaded as a
     warm-start for PPO's actor input+hidden layers AND can serve as the dist
     teacher without any feature adapter (input dims match exactly).

ACTIVATION: cfg["MULTI_TF_OBS"] = True
  - When True, BatchedFTMOEnv routes _get_state() through this provider.
  - When False (default), the env behaves byte-for-byte as before.

Indicator code is in core/env/indicators_legacy.py (also a verbatim port).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from core.env.indicators_legacy import build_feature_matrix as legacy_build_features

# ── Public constants ────────────────────────────────────────────────────────
LEGACY_NUM_FEATURES = 27          # see indicators_legacy.build_feature_matrix
LEGACY_TF_FACTORS = (1, 15, 60, 1440)   # 1m, 15m, 1h, 1d (DQN-era schema)
LEGACY_LOOKBACK = 20
LEGACY_N_TRAILING = 6             # position, unrealised, eq_chg, gap, dd_head, daily_ret
LEGACY_STATE_DIM = (
    LEGACY_LOOKBACK * LEGACY_NUM_FEATURES * len(LEGACY_TF_FACTORS) + LEGACY_N_TRAILING
)
assert LEGACY_STATE_DIM == 2166, "Legacy state_dim invariant broken — DQN checkpoint will not load"

# Column indices inside the 27-feature output (mirrors indicators_legacy header comment)
LEGACY_COL = {
    "open": 0, "high": 1, "low": 2, "close": 3, "volume": 4,
    "atr14": 5, "atr45": 6,
    "rsi7": 7, "rsi14": 8,
    "cci14": 9, "cci30": 10, "cci100": 11,
    "sma20": 12, "sma50": 13, "sma200": 14,
    "bb20_upper": 15, "bb20_mid": 16, "bb20_lower": 17,
    "bb200_upper": 18, "bb200_mid": 19, "bb200_lower": 20,
    "high_sma4_sh8": 21, "low_sma4_sh8": 22,
    "sma2_sh0": 23, "sma2_sh1": 24, "sma2_sh2": 25,
    "atr14_sma1_sh8": 26,
}


class LegacyMultiTFStateProvider:
    """Builds the 2166-dim DQN-era observation from raw 1-minute OHLCV.

    Holds, on-device, one resampled 27-feature tensor per timeframe. Resampling
    matches the predecessor's logic EXACTLY: last bar of each TF window
    (idx = arange(tf-1, T, tf)).

    Args:
        raw_ohlcv: shape (T, 5) np.ndarray of float64/32 [open,high,low,close,volume].
        device: torch device.
        tf_factors: tuple of resample factors. Default LEGACY_TF_FACTORS.
        lookback: bars of history. Default LEGACY_LOOKBACK (20).
    """

    def __init__(
        self,
        raw_ohlcv: np.ndarray,
        device: torch.device,
        tf_factors: tuple = LEGACY_TF_FACTORS,
        lookback: int = LEGACY_LOOKBACK,
    ):
        assert raw_ohlcv.ndim == 2 and raw_ohlcv.shape[1] == 5, (
            f"raw_ohlcv must be (T, 5) OHLCV; got {raw_ohlcv.shape}"
        )
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.tf_factors = tuple(tf_factors)
        self.lookback = int(lookback)
        self.T = int(raw_ohlcv.shape[0])
        self.F = LEGACY_NUM_FEATURES

        o, h, l, c, v = (raw_ohlcv[:, i] for i in range(5))
        feat_1m = legacy_build_features(o, h, l, c, v)   # (T, 27) numpy
        # Move to device once; never recompute.
        self._feat_1m = torch.tensor(feat_1m, dtype=torch.float32, device=self.device)

        # Resample: last bar of each TF window — verbatim from predecessor.
        self._resampled: Dict[int, torch.Tensor] = {}
        for tf in self.tf_factors:
            if tf == 1:
                self._resampled[tf] = self._feat_1m
            else:
                idx = torch.arange(tf - 1, self.T, tf, device=self.device)
                self._resampled[tf] = self._feat_1m[idx]

        # Warmup: enough history that the largest TF has its lookback filled.
        # max(tf) * lookback + 200 safety pad matches predecessor's reset() warmup.
        self.min_warmup_idx = max(self.tf_factors) * self.lookback + 200

    @property
    def state_dim(self) -> int:
        return LEGACY_STATE_DIM

    def build_window_features(self, abs_idx: torch.Tensor) -> torch.Tensor:
        """Return the (B, lkbk*F*NumTF) feature portion of the obs (no trailing).

        Per-window mean/std normalization PER TF — matches predecessor exactly.
        """
        B = abs_idx.shape[0]
        offsets = torch.arange(self.lookback - 1, -1, -1, device=self.device)
        parts = []
        for tf in self.tf_factors:
            feat = self._resampled[tf]
            tf_idx = (abs_idx // tf).clamp(0, feat.shape[0] - 1)
            win_idx = (tf_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp(0, feat.shape[0] - 1)
            window = feat[win_idx]                                # (B, lkbk, F)
            mu = window.mean(dim=(1, 2), keepdim=True)
            std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            parts.append(((window - mu) / std).reshape(B, -1))     # (B, lkbk*F)
        return torch.cat(parts, dim=1)                             # (B, lkbk*F*NumTF)

    def build_trailing_features(
        self,
        position: torch.Tensor,
        entry_px: torch.Tensor,
        equity: torch.Tensor,
        initial_equity: torch.Tensor,
        day_start_eq: torch.Tensor,
        day_high_eq: torch.Tensor,
        target_pct: torch.Tensor,
        max_dd_pct: torch.Tensor,
        curr_close: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 6 trailing features in the DQN-era order:

          [position, unrealised, eq_chg, gap_to_target, dd_headroom, daily_ret]

        Each accepts a (B,) tensor. Result shape (B, 6).
        """
        unrealised = torch.where(
            position != 0,
            (curr_close - entry_px) / (entry_px + 1e-8) * position,
            torch.zeros_like(curr_close),
        )
        eq_chg = (equity - initial_equity) / (initial_equity + 1e-8)
        target_eq = day_start_eq * (1.0 + target_pct)
        gap_to_tgt = (target_eq - equity) / (initial_equity + 1e-8)
        dd_used = (day_high_eq - equity) / (day_high_eq + 1e-8)
        dd_headroom = (max_dd_pct - dd_used).clamp(min=0.0)
        daily_ret = (equity - day_start_eq) / (day_start_eq + 1e-8)
        return torch.stack(
            [position, unrealised, eq_chg, gap_to_tgt, dd_headroom, daily_ret],
            dim=1,
        )

    def build_state(
        self,
        abs_idx: torch.Tensor,
        position: torch.Tensor,
        entry_px: torch.Tensor,
        equity: torch.Tensor,
        initial_equity: torch.Tensor,
        day_start_eq: torch.Tensor,
        day_high_eq: torch.Tensor,
        target_pct: torch.Tensor,
        max_dd_pct: torch.Tensor,
    ) -> torch.Tensor:
        """One-call wrapper. Returns (B, 2166)."""
        curr_close = self._feat_1m[abs_idx.clamp(0, self.T - 1), LEGACY_COL["close"]]
        win = self.build_window_features(abs_idx)
        trail = self.build_trailing_features(
            position=position,
            entry_px=entry_px,
            equity=equity,
            initial_equity=initial_equity,
            day_start_eq=day_start_eq,
            day_high_eq=day_high_eq,
            target_pct=target_pct,
            max_dd_pct=max_dd_pct,
            curr_close=curr_close,
        )
        return torch.cat([win, trail], dim=1)


def collapse_dqn_7_to_3_probs(probs_7: torch.Tensor) -> torch.Tensor:
    """Map the DQN's 7-way action distribution to canonical [BUY, SELL, HOLD].

    DQN action table (from predecessor _SIGN_T / _SIZE_T):
       idx 0: HOLD
       idx 1,2,3: BUY (small, medium, large lot)
       idx 4,5,6: SELL (small, medium, large lot)

    PPO consumes only direction, so we sum the 3 size buckets per direction.
    Returns (..., 3) tensor in order [BUY, SELL, HOLD]. Always sums to 1.0.
    """
    assert probs_7.shape[-1] == 7, f"expected 7-way probs, got {probs_7.shape[-1]}"
    buy = probs_7[..., 1:4].sum(dim=-1)
    sell = probs_7[..., 4:7].sum(dim=-1)
    hold = probs_7[..., 0]
    out = torch.stack([buy, sell, hold], dim=-1)
    # Defensive renormalize (should already sum to 1 but float drift...).
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
