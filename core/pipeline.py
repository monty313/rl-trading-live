"""
core/pipeline.py
────────────────────────────────────────────────────────────────────────────
build_pipeline(cfg, device) -> (env, agent, sizer, guard, gate)

The SINGLE construction point for the trading stack. Training, backtest, eval,
and live_runner all build their objects here so wiring stays consistent and no
module reaches into env/agent/risk internals directly.

Data resolution order for the environment:
  1. cfg["FEATURES"] — a prebuilt (N,F) matrix or (N,5) OHLCV array (tests/smoke)
  2. cfg["DATA_CSV_EURUSD"] — a CSV path (loaded via load_ohlcv_csv)
  3. fallback — synthetic fixture data (with a printed warning), so the pipeline
     always builds even when Google Drive is not mounted.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from core.settings import auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.agent.ppo import PPOAgent
from core.risk.position_sizer import PositionSizer
from core.risk.daily_guard import DailyGuard
from core.risk.trade_gate import TradeGate


def load_ohlcv_csv(path: str) -> np.ndarray:
    """
    Load an M1 CSV into an (N,5) float32 [open,high,low,close,volume] array.
    Handles the user's schema (time,open,high,low,close,tick_volume,...) and the
    common MT5 tab/`<>`-delimited export. Robust to comma or tab delimiters.
    """
    import pandas as pd
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    vol_col = next((c for c in ("tick_volume", "tickvol", "volume", "vol")
                    if c in df.columns), None)
    cols = ["open", "high", "low", "close"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
    df = df.dropna(subset=cols)
    return df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)


def _resolve_features(cfg: dict) -> np.ndarray:
    if cfg.get("FEATURES") is not None:
        return cfg["FEATURES"]
    csv_path = cfg.get("DATA_CSV_EURUSD")
    if csv_path:
        try:
            return load_ohlcv_csv(csv_path)
        except Exception as exc:
            print(f"[pipeline] CSV load failed ({exc}); using synthetic fixture",
                  flush=True)
    from tests.fixtures.sample_candles import make_synthetic_ohlcv_array
    print("[pipeline] Drive/CSV not available — using synthetic fixture data. "
          "Results are for testing only.", flush=True)
    return make_synthetic_ohlcv_array(n=int(cfg.get("SYNTH_BARS", 2000)))


def build_pipeline(cfg: dict, device: torch.device,
                   phase: Optional[dict] = None, policy: Optional[dict] = None
                   ) -> Tuple[BatchedFTMOEnv, PPOAgent, PositionSizer,
                              DailyGuard, TradeGate]:
    """Construct and wire the full stack. Returns 5 objects."""
    cfg = auto_tune_batch(dict(cfg), device)
    cfg["device"] = device

    features = _resolve_features(cfg)
    env = BatchedFTMOEnv(features, cfg, device,
                         instrument=cfg.get("SYMBOL", "EURUSD"),
                         phase=phase, policy=policy)

    cfg["STATE_DIM"] = env.state_dim
    agent = PPOAgent(env.state_dim, cfg, device)   # PURE PPO (DQN deprecated)

    sizer = PositionSizer(cfg)
    mode = (policy or {}).get("mode", "ftmo") if policy else cfg.get("MODE", "ftmo")
    guard = DailyGuard(mode, cfg.get("INITIAL_EQUITY", 100_000.0), cfg)
    gate = TradeGate(daily_guard=guard,
                     log_path=cfg.get("TRADE_LOG", "logs/daily_trade_log.csv"))
    return env, agent, sizer, guard, gate
