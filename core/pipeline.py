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

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

from core.settings import auto_tune_batch
from core.env.environment import BatchedFTMOEnv
from core.agent.ppo import PPOAgent
from core.risk.position_sizer import PositionSizer
from core.risk.daily_guard import DailyGuard
from core.risk.trade_gate import TradeGate


def resolve_initial_equity(cfg: dict) -> float:
    """Single source of truth for the account's starting equity.

    The env, the daily guard, and the startup log MUST all agree on this number,
    otherwise the guard's fixed daily increment (initial_equity * target_pct)
    would be computed off a different base than the env's. Resolution order
    mirrors BatchedFTMOEnv.__init__: ACCOUNT_SIZE first, then INITIAL_EQUITY,
    then a $10,000 default (learning_loop_fix.md FIX 3 — comprehensible numbers).
    """
    return float(cfg.get("ACCOUNT_SIZE", cfg.get("INITIAL_EQUITY", 10_000.0)))


def ftmo_rule_summary(cfg: dict) -> str:
    """Build the authoritative one-line startup banner describing the ACTIVE FTMO
    rules for this run, e.g.:

        [ftmo] daily target = 2.50% (=$250 on $10,000 account)  |  daily max DD = 1.00%

    These values are read STRAIGHT from cfg at RUNTIME (the CLI flags
    --target-pct / --max-dd-pct / --daily-target-usd populate them), so the line
    always reflects what is actually enforced THIS run — including on a resume,
    where the CURRENT cfg/CLI wins over anything a checkpoint might have carried.
    See the principles block in core/env/environment.py for the full rule set.
    """
    init_eq = resolve_initial_equity(cfg)
    target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
    max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
    daily_increment = init_eq * target_pct
    return (f"[ftmo] daily target = {target_pct * 100:.2f}% "
            f"(=${daily_increment:,.2f} on ${init_eq:,.0f} account)"
            f"  |  daily max DD = {max_dd_pct * 100:.2f}%")


class DataFileNotFoundError(FileNotFoundError):
    """Raised when the OHLCV CSV does not exist on disk.

    ────────────────────────────────────────────────────────────────────────
    NOTE TO A FUTURE LLM / MAINTAINER:
    If you are reading this because a run crashed with FileNotFoundError on the
    CSV, it is ALMOST ALWAYS an unmounted Google Drive in a fresh/restarted
    Colab session — NOT a code bug in the data loader. When the Colab runtime
    restarts or times out, /content/drive UNMOUNTS, so the CSV path resolves to
    nothing even though nothing in the code changed.

    DO NOT start rewriting load_ohlcv_csv or the pipeline. Instead:
      1. Re-run Cell 2 (MOUNT DRIVE). If it still fails, force a remount:
             from google.colab import drive
             drive.mount('/content/drive', force_remount=True)
      2. Verify the file is actually there:
             !ls -la /content/drive/MyDrive/RL-Trading-Data/
      3. Confirm the filename EXACTLY matches the --csv argument.
      4. Re-run Cell 6 (training).
    See docs/COLAB_RUNBOOK.md for the full ordered checklist + troubleshooting.
    ────────────────────────────────────────────────────────────────────────
    """


def _missing_csv_message(path: str) -> str:
    """Build an actionable, copy-pasteable remediation message for a missing CSV.

    Tailors the wording for Colab: if /content/drive is not even mounted we say
    so explicitly, because that (not a wrong path) is the single most common
    cause of this error in a fresh/restarted Colab session.
    """
    # Detect Colab + whether Drive is mounted at all. On a fresh/restarted Colab
    # session Drive is UNMOUNTED, so the path is empty and read_csv would crash
    # with an opaque FileNotFoundError — we get ahead of that with guidance.
    on_colab = ("google.colab" in sys.modules) or os.path.isdir("/content/drive") \
        or os.path.isdir("/content")
    drive_mounted = os.path.ismount("/content/drive") or os.path.exists("/content/drive/MyDrive")

    lines = [
        "PRIMARY DATA FILE NOT FOUND — the training CSV does not exist.",
        f"  Missing path: {path}",
        "",
        "Most likely cause:",
        "  • Google Drive is NOT mounted in this Colab session (it unmounts on",
        "    every runtime restart / timeout), OR",
        "  • the file path/name is wrong, OR",
        "  • the CSV simply isn't in that Drive folder.",
    ]
    if on_colab and not drive_mounted:
        lines += [
            "",
            "  ⚠️  /content/drive does NOT appear to be mounted — run Cell 2 first.",
        ]
    lines += [
        "",
        "FIX checklist (do these in order):",
        "  (a) Re-run Cell 2 (MOUNT DRIVE). If it STILL fails, force a remount:",
        "          from google.colab import drive",
        "          drive.mount('/content/drive', force_remount=True)",
        "  (b) Verify the file exists:",
        "          !ls -la /content/drive/MyDrive/RL-Trading-Data/",
        "  (c) Confirm the filename EXACTLY matches the --csv argument.",
        "  (d) Then re-run Cell 6 (training).",
        "",
        "See docs/COLAB_RUNBOOK.md for the full ordered run checklist.",
    ]
    return "\n".join(lines)


def load_ohlcv_csv(path: str) -> np.ndarray:
    """
    Load an M1 CSV into an (N,5) float32 [open,high,low,close,volume] array.
    Handles the user's schema (time,open,high,low,close,tick_volume,...) and the
    common MT5 tab/`<>`-delimited export. Robust to comma or tab delimiters.

    Raises DataFileNotFoundError (with copy-pasteable remediation steps) BEFORE
    touching pandas if the path is missing.

    ── FUTURE-LLM NOTE ──────────────────────────────────────────────────────
    A FileNotFoundError here is NOT a loader bug. In Colab it means Drive isn't
    mounted (fresh/restarted runtime → /content/drive is gone). Do NOT rewrite
    this function — re-mount Drive (Cell 2, force_remount=True) and `ls` the
    RL-Trading-Data folder to confirm the file. See docs/COLAB_RUNBOOK.md.
    """
    # Guard FIRST: a clear, actionable error beats pandas' opaque crash.
    if not os.path.exists(path):
        raise DataFileNotFoundError(_missing_csv_message(path))
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
        # load_ohlcv_csv raises DataFileNotFoundError (with full remediation
        # steps) if the path is missing. In Colab that means Drive isn't mounted
        # — re-run Cell 2 (force_remount=True) and `ls` the data folder; do NOT
        # rewrite the loader. See docs/COLAB_RUNBOOK.md.
        data = load_ohlcv_csv(csv_path)
        print(f"[pipeline] Loaded {len(data):,} bars from {csv_path}", flush=True)
        return data
    # Synthetic fallback — tests and smoke scripts only (no CSV path supplied).
    from tests.fixtures.sample_candles import make_synthetic_ohlcv_array
    print("[pipeline] WARNING: no CSV path supplied — using synthetic fixture data. "
          "Phase gate masks will NOT reflect real market conditions.", flush=True)
    return make_synthetic_ohlcv_array(n=int(cfg.get("SYNTH_BARS", 2000)))


def build_pipeline(cfg: dict, device: torch.device,
                   phase: Optional[dict] = None, policy: Optional[dict] = None
                   ) -> Tuple[BatchedFTMOEnv, PPOAgent, PositionSizer,
                              DailyGuard, TradeGate]:
    """Construct and wire the full stack. Returns 5 objects."""
    cfg = auto_tune_batch(dict(cfg), device)
    cfg["device"] = device

    # ── FTMO RULE BANNER (ftmo_rules_fix.md RULE 5) ──────────────────────────
    # Print the ACTIVE daily target / max-DD for THIS run before anything trades.
    # Because it reads cfg (which the CLI flags --target-pct / --max-dd-pct /
    # --daily-target-usd just populated), it is correct for backtest, eval, live,
    # AND a resumed training run — the current cfg always wins over a checkpoint.
    print(ftmo_rule_summary(cfg), flush=True)

    features = _resolve_features(cfg)
    env = BatchedFTMOEnv(features, cfg, device,
                         instrument=cfg.get("SYMBOL", "EURUSD"),
                         phase=phase, policy=policy)

    cfg["STATE_DIM"] = env.state_dim
    agent = PPOAgent(env.state_dim, cfg, device)   # PURE PPO (DQN deprecated)

    sizer = PositionSizer(cfg)
    mode = (policy or {}).get("mode", "ftmo") if policy else cfg.get("MODE", "ftmo")
    # The guard's initial_balance MUST match the env's initial_equity so its fixed
    # daily increment (initial_balance * target_pct) is computed off the SAME base
    # as the env's classification. Use the shared resolver, not a divergent default.
    guard = DailyGuard(mode, resolve_initial_equity(cfg), cfg)
    gate = TradeGate(daily_guard=guard,
                     log_path=cfg.get("TRADE_LOG", "logs/daily_trade_log.csv"))
    return env, agent, sizer, guard, gate
