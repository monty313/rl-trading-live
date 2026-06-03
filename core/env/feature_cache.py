"""
core/env/feature_cache.py
────────────────────────────────────────────────────────────────────────────
Disk cache for the expensive, deterministic feature-engineering step
(learning_loop_fix.md FIX 4.1 — the highest-leverage GPU/cost win).

WHY THIS EXISTS
  Building the (N, NUM_FEATURES) feature matrix + the per-timeframe indicator
  DataFrames from ~1.9M 1-minute bars takes ~10-15 min on every restart. The
  computation is a PURE FUNCTION of (raw OHLCV bytes, feature-config) — it never
  changes between runs unless the CSV or the indicator code changes. So we cache
  the result to disk and load it in seconds on restart (target: sub-2-min boot).

CACHE KEY (invalidation)
  A single SHA-256 hash over:
    • the CSV path,
    • the CSV file mtime + size (cheap proxy for "did the data change"),
    • a feature-config signature: NUM_FEATURES, FEATURE_COLUMNS, the gate
      timeframes, and the md5 of indicators.py (so editing the indicator code
      invalidates every cache — same parity discipline as the manifest).
  If ANY of these change, the key changes and we rebuild. This makes stale
  caches impossible to load by construction.

ON-DISK FORMAT
  A single torch .pt blob: {"features": Tensor(N,F) cpu, "tf_indicators":
  {tf: records-list}, "meta": {...}}. torch.save/torch.load round-trips tensors
  and plain python containers losslessly. We store the feature tensor on CPU and
  the caller moves it to its device (keeps the cache device-agnostic).

PUBLIC API
  cache_key(csv_path, cfg)                 -> str | None
  cache_path(cfg, key)                     -> pathlib.Path | None
  load_cached(cfg, key)                    -> dict | None
  save_cached(cfg, key, features, tf_ind)  -> path | None
  build_or_load(csv_path, cfg, build_fn)   -> (features, tf_indicators)

This module changes NO learning logic — it only memoizes a deterministic build.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch


def _indicators_md5() -> str:
    """md5 of indicators.py so a change to the indicator code invalidates caches
    (mirrors the manifest parity-hash discipline)."""
    try:
        path = Path(__file__).with_name("indicators.py")
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return "noind"


def _feature_signature(cfg: dict) -> dict:
    """A small JSON-able dict describing the feature configuration. Any change
    here changes the cache key and forces a rebuild."""
    from core.env.indicators import FEATURE_COLUMNS, NUM_FEATURES
    return {
        "num_features": int(NUM_FEATURES),
        "columns": list(FEATURE_COLUMNS),
        "tf_factors": list(cfg.get("TF_FACTORS", [1, 15, 30, 60])),
        "indicators_md5": _indicators_md5(),
        "version": 1,
    }


def cache_key(csv_path: Optional[str], cfg: dict) -> Optional[str]:
    """SHA-256 over (csv path + mtime + size + feature-config signature). Returns
    None when there is no CSV path (e.g. synthetic fixture data) — the caller
    then just builds without caching."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    st = os.stat(csv_path)
    payload = {
        "csv_path": os.path.abspath(csv_path),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "size": int(st.st_size),
        "sig": _feature_signature(cfg),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _cache_dir(cfg: dict) -> Optional[Path]:
    d = cfg.get("FEATURE_CACHE_DIR")
    if d is None:
        # Default: next to the checkpoint/data Drive dir if provided, else a
        # local .feature_cache under the cwd. Configurable via FEATURE_CACHE_DIR.
        base = cfg.get("CHECKPOINT_DIR") or cfg.get("DATA_DIR") or "."
        d = os.path.join(str(base), ".feature_cache")
    return Path(d)


def cache_path(cfg: dict, key: Optional[str]) -> Optional[Path]:
    if not key:
        return None
    return _cache_dir(cfg) / f"features_{key}.pt"


def load_cached(cfg: dict, key: Optional[str]) -> Optional[dict]:
    """Load a cached build if present and loadable; else None (caller rebuilds)."""
    if not bool(cfg.get("USE_FEATURE_CACHE", True)):
        return None
    p = cache_path(cfg, key)
    if p is None or not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "features" in blob:
            return blob
    except Exception:
        pass
    return None


def save_cached(cfg: dict, key: Optional[str], features: torch.Tensor,
                tf_indicators: dict) -> Optional[Path]:
    """Persist a build keyed by `key`. tf_indicators values are stored as plain
    records-lists (DataFrame -> to_dict('records')) so the blob is portable."""
    if not bool(cfg.get("USE_FEATURE_CACHE", True)):
        return None
    p = cache_path(cfg, key)
    if p is None:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tf_ser = {}
    for tf, df in (tf_indicators or {}).items():
        try:
            tf_ser[int(tf)] = {"columns": list(df.columns),
                               "records": df.to_dict("records")}
        except Exception:
            continue
    blob = {
        "features": features.detach().to("cpu"),
        "tf_indicators": tf_ser,
        "meta": {"key": key, "sig": _feature_signature(cfg)},
    }
    tmp = p.with_suffix(".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, p)            # atomic publish — never a half-written cache
    return p


def _records_to_df(payload: dict):
    import pandas as pd
    cols = payload.get("columns")
    recs = payload.get("records", [])
    df = pd.DataFrame(recs)
    if cols:
        df = df.reindex(columns=cols)
    return df.reset_index(drop=True)


def build_or_load(csv_path: Optional[str], cfg: dict,
                  build_fn: Callable[[], Tuple[torch.Tensor, dict]]
                  ) -> Tuple[torch.Tensor, dict]:
    """Return (features_tensor_cpu, tf_indicators_dict_of_DataFrames).

    Fast path: a valid cache exists -> load it (seconds). Slow path: call
    build_fn() (the ~10-15 min build), then persist for next time. build_fn must
    return (features_tensor, {tf: DataFrame}). No learning logic lives here."""
    import pandas as pd  # noqa: F401  (used by _records_to_df)
    key = cache_key(csv_path, cfg)
    cached = load_cached(cfg, key)
    if cached is not None:
        feats = cached["features"]
        tf_ind = {int(tf): _records_to_df(p)
                  for tf, p in (cached.get("tf_indicators") or {}).items()}
        print(f"[feature_cache] HIT {cache_path(cfg, key)} — loaded "
              f"{tuple(feats.shape)} features + {len(tf_ind)} TF frames "
              f"(skipped full rebuild)", flush=True)
        return feats, tf_ind

    print("[feature_cache] MISS — building features from raw bars "
          "(cached for next restart)", flush=True)
    feats, tf_ind = build_fn()
    saved = save_cached(cfg, key, feats, tf_ind)
    if saved is not None:
        print(f"[feature_cache] SAVED -> {saved}", flush=True)
    return feats, tf_ind
