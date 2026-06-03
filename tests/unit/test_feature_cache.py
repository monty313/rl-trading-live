"""
tests/unit/test_feature_cache.py
────────────────────────────────────────────────────────────────────────────
Tests for the disk feature cache (learning_loop_fix.md FIX 4.1). The cache is a
PURE memoization of the deterministic feature build — it must never change the
result, only skip recomputation. We verify:

  • round-trip: save_cached -> load_cached returns the same feature tensor and
    the same per-timeframe indicator frames (records-list round-trips losslessly).
  • build_or_load: first call MISSES (runs build_fn), second call HITS (build_fn
    is NOT re-run) and returns identical data.
  • invalidation: changing the feature-config signature (e.g. TF factors) or the
    underlying CSV (mtime/size) changes the cache key so a stale blob is never
    loaded by construction.
  • no-CSV (synthetic) path: cache_key is None and build runs every time.

These use a temp dir + tiny synthetic frames — no real data, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from core.env import feature_cache as FC

DEV = torch.device("cpu")


def _tiny_build():
    """A cheap deterministic build_fn() -> (features_tensor, {tf: DataFrame})."""
    feats = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    df1 = pd.DataFrame({"cci": [1.0, 2.0, 3.0], "rsi": [50.0, 60.0, 70.0]})
    df15 = pd.DataFrame({"cci": [-1.0, -2.0], "rsi": [40.0, 30.0]})
    return feats, {1: df1, 15: df15}


def _write_csv(tmp_path, text="a,b\n1,2\n3,4\n"):
    p = tmp_path / "bars.csv"
    p.write_text(text)
    return str(p)


def test_save_load_round_trip(tmp_path):
    """save_cached then load_cached returns identical features + TF frames."""
    csv = _write_csv(tmp_path)
    cfg = {"FEATURE_CACHE_DIR": str(tmp_path / "cache"), "USE_FEATURE_CACHE": True}
    key = FC.cache_key(csv, cfg)
    assert key is not None

    feats, tf = _tiny_build()
    path = FC.save_cached(cfg, key, feats, tf)
    assert path is not None and path.exists()

    blob = FC.load_cached(cfg, key)
    assert blob is not None
    assert torch.equal(blob["features"], feats)
    # tf_indicators stored as records-lists keyed by int tf
    assert set(blob["tf_indicators"].keys()) == {1, 15}
    rec1 = blob["tf_indicators"][1]["records"]
    assert rec1[0]["cci"] == 1.0 and rec1[2]["rsi"] == 70.0


def test_build_or_load_miss_then_hit(tmp_path):
    """First build_or_load MISSES (build_fn runs); second HITS (build_fn does NOT
    run) and returns the same tensor + reconstructed DataFrames."""
    csv = _write_csv(tmp_path)
    cfg = {"FEATURE_CACHE_DIR": str(tmp_path / "cache"), "USE_FEATURE_CACHE": True}

    calls = {"n": 0}

    def build_fn():
        calls["n"] += 1
        return _tiny_build()

    f1, tf1 = FC.build_or_load(csv, cfg, build_fn)
    assert calls["n"] == 1                              # first call built
    f2, tf2 = FC.build_or_load(csv, cfg, build_fn)
    assert calls["n"] == 1                              # second call HIT (no rebuild)

    assert torch.equal(f1, f2)
    assert set(tf2.keys()) == {1, 15}
    assert isinstance(tf2[1], pd.DataFrame)
    pd.testing.assert_frame_equal(tf1[1].reset_index(drop=True),
                                  tf2[1].reset_index(drop=True))


def test_key_changes_on_config_signature(tmp_path):
    """Changing the feature-config signature (TF factors) changes the cache key,
    so an old build can never be served for a new feature configuration."""
    csv = _write_csv(tmp_path)
    cfg_a = {"FEATURE_CACHE_DIR": str(tmp_path / "c"), "TF_FACTORS": [1, 15, 30, 60]}
    cfg_b = {"FEATURE_CACHE_DIR": str(tmp_path / "c"), "TF_FACTORS": [1, 5, 15]}
    ka = FC.cache_key(csv, cfg_a)
    kb = FC.cache_key(csv, cfg_b)
    assert ka and kb and ka != kb


def test_key_changes_on_csv_change(tmp_path):
    """Changing the CSV contents (size/mtime) changes the key — a modified data
    file invalidates the cache by construction."""
    csv = _write_csv(tmp_path, "a,b\n1,2\n")
    cfg = {"FEATURE_CACHE_DIR": str(tmp_path / "c")}
    k1 = FC.cache_key(csv, cfg)
    # rewrite with different size/content
    _write_csv(tmp_path, "a,b\n1,2\n3,4\n5,6\n7,8\n")
    k2 = FC.cache_key(csv, cfg)
    assert k1 and k2 and k1 != k2


def test_no_csv_path_disables_cache(tmp_path):
    """Synthetic-data path (no CSV): cache_key is None and build_or_load always
    runs build_fn (cache is a no-op, never persisted)."""
    cfg = {"FEATURE_CACHE_DIR": str(tmp_path / "c"), "USE_FEATURE_CACHE": True}
    assert FC.cache_key(None, cfg) is None

    calls = {"n": 0}

    def build_fn():
        calls["n"] += 1
        return _tiny_build()

    FC.build_or_load(None, cfg, build_fn)
    FC.build_or_load(None, cfg, build_fn)
    assert calls["n"] == 2                              # no caching without a CSV


def test_use_feature_cache_false_skips_persist(tmp_path):
    """With USE_FEATURE_CACHE False, save/load are no-ops even with a valid key."""
    csv = _write_csv(tmp_path)
    cfg = {"FEATURE_CACHE_DIR": str(tmp_path / "c"), "USE_FEATURE_CACHE": False}
    key = FC.cache_key(csv, cfg)
    feats, tf = _tiny_build()
    assert FC.save_cached(cfg, key, feats, tf) is None
    assert FC.load_cached(cfg, key) is None
