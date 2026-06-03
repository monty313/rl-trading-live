"""Friendly error for a missing data CSV (unmounted-Drive guard).

These tests lock in the behavior that a missing training CSV produces an
ACTIONABLE error with exact remediation steps — never a generic crash. The
original incident: a Colab run died with a scary "Unexpected error (not in
known-error list)" because Drive wasn't mounted and the CSV path was empty.

Covered:
  1. core.pipeline.load_ohlcv_csv raises DataFileNotFoundError (a FileNotFoundError
     subclass) BEFORE touching pandas, with the remediation keywords.
  2. build_pipeline surfaces the same friendly error via _resolve_features.
  3. training.train._diagnose maps a raw missing-CSV FileNotFoundError to the
     friendly FIX block (not the generic "Unexpected error" fallback).
"""
import os

import pytest
import torch

from core.pipeline import DataFileNotFoundError, build_pipeline, load_ohlcv_csv
from core.settings import CFG

MISSING = "/definitely/not/a/real/path/EURUSD_M1_does_not_exist.csv"

# Keywords every remediation message MUST contain so the user knows exactly what
# to do (re-mount Drive, force a remount, list the folder).
REMEDIATION_KEYWORDS = ("Cell 2", "force_remount", "ls", "Drive")


def _assert_actionable(text: str):
    assert MISSING in text, "message must name the missing path"
    for kw in REMEDIATION_KEYWORDS:
        assert kw in text, f"remediation keyword {kw!r} missing from message"
    assert "COLAB_RUNBOOK" in text, "message must point to the runbook"


def test_load_ohlcv_csv_raises_friendly_error():
    assert not os.path.exists(MISSING)
    with pytest.raises(DataFileNotFoundError) as ei:
        load_ohlcv_csv(MISSING)
    assert isinstance(ei.value, FileNotFoundError)  # callers can still catch it
    _assert_actionable(str(ei.value))


def test_build_pipeline_surfaces_friendly_error():
    c = dict(CFG)
    c.update({"FEATURES": None, "DATA_CSV_EURUSD": MISSING,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    with pytest.raises(DataFileNotFoundError) as ei:
        build_pipeline(c, torch.device("cpu"))
    _assert_actionable(str(ei.value))


def test_train_diagnose_maps_missing_csv_to_fix():
    from training.train import _diagnose
    exc = FileNotFoundError(
        "[Errno 2] No such file or directory: "
        "'/content/drive/MyDrive/RL-Trading-Data/EURUSD_M1.csv'"
    )
    advice = _diagnose(exc)
    # Must NOT fall through to the generic "not in known-error list" message.
    assert "not in known-error list" not in advice
    for kw in ("Cell 2", "force_remount", "ls"):
        assert kw in advice
    assert "COLAB_RUNBOOK" in advice
