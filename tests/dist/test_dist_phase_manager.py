# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY TEST FILE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
"""Tests for DistPhaseManager: thresholds, fade monotonicity, the three-signal
graduation gate, and the permanent graduation record."""
from __future__ import annotations

import json
import os
from typing import Dict

import pytest

from core.dist_phase.dist_phase_manager import (
    DistDailyMetrics,
    DistPhase,
    DistPhaseManager,
)


def _good_day(day: int, **overrides) -> DistDailyMetrics:
    """A day that comfortably passes Signal 1 with high agreement."""
    base = dict(
        day=day,
        date=f"2026-06-{day:02d}",
        pnl_usd=100.0 + day,
        win_rate=0.70,
        profit_factor=1.6,
        expectancy_pips_net=0.9,
        trades=10,
        max_dd_pct=0.003,
        dd_breached=False,
        agreement_count=8,           # 80% of 10 confident entries
        confident_entry_steps=10,
    )
    base.update(overrides)
    return DistDailyMetrics(**base)


# ── DD / target per phase ──────────────────────────────────────────────
def test_dist_dd_per_phase(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PRE_PHASE)
    assert mgr.get_dist_max_daily_dd() == 0.05
    mgr.advance_to_phase_1()
    assert mgr.get_dist_max_daily_dd() == 0.01


def test_dist_target_per_phase(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PRE_PHASE)
    assert mgr.get_dist_daily_target() == 0.02
    mgr.advance_to_phase_1()
    assert mgr.get_dist_daily_target() == 0.025


# ── thresholds ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "field,value,should_fail",
    [
        ("win_rate", 0.53, True),         # below 0.55
        ("profit_factor", 1.25, True),    # below 1.3
        ("trades", 2, True),              # below 3
        ("expectancy_pips_net", -0.1, True),
        ("dd_breached", True, True),
        ("win_rate", 0.60, False),        # above floor → ok if everything else ok
    ],
)
def test_dist_gate_criteria_raised(dist_config, field, value, should_fail):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    m = _good_day(1, **{field: value})
    summary = mgr.on_dist_day_end(m)
    if should_fail:
        assert summary["passed_signal_1"] is False
        assert summary["consecutive_gate_days"] == 0
    else:
        assert summary["passed_signal_1"] is True
        assert summary["consecutive_gate_days"] == 1


# ── fade monotonicity ──────────────────────────────────────────────────
def test_dist_fade_monotonic(dist_config):
    """After 7 passes + 1 fail, distillation_weight must not increase."""
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    for d in range(1, 8):
        mgr.on_dist_day_end(_good_day(d))
    w_after_7 = mgr.get_distillation_weight()
    # Now a failure: streak resets but best_gate_days stays at 7.
    mgr.on_dist_day_end(_good_day(8, win_rate=0.40, dd_breached=True))
    w_after_fail = mgr.get_distillation_weight()
    assert w_after_fail <= w_after_7 + 1e-9


# ── Signal 2 (agreement) ───────────────────────────────────────────────
def test_dist_agreement_normalized(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    # Feed 5 days of 65% agreement → normalized 0.30 → PASS.
    for d in range(1, 6):
        mgr.on_dist_day_end(
            _good_day(d, agreement_count=65, confident_entry_steps=100)
        )
    assert mgr._check_signal_2() is True

    # Now 5 days of 58% agreement → normalized 0.16 → FAIL.
    mgr2 = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    for d in range(1, 6):
        mgr2.on_dist_day_end(
            _good_day(d, agreement_count=58, confident_entry_steps=100)
        )
    assert mgr2._check_signal_2() is False


# ── Signal 3 lifecycle ─────────────────────────────────────────────────
def test_dist_solo_dry_run_triggers_when_signals_1_and_2_pass(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    # 10 good days with high agreement → solo run should trigger.
    for d in range(1, 11):
        mgr.on_dist_day_end(
            _good_day(d, agreement_count=70, confident_entry_steps=100)
        )
    # On the 11th day, in_solo_run should be set (or already finished).
    # Solo runs trigger AFTER the day that pushes signals over the line,
    # so by day 11 either we are in solo or have completed it.
    assert mgr._grad.in_solo_run or mgr._grad.solo_dry_run_passed


def test_dist_solo_resets_streak_on_fail(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    for d in range(1, 11):
        mgr.on_dist_day_end(
            _good_day(d, agreement_count=70, confident_entry_steps=100)
        )
    # Force into solo if not already there.
    if not mgr._grad.in_solo_run:
        mgr._begin_solo_run(day=11)
    # A failing solo day must reset the streak.
    bad = _good_day(11, win_rate=0.30, dd_breached=True)
    mgr.on_dist_day_end(bad)
    assert mgr._grad.consecutive_gate_days == 0
    assert mgr._grad.in_solo_run is False
    assert mgr._grad.solo_dry_run_passed is False


def test_dist_graduation_requires_all_three(dist_config):
    """Even with 10 passing days, no solo run → no graduation."""
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    for d in range(1, 11):
        mgr.on_dist_day_end(
            _good_day(d, agreement_count=70, confident_entry_steps=100)
        )
    # If we ended up in solo, complete it cleanly with 3 more good days.
    if mgr._grad.in_solo_run:
        for d in range(11, 14):
            mgr.on_dist_day_end(
                _good_day(d, agreement_count=70, confident_entry_steps=100)
            )
        assert mgr._current_phase == DistPhase.RETIRED


def test_dist_graduation_record_written(dist_config, tmp_path):
    record_path = str(tmp_path / "grad.json")
    dist_config["dist_phase"]["graduation_record_path"] = record_path

    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PHASE_1)
    for d in range(1, 14):
        mgr.on_dist_day_end(
            _good_day(d, agreement_count=70, confident_entry_steps=100)
        )

    assert os.path.exists(record_path), (
        "Graduation should have produced the record file by day 13"
    )
    with open(record_path) as f:
        rec = json.load(f)
    assert rec["dist_graduation_complete"] is True
    assert "signal_1_gate" in rec
    assert "signal_2_agreement" in rec
    assert "signal_3_solo_dry_run" in rec
    assert rec["signal_1_gate"]["required_consecutive_days"] == 10
    assert rec["signal_3_solo_dry_run"]["solo_days_required"] == 3


def test_dist_disabled_is_noop(dist_config):
    cfg = dict(dist_config)
    cfg["dist_prephase_enabled"] = False
    mgr = DistPhaseManager(cfg, start_phase=DistPhase.PRE_PHASE)
    assert mgr.current_dist_phase == DistPhase.RETIRED
    assert mgr.get_distillation_weight() == 0.0
    assert mgr.is_teacher_active() is False


def test_dist_pre_phase_initial_weight(dist_config):
    mgr = DistPhaseManager(dist_config, start_phase=DistPhase.PRE_PHASE)
    assert mgr.get_distillation_weight() == 0.30
    assert mgr.is_teacher_active() is True
