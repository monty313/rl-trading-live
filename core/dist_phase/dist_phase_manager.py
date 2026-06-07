# ═══════════════════════════════════════════════════════════════════════
# [DIST PRE-PHASE] FULL LIFECYCLE — READ THIS HEADER BEFORE TOUCHING
# ═══════════════════════════════════════════════════════════════════════
# This file is the single source of truth for the distillation lifecycle.
# Every other dist_* file points back to this header. If an LLM reads only
# this comment it should know exactly what to do at every stage.
#
# WHAT THIS SYSTEM DOES (in one sentence)
# A frozen pre-trained DQN (eurusd_gpu_ph0_ep0120.pt) teaches the PPO
# agent ONE thing — which direction to take on entry (BUY vs SELL) —
# then retires itself once PPO proves it has learned the strategy.
#
# THE FOUR LIFECYCLE STAGES, IN ORDER
# ===================================
#
# ┌── STAGE 1: SETUP ─────────────────────────────────────────────────────┐
# │ Action:   Open dist_prephase_run_all.ipynb in Colab → Runtime →       │
# │           Run all. No paste-in needed; cells communicate via vars.    │
# │ Cells do: clone branch, download checkpoint if missing, probe         │
# │           checkpoint, build env+teacher+wrapper through pipeline,     │
# │           1-step sanity check, launch training/train.py.              │
# │ Kill switch: ENABLE_DIST=False in Cell 2 → runs base repo unchanged.  │
# └───────────────────────────────────────────────────────────────────────┘
#                                ↓
# ┌── STAGE 2: DIST_PRE_PHASE (DQN teaching, wide DD) ────────────────────┐
# │ Settings: distillation_weight=0.30 (constant)                         │
# │           max_daily_dd=5%  (wide — lets PPO explore safely)           │
# │           daily_target=2%  (low bar)                                  │
# │           DQN active: YES — real probs in obs slots                   │
# │ Exit:     Automatic on first day-end → DIST_PHASE_1.                  │
# │           You do not have to do anything.                             │
# │ Logs:     '[DIST] DQN Teacher loaded' (once)                          │
# │           per-day: dist_bonus > 0 on entry steps, 0 otherwise         │
# │           per-day: agreement_rate climbing above 50% (baseline)       │
# └───────────────────────────────────────────────────────────────────────┘
#                                ↓
# ┌── STAGE 3: DIST_PHASE_1 (DQN fading, full FTMO) ──────────────────────┐
# │ Settings: distillation_weight = 0.30 * (1 - best_gate_days/10)        │
# │           max_daily_dd=1%   (real FTMO rules from here)               │
# │           daily_target=2.5%                                           │
# │                                                                       │
# │ Graduation gate — ALL THREE signals must pass:                        │
# │                                                                       │
# │   Signal 1 (Performance, evaluated every day):                        │
# │     10 consecutive days where ALL hold:                               │
# │       win_rate         > 0.55                                         │
# │       profit_factor    > 1.3                                          │
# │       expectancy_pips  > 0                                            │
# │       trades_per_day   ≥ 3                                            │
# │       max_daily_dd     not breached                                   │
# │     A single failing day resets the streak to 0.                      │
# │                                                                       │
# │   Signal 2 (Convergence, 5-day rolling):                              │
# │     normalized_agreement = (raw - 0.5) / 0.5  ≥ 0.30                  │
# │     i.e. PPO agrees with DQN on ≥65% of confident entry steps         │
# │     averaged across the last 5 days. Baseline is 50% (because         │
# │     FLAT is masked under force_in_and_gate → BUY/SELL is a coin       │
# │     flip even for a random policy).                                   │
# │                                                                       │
# │   Signal 3 (Independence, 3-day solo):                                │
# │     After Signals 1 AND 2 are both passing the manager auto-triggers  │
# │     a solo dry run: 3 consecutive days with DQN silent (weight=0,     │
# │     obs slots frozen). All 3 days must independently pass the         │
# │     Signal 1 criteria. ANY failure aborts the solo run AND resets     │
# │     the Signal 1 streak to 0 — you start the 10-day count from        │
# │     scratch. Cooldown of 3 days between solo attempts.                │
# │                                                                       │
# │ What YOU do during Stage 3: NOTHING.                                  │
# │   on_dist_day_end() is called automatically from the training loop.   │
# │   The manager handles streak counting, weight fade, solo triggering,  │
# │   and writing dist_graduation_record.json at the moment of success.   │
# │                                                                       │
# │ How to tell PPO has learned the strategy → look for this banner:      │
# │   ┌───────────────────────────────────────────────────┐               │
# │   │ [DIST] GRADUATION PROOF COMPLETE ✅                │               │
# │   │ Signal 1: 10/10 consecutive gate days passed     │               │
# │   │ Signal 2: XX.X% agreement (normalized: 0.YY)     │               │
# │   │ Signal 3: Solo dry run — 3/3 days passed         │               │
# │   │ → DQN RETIRING. Graduation record written.       │               │
# │   └───────────────────────────────────────────────────┘               │
# └───────────────────────────────────────────────────────────────────────┘
#                                ↓
# ┌── STAGE 4: GRADUATION & REMOVAL ──────────────────────────────────────┐
# │ When: After the graduation banner.                                    │
# │ Now:  PPO trades independently. DQN is no longer called. Its 3 obs    │
# │       slots are frozen to the EMPIRICAL mean of probs observed during │
# │       training (NOT uniform 0.333) — this avoids distribution shift.  │
# │       You can keep training in this state indefinitely; the wrapper   │
# │       is now passive aside from the 3 frozen slots. Lot sizing and    │
# │       higher-phase features continue to improve PPO.                  │
# │                                                                       │
# │ To fully REMOVE the dist system (do this within a few weeks):         │
# │   1. Confirm /content/drive/MyDrive/checkpoints/                      │
# │      dist_graduation_record.json exists and shows                     │
# │      'dist_graduation_complete': true. KEEP THIS FILE FOREVER — it is │
# │      your audit trail of WHY the teacher was removed.                 │
# │   2. core/settings.py — delete the entire block bookended by          │
# │      '[DIST PRE-PHASE START]' / '[DIST PRE-PHASE END]'.               │
# │   3. core/pipeline.py — delete the wrapper block bookended by         │
# │      '[DIST PRE-PHASE START]' / '[DIST PRE-PHASE END]'.               │
# │   4. Delete the dist files:                                           │
# │        rm -rf core/dist_teacher/ core/dist_phase/                     │
# │        rm -rf tests/dist/                                             │
# │        rm    scripts/dist_checkpoint_probe.py                         │
# │        rm    docs/dist_prephase_colab_cells.md                        │
# │        rm    dist_prephase_run_all.ipynb                              │
# │   5. pytest tests/   — every original test must still pass.           │
# │   6. Verify removal is complete:                                      │
# │        grep -r 'DIST PRE-PHASE' .   → must return empty               │
# │        grep -r 'dist_prephase'   .  → must return empty               │
# │   7. Commit on chore/remove-dist-prephase branch and open PR.         │
# │                                                                       │
# │ IMPORTANT: PPO's actor-critic was built with state_dim = base + 3     │
# │ during DIST_*. The FIRST training run after removal must use          │
# │ --force-fresh OR re-initialize the actor head against the smaller    │
# │ obs dim. Existing checkpoints from the dist era can be loaded only   │
# │ if you re-add the 3 frozen slots (essentially the wrapper) — at that │
# │ point you haven't really removed it. Cleanest path: --force-fresh.   │
# └───────────────────────────────────────────────────────────────────────┘
#
# DECISION TABLE — 'WHAT DO I DO NOW?'
# ====================================
#   You see                                       → Action
#   ----------------------------------------------------------------
#   Just cloned branch, never trained yet         → Stage 1 (Run All)
#   Training running, '[DIST] DIST_PRE_PHASE'     → Wait. Let it run.
#   Banner '[DIST] DIST_PHASE_1 STARTED'          → Wait. Watch streak.
#   Logs: consecutive_gate_days climbing 0→10     → Good. Keep waiting.
#   Banner '[DIST] SOLO DRY RUN BEGIN'            → Wait 3 days.
#   Banner '[DIST] SOLO DRY RUN FAILED'           → Don't panic. Streak
#                                                   resets; PPO keeps
#                                                   training. Next attempt
#                                                   after 3-day cooldown.
#   Banner '[DIST] GRADUATION PROOF COMPLETE ✅'  → PPO has learned the
#                                                   strategy. Go to Stage 4.
#   dist_graduation_record.json written           → Save it, then remove
#                                                   the dist code per
#                                                   Stage 4 instructions.
#
# WHERE TO FIND THINGS
# ====================
#   Master kill switch:        core/settings.py [DIST PRE-PHASE] block
#   Wrapper attach site:       core/pipeline.py [DIST PRE-PHASE] block
#   Frozen DQN loader:         core/dist_teacher/dist_dqn_teacher.py
#   Obs adapter (slice-only):  core/dist_teacher/dist_obs_adapter.py
#   Env wrapper (obs+bonus):   core/dist_teacher/dist_prephase_wrapper.py
#   Gate logic (THIS FILE):    core/dist_phase/dist_phase_manager.py
#   Run-all notebook:          dist_prephase_run_all.ipynb
#   Tests:                     tests/dist/
#   Graduation record (proof): /content/drive/MyDrive/checkpoints/
#                              dist_graduation_record.json
# ═══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional


class DistPhase(str, Enum):
    PRE_PHASE = "DIST_PRE_PHASE"
    PHASE_1 = "DIST_PHASE_1"
    RETIRED = "DIST_RETIRED"


@dataclass
class DistDailyMetrics:
    """Closed-day metrics consumed by the phase manager."""

    day: int
    date: str = ""
    pnl_usd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_pips_net: float = 0.0
    trades: int = 0
    max_dd_pct: float = 0.0
    dd_breached: bool = False
    # Agreement on entry steps (DQN top action with confidence >= threshold).
    agreement_count: int = 0
    confident_entry_steps: int = 0


@dataclass
class _DailyRecord:
    day: int
    date: str
    pnl_usd: float
    win_rate: float
    profit_factor: float
    expectancy_pips: float
    trades: int
    max_dd_pct: float
    passed: bool
    is_solo: bool = False
    agreement_rate: float = 0.0


@dataclass
class _GraduationState:
    consecutive_gate_days: int = 0
    best_gate_days: int = 0
    daily_agreement_window: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    daily_records: List[_DailyRecord] = field(default_factory=list)
    # Solo-run bookkeeping
    in_solo_run: bool = False
    solo_days_passed: int = 0
    solo_results: List[Dict[str, Any]] = field(default_factory=list)
    solo_dry_run_passed: bool = False
    last_solo_attempt_day: int = -10**9


def _get(d, *keys, default=None):
    """Nested lookup with a default. Accepts dict or attr objects."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default if k == keys[-1] else {})
        else:
            cur = getattr(cur, k, default if k == keys[-1] else None)
    return cur if cur is not None else default


class DistPhaseManager:
    """[DIST] Drives DIST_PRE_PHASE → DIST_PHASE_1 → graduation/retirement.

    Args:
        config: full settings dict (must contain ``dist_phase`` and
            ``dist_teacher`` blocks plus ``dist_prephase_enabled``).
        start_phase: phase to start in. Default DIST_PRE_PHASE.

    REVERT: delete this file with the rest of core/dist_phase/.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        start_phase: DistPhase = DistPhase.PRE_PHASE,
    ):
        self.config = config
        cfg_p = config.get("dist_phase", {})
        cfg_t = config.get("dist_teacher", {})

        self.enabled = bool(config.get("dist_prephase_enabled", False))
        self._current_phase: DistPhase = start_phase if self.enabled else DistPhase.RETIRED

        # Cached settings.
        self.required_gate_days = int(cfg_p.get("required_gate_days", 10))
        self.gate_win_rate = float(cfg_p.get("gate_win_rate", 0.55))
        self.gate_profit_factor = float(cfg_p.get("gate_profit_factor", 1.3))
        self.gate_expectancy_pips = float(cfg_p.get("gate_expectancy_pips", 0.0))
        self.gate_min_trades_per_day = int(
            cfg_p.get("gate_min_trades_per_day", 3)
        )
        self.signal2_min = float(
            cfg_p.get("signal2_agreement_normalized_min", 0.30)
        )
        self.signal2_window = int(cfg_p.get("signal2_rolling_window_days", 5))
        self.signal3_required = int(cfg_p.get("signal3_solo_days_required", 3))
        self.signal3_cooldown = int(cfg_p.get("signal3_cooldown_days", 3))
        self.graduation_record_path = str(
            cfg_p.get(
                "graduation_record_path",
                "/content/drive/MyDrive/checkpoints/dist_graduation_record.json",
            )
        )
        self.initial_weight = float(
            cfg_t.get("initial_distillation_weight", 0.30)
        )
        self.monotonic_fade = bool(cfg_p.get("monotonic_fade", True))

        # Risk windows per phase.
        self._prephase_max_dd = float(cfg_p.get("prephase_max_daily_dd", 0.05))
        self._prephase_target = float(cfg_p.get("prephase_daily_target", 0.02))
        self._phase1_max_dd = float(cfg_p.get("phase1_max_daily_dd", 0.01))
        self._phase1_target = float(cfg_p.get("phase1_daily_target", 0.025))

        # State.
        self._grad = _GraduationState(
            daily_agreement_window=deque(maxlen=self.signal2_window)
        )
        self._adapter_info: Dict[str, Any] = {}

    # ── public state ────────────────────────────────────────────────────
    @property
    def current_dist_phase(self) -> DistPhase:
        return self._current_phase

    def is_teacher_active(self) -> bool:
        """Teacher is active in PRE_PHASE and PHASE_1; never during solo days."""
        if not self.enabled:
            return False
        if self._current_phase == DistPhase.RETIRED:
            return False
        if self._grad.in_solo_run:
            return False
        return self.get_distillation_weight() > 0.0

    def get_distillation_weight(self) -> float:
        if not self.enabled or self._current_phase == DistPhase.RETIRED:
            return 0.0
        if self._current_phase == DistPhase.PRE_PHASE:
            return self.initial_weight
        # PHASE_1 — gate-based fade.
        if self._grad.in_solo_run:
            return 0.0
        progress = min(
            1.0, self._grad.best_gate_days / max(1, self.required_gate_days)
        )
        return self.initial_weight * (1.0 - progress)

    def get_dist_max_daily_dd(self) -> float:
        return self._prephase_max_dd if self._current_phase == DistPhase.PRE_PHASE \
            else self._phase1_max_dd

    def get_dist_daily_target(self) -> float:
        return self._prephase_target if self._current_phase == DistPhase.PRE_PHASE \
            else self._phase1_target

    # ── transitions ─────────────────────────────────────────────────────
    def advance_to_phase_1(self) -> None:
        """Move from DIST_PRE_PHASE to DIST_PHASE_1 (restores normal FTMO rules)."""
        if self._current_phase != DistPhase.PRE_PHASE:
            return
        self._current_phase = DistPhase.PHASE_1
        self._log_phase_banner(
            "DIST_PHASE_1 STARTED",
            [
                "NORMAL FTMO RULES RESUMED: 1% DD, 2.5% target",
                "DQN fading by gate | 10-day proof required",
            ],
        )

    def _retire_teacher(self) -> None:
        self._current_phase = DistPhase.RETIRED

    # ── day-end gate ────────────────────────────────────────────────────
    def on_dist_day_end(self, metrics: DistDailyMetrics) -> Dict[str, Any]:
        """Process a closed day. Returns a summary dict for logging."""
        if not self.enabled or self._current_phase == DistPhase.RETIRED:
            return {"phase": self._current_phase.value, "noop": True}

        # ── Signal 1 check ───────────────────────────────────────────────
        passed_gate = self._check_signal_1(metrics)
        # Agreement (Signal 2) is tracked for entry-step PPO/DQN alignment.
        agreement_rate = self._agreement_rate_for_day(metrics)
        if metrics.confident_entry_steps > 0:
            self._grad.daily_agreement_window.append(agreement_rate)

        # Are we currently inside a solo dry run window?
        is_solo = self._grad.in_solo_run

        # Record the day.
        self._grad.daily_records.append(
            _DailyRecord(
                day=metrics.day,
                date=metrics.date,
                pnl_usd=metrics.pnl_usd,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                expectancy_pips=metrics.expectancy_pips_net,
                trades=metrics.trades,
                max_dd_pct=metrics.max_dd_pct,
                passed=passed_gate,
                is_solo=is_solo,
                agreement_rate=agreement_rate,
            )
        )

        # Update streak counter (solo days COUNT toward Signal 1's streak).
        if passed_gate:
            self._grad.consecutive_gate_days += 1
            self._grad.best_gate_days = max(
                self._grad.best_gate_days, self._grad.consecutive_gate_days
            )
        else:
            self._grad.consecutive_gate_days = 0
            if is_solo:
                # Solo failure aborts the run and resets streak.
                self._grad.solo_results.append(
                    {
                        "day": metrics.day,
                        "date": metrics.date,
                        "pnl": metrics.pnl_usd,
                        "pf": metrics.profit_factor,
                        "passed": False,
                    }
                )
                self._abort_solo_run(metrics.day)
                return self._summarize(metrics, passed_gate, agreement_rate)

        # If we're inside a solo dry run, tally the day and possibly succeed.
        if is_solo:
            self._grad.solo_days_passed += 1
            self._grad.solo_results.append(
                {
                    "day": metrics.day,
                    "date": metrics.date,
                    "pnl": metrics.pnl_usd,
                    "pf": metrics.profit_factor,
                    "passed": True,
                }
            )
            if self._grad.solo_days_passed >= self.signal3_required:
                self._grad.solo_dry_run_passed = True
                self._grad.in_solo_run = False

        # Auto-transition PRE_PHASE → PHASE_1 the moment we have any data.
        if self._current_phase == DistPhase.PRE_PHASE:
            self.advance_to_phase_1()

        # Possibly trigger a fresh solo run.
        if (
            self._current_phase == DistPhase.PHASE_1
            and not self._grad.in_solo_run
            and not self._grad.solo_dry_run_passed
            and self._check_signal_1_streak_ready()
            and self._check_signal_2()
            and (metrics.day - self._grad.last_solo_attempt_day)
            >= self.signal3_cooldown
        ):
            self._begin_solo_run(metrics.day)

        # Finally — full graduation?
        if (
            self._current_phase == DistPhase.PHASE_1
            and self._check_signal_1_streak_ready()
            and self._check_signal_2()
            and self._grad.solo_dry_run_passed
        ):
            self._graduate()

        return self._summarize(metrics, passed_gate, agreement_rate)

    # ── signal checks ───────────────────────────────────────────────────
    def _check_signal_1(self, m: DistDailyMetrics) -> bool:
        if m.dd_breached:
            return False
        if m.trades < self.gate_min_trades_per_day:
            return False
        if m.win_rate <= self.gate_win_rate:
            return False
        if m.profit_factor <= self.gate_profit_factor:
            return False
        if m.expectancy_pips_net <= self.gate_expectancy_pips:
            return False
        return True

    def _check_signal_1_streak_ready(self) -> bool:
        return self._grad.consecutive_gate_days >= self.required_gate_days

    def _check_signal_2(self) -> bool:
        if len(self._grad.daily_agreement_window) < self.signal2_window:
            return False
        avg = sum(self._grad.daily_agreement_window) / len(
            self._grad.daily_agreement_window
        )
        normalized = (avg - 0.50) / 0.50
        return normalized >= self.signal2_min

    def _agreement_rate_for_day(self, m: DistDailyMetrics) -> float:
        if m.confident_entry_steps <= 0:
            return 0.0
        return float(m.agreement_count) / float(m.confident_entry_steps)

    # ── solo-run lifecycle ──────────────────────────────────────────────
    def _begin_solo_run(self, day: int) -> None:
        self._grad.in_solo_run = True
        self._grad.solo_days_passed = 0
        self._grad.last_solo_attempt_day = day
        # Solo runs silence the teacher — leave records open for write-up.
        print(
            f"[DIST] SOLO DRY RUN BEGIN | day={day} | "
            f"target {self.signal3_required} consecutive passes with DQN silent"
        )

    def _abort_solo_run(self, day: int) -> None:
        self._grad.in_solo_run = False
        self._grad.solo_days_passed = 0
        self._grad.solo_dry_run_passed = False
        self._grad.consecutive_gate_days = 0
        print(
            f"[DIST] SOLO DRY RUN FAILED | day={day} | "
            "streak reset; teacher reactivated"
        )

    # ── graduation ──────────────────────────────────────────────────────
    def _graduate(self) -> None:
        self.write_graduation_record()
        self._retire_teacher()
        agreement_avg = (
            sum(self._grad.daily_agreement_window)
            / max(1, len(self._grad.daily_agreement_window))
        )
        normalized = (agreement_avg - 0.5) / 0.5
        self._log_phase_banner(
            "GRADUATION PROOF COMPLETE ✅",
            [
                f"Signal 1: {self._grad.consecutive_gate_days}/"
                f"{self.required_gate_days} consecutive gate days passed",
                f"Signal 2: {agreement_avg*100:.1f}% agreement "
                f"(normalized: {normalized:.2f})",
                f"Signal 3: Solo dry run — "
                f"{self.signal3_required}/{self.signal3_required} days passed",
                "→ DQN RETIRING. Graduation record written.",
                "→ REVERT TO NORMAL per Section 9 revert comments",
            ],
        )

    def write_graduation_record(self) -> str:
        out_path = self.graduation_record_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        record = self._build_graduation_record()
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)
        print("═" * 60)
        print("[DIST] GRADUATION RECORD WRITTEN:")
        print(f"  {out_path}")
        print("  This file is your proof that PPO learned the strategy")
        print("  before DQN was retired. Keep it permanently.")
        print("═" * 60)
        return out_path

    def _build_graduation_record(self) -> Dict[str, Any]:
        agreement_avg = (
            sum(self._grad.daily_agreement_window)
            / max(1, len(self._grad.daily_agreement_window))
        )
        # Take the most recent N=required_gate_days closed records.
        recent = [r for r in self._grad.daily_records[-self.required_gate_days:]]
        return {
            "dist_graduation_complete": True,
            "graduation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dist_phase_used": self._current_phase.value,
            "obs_adapter_used": bool(self._adapter_info.get("used", False)),
            "obs_adapter_dqn_input_dim": int(self._adapter_info.get("dqn_input_dim", -1)),
            "ppo_obs_dim": int(self._adapter_info.get("ppo_obs_dim", -1)),
            "signal_1_gate": {
                "required_consecutive_days": self.required_gate_days,
                "achieved_consecutive_days": self._grad.consecutive_gate_days,
                "win_rate_min": self.gate_win_rate,
                "profit_factor_min": self.gate_profit_factor,
                "expectancy_pips_min": self.gate_expectancy_pips,
                "min_trades_per_day": self.gate_min_trades_per_day,
                "daily_record": [r.__dict__ for r in recent],
            },
            "signal_2_agreement": {
                "required_normalized_rate": self.signal2_min,
                "achieved_normalized_rate": (agreement_avg - 0.5) / 0.5,
                "actual_agreement_pct": agreement_avg * 100.0,
                "rolling_window_days": self.signal2_window,
                "daily_agreement_rates": list(self._grad.daily_agreement_window),
            },
            "signal_3_solo_dry_run": {
                "solo_days_required": self.signal3_required,
                "solo_days_passed": self._grad.solo_days_passed,
                "solo_day_results": list(self._grad.solo_results),
            },
            "revert_instructions": (
                "See REVERT COMMENTS in core/dist_teacher/ and core/dist_phase/. "
                "Remove all dist_ files. Remove dist_ config keys. Remove the "
                "DistPrePhaseWrapper wrap site in training/train.py. Restore "
                "PPO actor-critic state_dim if expanded."
            ),
            "verified_by": "dist_graduation_record.json — auto-generated at graduation",
        }

    # ── adapter metadata (set by Colab init cell) ───────────────────────
    def record_adapter_info(self, used: bool, dqn_input_dim: int, ppo_obs_dim: int) -> None:
        self._adapter_info = {
            "used": bool(used),
            "dqn_input_dim": int(dqn_input_dim),
            "ppo_obs_dim": int(ppo_obs_dim),
        }

    # ── logging helpers ─────────────────────────────────────────────────
    def _summarize(
        self, m: DistDailyMetrics, passed: bool, agreement_rate: float
    ) -> Dict[str, Any]:
        agreement_window_avg = (
            sum(self._grad.daily_agreement_window)
            / max(1, len(self._grad.daily_agreement_window))
        )
        normalized = (agreement_window_avg - 0.5) / 0.5
        return {
            "phase": self._current_phase.value,
            "day": m.day,
            "passed_signal_1": passed,
            "consecutive_gate_days": self._grad.consecutive_gate_days,
            "best_gate_days": self._grad.best_gate_days,
            "agreement_rate_day": agreement_rate,
            "agreement_window_avg": agreement_window_avg,
            "agreement_normalized": normalized,
            "signal_2_pass": self._check_signal_2(),
            "in_solo_run": self._grad.in_solo_run,
            "solo_days_passed": self._grad.solo_days_passed,
            "solo_run_passed": self._grad.solo_dry_run_passed,
            "distillation_weight": self.get_distillation_weight(),
        }

    def _log_phase_banner(self, title: str, lines: List[str]) -> None:
        bar = "╔" + "═" * 58 + "╗"
        bot = "╚" + "═" * 58 + "╝"
        print(bar)
        print(f"║  [DIST] {title:<48}║")
        for ln in lines:
            print(f"║  {ln:<54}║")
        print(bot)
