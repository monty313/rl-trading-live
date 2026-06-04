"""
core/env/environment.py
────────────────────────────────────────────────────────────────────────────
BatchedFTMOEnv — B parallel trading episodes stepped in lockstep on GPU tensors.
Ported from gpu_rl_trading/env/environment.py (REPO1) with these changes:

  (a) ENTRY/EXIT FILLS: the training hot path fills entries and mark-to-market
      at the bar CLOSE (self._entry_px = close on open; MTM at next_close). It
      does NOT call core/env/intrabar_fills.compute_fill — that parity function
      (spread*0.5 + slippage on the entry, SL/TP from OHLC) is used by the
      backtest and the live MT5 runner. Per-trade COST in training is modeled via
      _commission_for_lots (round-trip commission) only. An OPTIONAL entry-price
      spread+slippage adjustment is available behind cfg["ENTRY_FRICTION_ENABLED"]
      (default False) to close the train-vs-live entry-cost gap (applied inline at
      the entry-fill step, see self._friction_px). AUDIT NOTE (P1): with friction
      OFF, training entries are frictionless and thus more optimistic than live
      fills.
  (b) Adds a (B, DIRECTION_DIM) direction mask applied to PPO logits before
      argmax. The mask is produced by conditions_engine from the active phase.
  (c) Multi-symbol: load EURUSD / GBPUSD / XAUUSD / US30 (or aligned baskets).
  (d) CLASSIFICATION — 5-TIER FAIL/OK/PASS/EXCEED + stacked SURVIVAL (commit
      2166ec8). The binary passed/failed flags are retained for compatibility and
      derived from the tiers (passed = PASS or EXCEED). The FIXED-$ daily target
      is unchanged (ftmo_rules_fix.md RULE 1):
        daily_increment = initial_equity * target_pct   (FIXED $, once at open)
        daily_target    = day_start_equity + daily_increment
        passed          = (final_or_halt_equity >= daily_target)
        FAIL            = everything under target (incl. zero-trade days and
                          DD-breach days that end under target). A DD breach does
                          NOT auto-fail: halt_equity >= daily_target still PASSES,
                          but a breached day can never earn SURVIVAL/EXCEED.
        OK is an intermediate tier below PASS; EXCEED is a superset of PASS;
        SURVIVAL stacks additively and only on non-breached days.
  (e) Entire feature tensor preloaded to device at __init__; episodes index slices.
  (f) All tensors live on cfg["device"]; day-boundary logic is vectorized
      (no Python per-batch loop in the hot path — fixes bottleneck #1).
  (g) reset() samples each episode's start bar inside an optional chronological
      window [_start_lo_frac, _start_hi_frac) (set_start_window) so training and
      out-of-sample eval use DISJOINT slices (default = full range).

State layout: a normalized lookback window of the feature matrix, then 6 v1 FTMO
position/account features (position, unrealised, equity change, gap-to-target,
dd-headroom, daily-return), then 7 v2 TARGET/RISK-AWARE features (target_pct,
max_dd_pct, today's difficulty, progress_to_target, dd_headroom,
fraction_of_day_remaining, log-normalized account size) so the policy can
CONDITION on the active FTMO inputs (target_aware_policy.md item 1). state_dim is
computed in __init__ as lkbk*F + N_POSITION_FEATS + N_FTMO_FEATS; the obs layout
is versioned by OBS_SCHEMA_VERSION so a resume can detect an input-dim change.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

import pandas as pd

from core.agent.action_space import (DIRECTION_DIM, EXIT_DIM, FLAT, BUY, SELL,
                                     HOLD, EXIT_REDUCE, EXIT_CLOSE, map_lot)
from core.env.indicators import (build_feature_matrix, NUM_FEATURES, COL,
                                 feature_row_dict, compute_indicators,
                                 resample_ohlcv)
from core.env import conditions_engine
from core.env.gate_precompute import precompute_gate_signal

_NEG_INF = -1e9

# ── OBSERVATION SCHEMA VERSION (target_aware_policy.md item 1 + 4) ───────────
# Bumped whenever the observation LAYOUT changes (number/meaning of features),
# so a checkpoint's input-layer width can be validated on resume. v1 was the
# original `lkbk*F + 6` layout (position/unrealised/eq-change/gap/dd-head/daily-
# ret). v2 ADDED 7 target/risk-aware features (target_pct, max_dd_pct, day
# difficulty, progress_to_target, dd_headroom, fraction_of_day_remaining,
# account_size) so the policy can CONDITION on the FTMO inputs. v3 (reward
# redesign Section 6) APPENDS 7 SESSION/CONTEXT features (CEST time-of-day,
# progress-to-target, remaining-time-in-day, session one-hot-ish label, DD budget
# remaining, signed streak length, per-symbol commission cost). A checkpoint whose
# input layer no longer matches the current state_dim is detected by
# PPOAgent.load(), which reinitializes JUST the input layer (loud log) instead of
# silently mis-loading — the existing partial-reinit-on-resume path is reused.
OBS_SCHEMA_VERSION = 3

# Number of features in each appended block (kept in sync with the blocks built in
# _get_state(); state_dim = lkbk*F + N_POSITION_FEATS + N_FTMO_FEATS + N_SESSION_FEATS).
N_POSITION_FEATS = 6     # position, unrealised, eq_chg, gap, dd_head, daily_ret (v1)
N_FTMO_FEATS = 7         # target_pct, max_dd_pct, difficulty, progress, dd_headroom,
                         # frac_day_remaining, account_size_log (v2, item 1)
N_SESSION_FEATS = 7      # cest_tod, progress_to_target, remaining_time, session_code,
                         # dd_budget_remaining, signed_streak, commission_cost (v3, S6)

# ── FTMO SESSION CLOCK (Section 6) ───────────────────────────────────────────
# FTMO's trading day runs on CEST (Europe/Berlin). Our 1m bars carry no real
# timestamp (alignment is by integer row position), so we derive a SYNTHETIC
# time-of-day from the bar-of-day index: bar i of the day -> minute-of-day
# (i % bars_per_day) / bars_per_day * 1440, offset by the configured session open.
# This is sufficient for the policy to learn intraday timing (session edges,
# remaining time) without needing a real calendar. All session boundaries are
# config-driven (TRADING_SESSIONS in settings) so nothing is hardcoded here.
def session_code_for_minute(minute_of_day: float, sessions: list) -> float:
    """Map a CEST minute-of-day (0..1440) to a normalized session CODE in [0,1].

    `sessions` is a list of (name, start_min, end_min, code) rows from CFG
    ("TRADING_SESSIONS"). Returns the matching row's normalized code (code/ N so
    the four sessions map to ~{0.25,0.5,0.75,1.0}); 0.0 if outside all sessions
    (market effectively closed / thin). Pure + vectorizable per element; the env
    builds the whole batch with a tensor version inline."""
    for name, start, end, code in sessions:
        if start <= minute_of_day < end:
            return float(code)
    return 0.0


# ── COMMISSION (Section 5 — multi-asset framework, EURUSD active) ────────────
def resolve_commission(symbol: str, lots: float, price: float, cfg: dict,
                       side: str = "round_trip") -> float:
    """Return the commission in ACCOUNT CURRENCY for trading `lots` of `symbol`
    at `price`, for one SIDE ("open"/"close") or the full "round_trip".

    The CFG["COMMISSION"] table is keyed by ASSET CLASS; the symbol is mapped to
    its class (explicit lists in CFG["COMMISSION_SYMBOLS"], else heuristic). Two
    cost kinds (Section 5.1):
      • per_lot_round_trip — flat $/standard-lot for the FULL round trip (forex).
        per-side = value/2 * lots; round_trip = value * lots.
      • pct_notional       — fraction of NOTIONAL (lots*contract*price) PER SIDE
        (metals/crypto). round_trip = 2 * per-side.
      • zero               — no commission (indices/oils/agriculture).

    EURUSD worked example (the active path): value=$5.00 round trip per std lot,
    so 0.5 lot -> $2.50 round trip ($1.25/side); 2.0 lot -> $10.00 round trip.
    Pure function so commission is independently unit-testable; the env calls a
    vectorized mirror for forex (the active class) on the hot path."""
    table = cfg.get("COMMISSION", {}) or {}
    cls = classify_symbol(symbol, cfg)
    spec = table.get(cls, {"kind": "zero", "value": 0.0})
    kind = spec.get("kind", "zero")
    value = float(spec.get("value", 0.0))
    lots = float(lots)
    if kind == "zero" or value == 0.0:
        return 0.0
    if kind == "per_lot_round_trip":
        rt = value * lots
        return rt if side == "round_trip" else rt * 0.5
    if kind == "pct_notional":
        contract = float(cfg.get("CONTRACT_SIZE", 100_000.0))
        notional = lots * contract * float(price)
        per_side = value * notional
        return per_side * (2.0 if side == "round_trip" else 1.0)
    return 0.0


def classify_symbol(symbol: str, cfg: dict) -> str:
    """Map a trading SYMBOL to its commission ASSET CLASS. Explicit lists in
    CFG["COMMISSION_SYMBOLS"] win; otherwise heuristic: BTC*/ETH* -> crypto,
    trailing ".cash" -> indices, a 6-letter all-alpha pair -> forex, else the
    default forex (so an unknown FX-looking symbol still gets the active path)."""
    sym = (symbol or "").upper()
    routing = cfg.get("COMMISSION_SYMBOLS", {}) or {}
    for cls, names in routing.items():
        if any(sym == str(n).upper() for n in names):
            return cls
    if sym.startswith("BTC") or sym.startswith("ETH"):
        return "crypto"
    if sym.endswith(".CASH"):
        return "indices"
    alpha = sym.replace(".", "").replace("_", "")
    if len(alpha) == 6 and alpha.isalpha():
        return "forex"
    return "forex"


def proportional_lot_scale(current_target_pct: float, current_max_dd_pct: float,
                           trained_target_pct: float, trained_max_dd_pct: float,
                           lo: float = 0.25, hi: float = 2.0) -> float:
    """Deterministic, BOUNDED lot/aggression scaler (target_aware_policy.md item 6).

    Returns a multiplier applied ON TOP of the agent's own chosen lot at
    inference/eval/live — it NEVER forces direction or exit, only scales exposure.
    It expresses "behave the way you learned, but proportional to how the new
    target/DD differ from the trained baseline":

        target_ratio = current_target_pct / trained_target_pct
        dd_ratio     = current_max_dd_pct / trained_max_dd_pct
        effective_lot_scale = clamp(dd_ratio * f(target_ratio), lo, hi)

    Design of f(target_ratio): a HIGHER target permits more aggression but we damp
    it with a square root so a 2x target does not blindly double exposure
    (f = sqrt(target_ratio)). A TIGHTER DD (dd_ratio<1) scales exposure DOWN
    linearly (the risk budget shrank). At the trained baseline both ratios are 1,
    so the scaler is EXACTLY 1.0 — no behaviour change. Always clamped to [lo, hi]
    so it can never do anything wild even far out of the trained range.
    """
    tt = max(float(trained_target_pct), 1e-9)
    td = max(float(trained_max_dd_pct), 1e-9)
    target_ratio = float(current_target_pct) / tt
    dd_ratio = float(current_max_dd_pct) / td
    raw = dd_ratio * (target_ratio ** 0.5)        # f(target_ratio) = sqrt(target_ratio)
    return float(min(max(raw, lo), hi))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FTMO DAILY PRINCIPLES — SINGLE SOURCE OF TRUTH                            ║
# ║  (referenced by core/risk/daily_guard.py and core/reward/shaper.py)       ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  This block is written in plain language for a future human/LLM. The code ║
# ║  in BatchedFTMOEnv.step() below is the authoritative implementation; if    ║
# ║  the two ever disagree, the CODE wins and this comment is the bug.         ║
# ║                                                                            ║
# ║  ── 1. DAILY TARGET (a FIXED dollar amount) ───────────────────────────── ║
# ║    daily_increment = initial_equity * target_pct                          ║
# ║       • a FLAT dollar amount, computed ONCE at account open.              ║
# ║       • ALWAYS a percentage of the ORIGINAL initial equity — NEVER of the  ║
# ║         current/opening balance, and it does not grow as the account does. ║
# ║    daily_target = day_start_equity + daily_increment                      ║
# ║    PASS  iff  final_or_halt_equity >= daily_target                        ║
# ║    Worked example — $10,000 account @ target_pct = 2.5%:                  ║
# ║       daily_increment = 10,000 * 0.025 = $250  (every single day)         ║
# ║       a day OPENING at 10,300  ->  target = 10,300 + 250 = 10,550         ║
# ║       (NOT 10,300 * 1.025 = 10,557.50 — it is +$250 flat, not +2.5%).     ║
# ║                                                                            ║
# ║  ── 2. TRAILING 1% DRAWDOWN (ratcheting floor, resets daily) ──────────── ║
# ║    At each new day (00:00 CEST intent; the sim boundary is the 1440-bar    ║
# ║    rollover, snapshot-before-rollforward):                                 ║
# ║      start_balance     = day-open balance                                  ║
# ║      max_equity_today  = start_balance                                     ║
# ║      daily_dd_floor     = start_balance * (1 - max_dd_pct)   [default *0.99]║
# ║    Every bar:  equity = balance + open-position MTM  (commissions are      ║
# ║      already in balance, per the 569aeca equity fix). On a NEW equity HIGH ║
# ║      the floor RATCHETS UP and NEVER down within the day:                  ║
# ║         if equity > max_equity_today:                                      ║
# ║             max_equity_today = equity                                      ║
# ║             daily_dd_floor    = max_equity_today * (1 - max_dd_pct)        ║
# ║      (Implemented as peak == _day_high_eq, breach == equity < peak*(1-dd). ║
# ║       _day_high_eq starts at day_start_eq, so the floor opens at *0.99 and ║
# ║       only ever ratchets up — the two formulations are identical.)         ║
# ║    BREACH when:   equity < daily_dd_floor.                                 ║
# ║    On breach -> HALT the day: flatten the position (realizing its MTM into ║
# ║      balance), suppress force-entry, open no new trades until next day.    ║
# ║    A BREACH IS NOT AN AUTO-FAIL: the balance AT THE HALT is classified by   ║
# ║      the SAME 5-tier logic below (halt_balance >= full target still PASSES).║
# ║                                                                            ║
# ║  ── 3. FIVE-TIER CLASSIFICATION (dd_classification_refine.md) ──────────── ║
# ║    Applied to the END-OF-DAY balance, or the HALT balance if breached      ║
# ║    (identical calc). Thresholds are off INITIAL equity (fixed $: +$250 /   ║
# ║    +$125 on $10k), NOT the day's open. prior_day == today's start_balance. ║
# ║    Precedence (FIRST match wins):                                          ║
# ║      1. final < prior_day_balance            -> FAIL_CAPITAL_LOSS  [NEW]   ║
# ║      2. elif final >= initial*(1+target_pct) -> PASS  (>= +2.5%)           ║
# ║      3. elif final >= initial*(1+half_pct)   -> OK    (>= +1.25%, >=50% tgt)║
# ║      4. else                                 -> FAIL_UNDER_TARGET          ║
# ║    EXCEED = PASS AND strictly above the full target AND never breached.     ║
# ║    SURVIVAL bonus = traded AND never breached (a breached day, even one     ║
# ║    classifying PASS/OK at its halt balance, earns NEITHER EXCEED NOR        ║
# ║    SURVIVAL). A zero-trade day ends flat (final==start==prior) -> not below ║
# ║    prior, below half -> FAIL_UNDER_TARGET. Streak: PASS/EXCEED advance the  ║
# ║    pass-streak; OK does NOT; all FAIL_* break it per the mulligan rules.   ║
# ║    Reward ordering holds: PASS/EXCEED > OK > FAIL, and a breach or a        ║
# ║    capital-loss day never out-rewards an OK day.                          ║
# ║                                                                            ║
# ║  ── 4. RUNTIME CONFIG INPUTS ──────────────────────────────────────────── ║
# ║    Both target_pct (CFG["DAILY_TARGET_PCT"]) and max_dd_pct               ║
# ║    (CFG["DAILY_MAX_DD_PCT"]) are RUNTIME inputs set via the CLI flags     ║
# ║    --target-pct / --max-dd-pct (or --daily-target-usd). Changing them     ║
# ║    changes RULE ENFORCEMENT immediately — including on a resume, where the ║
# ║    current cfg wins (checkpoints never store these). The learned POLICY,  ║
# ║    though, was tuned for the values it trained on; large runtime changes  ║
# ║    classify correctly but may need a retrain for best behaviour.          ║
# ║                                                                            ║
# ║  ── 5. TARGET/RISK-AWARE POLICY (target_aware_policy.md) ──────────────── ║
# ║    The agent does NOT only have the rules ENFORCED on it — it OBSERVES the  ║
# ║    active target_pct, max_dd_pct, today's difficulty, its progress_to_      ║
# ║    target, its dd_headroom, the fraction of the day remaining, and the      ║
# ║    account size (see _get_state, item 1). It is meant to ACT PURSUANT to    ║
# ║    them: size up when progress is behind and DD headroom is ample, protect  ║
# ║    gains as headroom thins, etc. Train with --randomize-ftmo to sample a    ║
# ║    (target_pct, max_dd_pct[, account]) PER EPISODE from configurable ranges ║
# ║    so ONE network learns a policy that GENERALIZES across target/risk,      ║
# ║    instead of being implicitly hardwired to 2.5%/1%. At inference the item-6 ║
# ║    proportional_lot_scale() additionally scales exposure relative to the    ║
# ║    trained baseline (bounded, deterministic, never forcing direction/exit). ║
# ║    HONESTY: observation-conditioning + randomized training substantially    ║
# ║    improve generalization, but EXTREME out-of-range target/DD may still     ║
# ║    need a retrain.                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class BatchedFTMOEnv:
    """Vectorized FTMO trading environment over B parallel episodes."""

    def __init__(self, features: "np.ndarray | torch.Tensor", cfg: dict,
                 device: torch.device, instrument: str = "EURUSD",
                 phase: Optional[dict] = None, policy: Optional[dict] = None):
        self.cfg = cfg
        self.device = device
        self.instrument = instrument
        # NOTE: use the private attr here; the public ``phase`` property setter
        # (defined below) triggers gate precompute, which needs self.features /
        # self._tf_indicators to already exist. We set those up first, then
        # assign self.phase at the END of __init__ to fire the first precompute.
        self._phase = phase or {"entry_conditions": {"buy": "any", "sell": "any"}}
        self.policy = policy or {}
        # Cache of the vectorized per-bar gate signal for the active phase.
        # Populated by _refresh_gate_signal() whenever the phase changes. This is
        # what replaces the per-bar pandas to_dict() that caused the 4h stall.
        self._gate_signal = None

        self.B = int(cfg.get("BATCH_SIZE_ENV", 4))
        self.lkbk = int(cfg.get("LOOKBACK", 20))
        # ── FTMO RULE INPUTS (config-driven, never hardcoded; see ftmo_rules_fix.md
        # RULE 5). target_pct / max_dd_pct come straight from CFG (which the CLI
        # flags --target-pct / --max-dd-pct populate in train.py). The user changes
        # these for live accounts, so NOTHING below may bake in 0.025 / 0.01.
        self.target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
        self.max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
        # ── OK-tier HALF target (dd_classification_refine.md) ────────────────────
        # OK fires when the final/halt balance reaches >= INITIAL*(1+half_target_pct)
        # but is still under the full target. Config-driven: DAILY_HALF_TARGET_PCT,
        # or DERIVED as target_pct/2 when it is None. Nothing hardcodes 0.0125.
        from core.reward.shaper import resolve_half_target_pct
        self.half_target_pct = resolve_half_target_pct(cfg)
        # ── RANDOMIZED-TARGET/DD TRAINING MODE (target_aware_policy.md item 2) ───
        # When RANDOMIZE_FTMO_INPUTS is ON, reset() samples target_pct / max_dd_pct
        # (and optionally account_size) PER EPISODE from the ranges below; the env
        # then uses the sampled values for THAT episode's classification/DD AND
        # exposes them in the observation (item 1). DEFAULT OFF — existing
        # curriculum is unchanged; the fixed cfg values are used and still appear
        # (constant) in the obs so inference-time changes still shift behaviour.
        self.randomize_ftmo = bool(cfg.get("RANDOMIZE_FTMO_INPUTS", False))
        self._target_range = tuple(cfg.get("RANDOMIZE_TARGET_RANGE", [0.01, 0.05]))
        self._dd_range = tuple(cfg.get("RANDOMIZE_DD_RANGE", [0.005, 0.02]))
        self._randomize_account = bool(cfg.get("RANDOMIZE_FTMO_ACCOUNT", False))
        # Observation schema version (validated against checkpoints on resume).
        self.obs_schema_version = OBS_SCHEMA_VERSION
        # ── ACCOUNT SIZE (learning_loop_fix.md FIX 3) ────────────────────────
        # Default starting equity is $10,000 (configurable via CLI --account-size
        # or CFG["INITIAL_EQUITY"] / CFG["ACCOUNT_SIZE"], supporting 10k/25k/50k/
        # 100k). FTMO targets/limits stay PERCENTAGE-based off start-of-day equity
        # so they scale automatically; reward is normalized so it is account-size
        # invariant. NOTHING here is hard-coded to 100k.
        #
        # FUTURE HOOK (documented, DISABLED): to teach size-relative risk, a per-
        # episode random account size could be drawn in reset() from
        # cfg["ACCOUNT_SIZE_CHOICES"] when cfg["RANDOMIZE_ACCOUNT_SIZE"] is True.
        # Left off for now — see reset() for the commented stub.
        self.initial_equity = float(
            cfg.get("ACCOUNT_SIZE", cfg.get("INITIAL_EQUITY", 10_000.0)))
        # ── FIXED DAILY PROFIT INCREMENT (ftmo_rules_fix.md RULE 1) ──────────────
        # The FTMO daily increment is a FIXED DOLLAR amount computed ONCE at account
        # open: initial_equity * target_pct. On a $10,000 account @ 2.5% that is a
        # flat $250 — always a fraction of the ORIGINAL initial equity. It is the
        # unit for the EXCEED progressive bonus and the diagnostic `daily_target`
        # (day_start + increment) still reported in info.
        #
        # REFINED CLASSIFICATION (dd_classification_refine.md): the PASS/OK tier
        # DECISION is measured against INITIAL equity, NOT the day's open:
        #     PASS iff final_or_halt >= initial*(1+target_pct)   (>= +2.5%)
        #     OK   iff final_or_halt >= initial*(1+half_target_pct) (>= +1.25%)
        #     FAIL_CAPITAL_LOSS iff final_or_halt < prior_day close (checked FIRST)
        # See the SINGLE-SOURCE-OF-TRUTH principles block below and step().
        self.daily_increment = self.initial_equity * self.target_pct
        self.bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
        self.max_lot = float(cfg.get("MAX_LOT", 2.0))
        self.direction_dim = DIRECTION_DIM

        # ── REWARD weights + Section-6/7/8 config (all from CFG, never hardcoded) ──
        self._rw = cfg.get("REWARD", {}) or {}
        # Section 2 streak-curve coefficients + state-machine weights (shared with
        # core/reward/shaper.py — the env mirrors that scalar reference vectorized).
        from core.reward.shaper import _STREAK_A_DEFAULT, _STREAK_B_DEFAULT
        self._streak_a = float(self._rw.get("streak_curve_a", _STREAK_A_DEFAULT))
        self._streak_b = float(self._rw.get("streak_curve_b", _STREAK_B_DEFAULT))
        # Section 5 commission: resolve THIS instrument's per-round-trip $/std-lot
        # once (EURUSD is forex -> $5/lot RT). Forex is a flat per-lot cost so the
        # hot path multiplies a scalar by lots; non-forex classes fall back to the
        # pure resolve_commission() (price-dependent) when used.
        self._commission_class = classify_symbol(self.instrument, cfg)
        comm_spec = (cfg.get("COMMISSION", {}) or {}).get(
            self._commission_class, {"kind": "zero", "value": 0.0})
        self._comm_kind = comm_spec.get("kind", "zero")
        self._comm_value = float(comm_spec.get("value", 0.0))
        # ── OPTIONAL ENTRY FRICTION (audit P1: train-vs-live entry-cost gap) ──
        # When enabled, the entry price is worsened by half the spread plus full
        # slippage, in the trade's direction — matching the entry leg of
        # intrabar_fills.compute_fill (BUY pays up, SELL is filled down). DEFAULT
        # OFF so existing trade economics and the hand-calc PnL tests are
        # unchanged; flip cfg["ENTRY_FRICTION_ENABLED"]=True to train against the
        # realistic, more-pessimistic entry. pip/spread/slippage come from the
        # instrument's trading_policy.yaml entry (same source compute_fill uses),
        # so nothing is hardcoded.
        self._entry_friction_enabled = bool(cfg.get("ENTRY_FRICTION_ENABLED", False))
        _instr = {**{"pip_value": 0.0001, "spread_pips": 1.0, "slippage_pips": 0.5},
                  **((self.policy.get("instrument_settings", {}) or {})
                     .get(self.instrument, {}))}
        self._friction_px = float(
            _instr["pip_value"]
            * (0.5 * float(_instr["spread_pips"]) + float(_instr["slippage_pips"])))
        # Section 6 sessions (synthetic CEST clock) + filter.
        self._sessions = list(cfg.get("TRADING_SESSIONS", []))
        self._n_sessions = float(cfg.get("N_SESSIONS", 4))
        self._session_day_open = float(cfg.get("SESSION_DAY_OPEN_MIN", 120.0))
        # Section 7 speed-bonus window (in bars == minutes on M1).
        self._speed_bonus = float(self._rw.get("speed_bonus", 0.0))
        self._speed_window = int(cfg.get("SPEED_BONUS_MINUTES", 3))
        # Section 8 lot curriculum: resolve the [lo, hi] clamp window for the active
        # strategy phase (widens as the curriculum advances). Beast mode lifts the
        # narrowing (cap == BEAST_MAX_LOT). Recomputed when the phase changes.
        self._lot_curriculum_enabled = bool(cfg.get("LOT_CURRICULUM_ENABLED", True))
        self._beast_mode = bool(cfg.get("BEAST_MODE", False))

        # Per-timeframe indicator rows for phase gating (populated below or from
        # the cache). Phase masks gate on TF pairs (e.g. [1m,15m]); the conditions
        # engine reads named indicator columns per bar from these DataFrames.
        self._tf_indicators: Dict[int, pd.DataFrame] = {}
        self._raw_ohlcv = self._extract_raw_ohlcv(features)

        # ── Build (or LOAD-FROM-CACHE) the feature matrix + TF indicators ─────
        # The feature build (1.9M bars -> indicators) is a deterministic pure
        # function of (CSV bytes, feature-config), so it is memoized to disk via
        # core/env/feature_cache (learning_loop_fix.md FIX 4.1). On restart a
        # valid cache loads in seconds instead of a ~10-15 min rebuild. The cache
        # only engages when a CSV path is known (cfg["DATA_CSV_EURUSD"]); for raw
        # arrays / synthetic fixtures (tests) we build directly with no caching.
        from core.env.feature_cache import build_or_load

        def _build():
            feat_t = self._ensure_feature_matrix(features).to("cpu")
            if self._raw_ohlcv is not None:
                self._build_tf_indicators()
            return feat_t, dict(self._tf_indicators)

        csv_path = cfg.get("DATA_CSV_EURUSD")
        if csv_path and cfg.get("FEATURES") is None:
            feat, tf_ind = build_or_load(csv_path, cfg, _build)
            self._tf_indicators = tf_ind
        else:
            feat, _tf = _build()      # tests / synthetic — no CSV, no cache
        self.features = feat.to(device=device, dtype=torch.float32)
        self.T, self.F = self.features.shape

        # episode length: bounded by data; default ~ a few days for dev/CPU
        self.ep_bars = min(int(cfg.get("EPISODE_BARS", 43_200)),
                           max(self.bars_per_day, self.T - self.lkbk - 2))

        # ── CHRONOLOGICAL START-WINDOW (train/val separation) ────────────────
        # reset() samples each episode's start bar uniformly inside this fraction
        # [lo, hi) of the dataset. DEFAULT (0.0, 1.0) = the whole series, which is
        # the historical behaviour. The trainer / run_eval set DISJOINT windows
        # (e.g. train=[0,0.8), val=[0.8,1.0)) so the validation pass-rate is
        # genuinely OUT-OF-SAMPLE and never overlaps training starts. The split is
        # config-driven (EVAL_SPLIT_FRAC) and never hardcoded into reset().
        self._start_lo_frac = 0.0
        self._start_hi_frac = 1.0

        # state_dim = lookback window (lkbk*F) + v1 position/account features
        # (N_POSITION_FEATS) + v2 target/risk-aware features (N_FTMO_FEATS, item 1)
        # + v3 session/context features (N_SESSION_FEATS, Section 6).
        self.state_dim = (self.lkbk * self.F + N_POSITION_FEATS + N_FTMO_FEATS
                          + N_SESSION_FEATS)
        self._alloc_episode_tensors()
        self._refresh_lot_window()

        if self._tf_indicators:
            print(f"[env] TF indicators ready for timeframes: {sorted(self._tf_indicators.keys())} "
                  f"({0 if self._raw_ohlcv is None else len(self._raw_ohlcv)} bars raw OHLCV)",
                  flush=True)
        elif self._raw_ohlcv is None:
            print("[env] WARNING: no raw OHLCV available — phase gate masks DISABLED. "
                  "Pass raw (N,5) OHLCV, not a prebuilt feature matrix.", flush=True)

        # ── Precompute the vectorized gate signal for the initial phase ──
        # Assigning through the property setter runs _refresh_gate_signal(), which
        # builds the length-T per-bar gate tensor ONCE. From here on the hot loop
        # only does a tensor gather (gate_on[abs_idx]) — no pandas per bar.
        self.phase = self._phase

    # ── phase property: changing the phase re-precomputes the gate signal ──────
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        """Setting env.phase (e.g. train.py's run_phase) swaps the active phase
        AND rebuilds the vectorized per-bar gate signal for it. This is the only
        place the (potentially expensive) precompute runs — never in the hot
        per-step loop."""
        self._phase = new_phase or {"entry_conditions": {"buy": "any", "sell": "any"}}
        self._refresh_gate_signal()
        self._refresh_lot_window()

    def _refresh_lot_window(self):
        """(Re)resolve the Section-8 lot-curriculum [lo, hi] clamp for the active
        strategy phase. The clamp is applied ON TOP of the PPO lot head (which keeps
        its full [0.01, MAX_LOT] range): early phases trade NARROW (learn direction
        first), later phases widen toward the full head. Beast mode (or a phase
        flagged "beast"/"live") lifts the narrowing and clamps only to BEAST_MAX_LOT.
        All windows come from CFG["LOT_CURRICULUM"]; nothing is hardcoded."""
        cfg = self.cfg
        cur = cfg.get("LOT_CURRICULUM", {}) or {}
        default_win = cur.get("_default", [0.10, 0.50])
        phase_name = (self._phase or {}).get("name", "") if isinstance(self._phase, dict) else ""
        beast = self._beast_mode or phase_name in ("beast", "live_improve")
        if not self._lot_curriculum_enabled:
            lo, hi = 0.01, self.max_lot          # curriculum off -> full head
        elif beast:
            lo, hi = 0.01, float(cfg.get("BEAST_MAX_LOT", self.max_lot))
        else:
            win = cur.get(phase_name, default_win)
            lo, hi = float(win[0]), float(win[1])
        # Never exceed the head ceiling; keep lo < hi.
        hi = min(hi, self.max_lot)
        lo = min(lo, hi)
        self._lot_lo, self._lot_hi = float(lo), float(hi)

    def _refresh_gate_signal(self):
        """(Re)build the length-T per-bar gate tensor for the active phase.

        Guarded so it is a no-op until the feature matrix + TF indicators exist
        (during __init__ the setter may run before those are ready — though we
        intentionally assign self.phase only at the END of __init__, this guard
        keeps the method safe to call at any time)."""
        if not hasattr(self, "features"):
            self._gate_signal = None
            return
        tf_ind = getattr(self, "_tf_indicators", {}) or {}
        self._gate_signal = precompute_gate_signal(
            self._phase, tf_ind, self.features, self.T, self.device)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ensure_feature_matrix(self, features) -> torch.Tensor:
        """Accept either a prebuilt (N,F) feature matrix or raw (N,5) OHLCV."""
        t = features if isinstance(features, torch.Tensor) else torch.as_tensor(
            np.asarray(features, dtype=np.float32))
        if t.ndim == 2 and t.shape[1] == NUM_FEATURES:
            return t                                   # already a feature matrix
        if t.ndim == 2 and t.shape[1] >= 5:            # raw OHLCV -> build features
            o, h, l, c, v = (t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4])
            return build_feature_matrix(o, h, l, c, v, self.device)
        raise ValueError(f"Unexpected features shape {tuple(t.shape)}")

    def _alloc_episode_tensors(self):
        d, B = self.device, self.B
        self._start = torch.zeros(B, dtype=torch.long, device=d)
        self._step_i = torch.zeros(B, dtype=torch.long, device=d)
        # REALIZED balance (cash). Only changes on close/reverse/commission — NEVER
        # carries unrealized mark-to-market. self._equity is the MARKED equity
        # (balance + open-position MTM) recomputed fresh each bar from this balance;
        # folding MTM back into the balance would double-count it every held bar.
        self._balance = torch.full((B,), self.initial_equity, device=d)
        self._equity = torch.full((B,), self.initial_equity, device=d)
        self._day_start_eq = torch.full((B,), self.initial_equity, device=d)
        self._day_high_eq = torch.full((B,), self.initial_equity, device=d)
        # ── PER-EPISODE FTMO INPUTS (item 1 + item 2) ────────────────────────
        # The env classifies/DD-checks and builds the observation off these
        # per-episode tensors. When RANDOMIZE_FTMO_INPUTS is OFF they hold the
        # constant scalar cfg values (so the obs still carries them and inference
        # changes still shift behaviour); when ON, reset() resamples them per
        # episode. _initial_equity_t supports optional per-episode account-size
        # randomization so the fixed-dollar daily increment scales with it.
        self._initial_equity_t = torch.full((B,), self.initial_equity, device=d)
        self._target_pct_t = torch.full((B,), self.target_pct, device=d)
        self._max_dd_pct_t = torch.full((B,), self.max_dd_pct, device=d)
        self._daily_increment_t = self._initial_equity_t * self._target_pct_t
        # OK-tier half-target fraction per episode (dd_classification_refine.md).
        # _half_target_explicit pins it; otherwise it DERIVES as target/2 even when
        # the per-episode target is randomized (so OK stays at exactly 50%).
        self._half_target_explicit = self.cfg.get("DAILY_HALF_TARGET_PCT", None)
        self._half_target_pct_t = self._derive_half_target_t(self._target_pct_t)
        self._position = torch.zeros(B, device=d)       # +lots buy / -lots sell / 0
        self._entry_px = torch.zeros(B, device=d)
        self._dd_breached = torch.zeros(B, dtype=torch.bool, device=d)
        self._trades_today = torch.zeros(B, dtype=torch.long, device=d)
        # Per-day WIN / LOSS trade counters. A trade is counted the moment its PnL
        # is REALIZED (an exit-close, an exit-reduce, or a flip that closes the old
        # position): realized > 0 -> win, realized < 0 -> loss (exactly-flat skipped).
        # Snapshotted on day-close like _trades_today and reported per day so the
        # logs show won vs lost trades, not just pass/fail DAYS.
        self._wins_today = torch.zeros(B, dtype=torch.long, device=d)
        self._losses_today = torch.zeros(B, dtype=torch.long, device=d)
        self._done = torch.zeros(B, dtype=torch.bool, device=d)
        # PPO/day-reward state
        self._day_halted = torch.zeros(B, dtype=torch.bool, device=d)  # DD-ended day
        self._day_passed = torch.zeros(B, dtype=torch.bool, device=d)
        self._pass_streak = torch.zeros(B, dtype=torch.long, device=d)
        self._equity_prev = torch.full((B,), self.initial_equity, device=d)
        # tracks bars where the gate was active this day (condition TRUE)
        self._gate_bars_today = torch.zeros(B, dtype=torch.long, device=d)

        # ── Section 2 STREAK STATE MACHINE (vectorized mirror of shaper.StreakTracker) ──
        # consec_fail_count drives the mulligan (1 free fail; 2 consecutive break the
        # pass streak). fail_streak is the length of the current losing run. last_was
        # _pass carries the momentum bias into the next day. _signed_streak is the
        # signed run length exposed in the observation (S6.6): +pass_streak / -fail_streak.
        self._consec_fail = torch.zeros(B, dtype=torch.long, device=d)
        self._fail_streak = torch.zeros(B, dtype=torch.long, device=d)
        self._last_was_pass = torch.zeros(B, dtype=torch.bool, device=d)
        self._signed_streak = torch.zeros(B, dtype=torch.long, device=d)
        # best pass-streak reached this EPISODE (Section 4.2 composite episode bonus)
        self._best_streak = torch.zeros(B, dtype=torch.long, device=d)

        # ── Section 3 give-back protection ──
        # _day_open_progress_reward accrues the intra-day progress shaping (S3.1) so
        # a FAIL day can retroactively WIPE it (S3.2). _multi_day_peak is the running
        # cross-episode/day equity peak for the cross-day give-back penalty (S3.3).
        self._day_progress_reward = torch.zeros(B, device=d)
        # last bar's clamped progress-to-target (for the per-bar increment, S3.1)
        self._day_progress_prev = torch.zeros(B, device=d)
        self._intraday_high_eq = torch.full((B,), self.initial_equity, device=d)
        self._multi_day_peak = torch.full((B,), self.initial_equity, device=d)

        # ── Section 7 speed bonus ──
        # _entry_bar records the bar index a position was opened; _speed_pending holds
        # the accrued-but-unconfirmed speed bonus (kept on a profitable close, else
        # revoked). _speed_armed marks a position that showed green within the window.
        self._entry_bar = torch.zeros(B, dtype=torch.long, device=d)
        self._speed_pending = torch.zeros(B, device=d)
        self._speed_armed = torch.zeros(B, dtype=torch.bool, device=d)
        # GLOBAL day index — the calendar-day counter shared by ALL B episodes.
        # Because reset() zeroes _step_i for every episode and step() increments
        # it in lockstep, `new_day = (_step_i % bars_per_day) == 0` fires on the
        # SAME step for all episodes. _day_idx therefore aligns every episode to
        # the same trading day, which is what lets the trainer aggregate one
        # honest per-day line across the whole batch (Bug A fix). A DD-halt does
        # NOT advance _day_idx (the calendar day still spans bars_per_day bars);
        # it only closes that episode's day early for classification.
        self._day_idx = torch.zeros(B, dtype=torch.long, device=d)

    def _abs_idx(self) -> torch.Tensor:
        return (self._start + self._step_i).clamp(0, self.T - 1)

    # ── multi-timeframe indicator support ────────────────────────────────────
    def _extract_raw_ohlcv(self, features):
        """Return a 1m OHLCV DataFrame from raw (N,>=5) input, or None if a
        prebuilt feature matrix was passed (no raw OHLC available).

        The DataFrame carries a plain RangeIndex (0..N-1). A DatetimeIndex is
        synthesized ONLY transiently inside _build_tf_indicators for pandas'
        resample(); TF-to-1m alignment is done by INTEGER row position
        (1m row i -> TF bar i // tf), never by timestamp searchsorted. This
        avoids the fake-timestamp drift that produced impossible base_ts values.
        """
        t = features if isinstance(features, torch.Tensor) else torch.as_tensor(
            np.asarray(features, dtype=np.float32))
        if t.ndim == 2 and t.shape[1] >= 5 and t.shape[1] != NUM_FEATURES:
            arr = t[:, :5].detach().cpu().numpy()
            return pd.DataFrame(arr, columns=["open", "high", "low", "close",
                                              "volume"])
        return None

    def _build_tf_indicators(self):
        """Build indicator DataFrames for all required timeframes by resampling
        the 1m OHLCV bars. The raw 1m data from the CSV is the single source of
        truth — all higher timeframes (15m, 30m, 60m, etc.) are derived from it.
        No separate higher-TF CSV is needed.

        pandas' resample() needs a DatetimeIndex, so we attach a synthetic 1-min
        DatetimeIndex *transiently* just for the resample step, then drop it: the
        resulting per-TF indicator frames carry a plain RangeIndex so alignment
        back to the 1m bar is done purely by integer position (see _rows_by_tf)."""
        tfs = set([1])
        gt = (self.phase or {}).get("gate_timeframes", []) or []
        tfs.update(gt)
        tfs.update([15, 30, 60])   # all phase gate TFs used across the curriculum
        # Transient DatetimeIndex (resample-only); never used for alignment.
        dt_raw = self._raw_ohlcv.copy()
        dt_raw.index = pd.date_range("2020-01-01", periods=len(dt_raw), freq="1min")
        for tf in tfs:
            try:
                df_tf = resample_ohlcv(dt_raw, tf)
                if len(df_tf) > 0:
                    ind = compute_indicators(df_tf).reset_index(drop=True)
                    self._tf_indicators[tf] = ind
            except Exception as e:
                print(f"[env] WARNING: failed to build TF{tf} indicators: {e}", flush=True)

    def _tf_pos(self, raw_i: int, tf: int, tf_len: int) -> int:
        """Integer-based TF alignment: 1m row `raw_i` belongs to the higher-TF
        bar covering it, i.e. position `raw_i // tf` (most recent completed bar
        at/under the current 1m bar). Clamped to the resampled length. No
        timestamp searchsorted — alignment is purely positional, so it can never
        drift to an impossible date."""
        pos = raw_i // max(tf, 1)
        return max(0, min(pos, tf_len - 1))

    def _rows_by_tf_batch(self, abs_idx: torch.Tensor) -> Dict[int, list]:
        """Return {tf_minutes: [feature_row_dict per episode]} aligned PER EPISODE
        to each episode's current 1m bar via integer row indexing. Length of each
        list == B. Empty dict if no raw OHLCV / TF indicators are available."""
        if not self._tf_indicators:
            return {}
        raw_idx = abs_idx.clamp(0, len(self._raw_ohlcv) - 1).detach().cpu().tolist()
        out: Dict[int, list] = {}
        for tf, df_ind in self._tf_indicators.items():
            tf_len = len(df_ind)
            recs = df_ind.to_dict("records")   # list of row dicts, indexed positionally
            out[tf] = [recs[self._tf_pos(int(ri), tf, tf_len)] for ri in raw_idx]
        return out

    def _rows_by_tf(self) -> Dict[int, dict]:
        """Episode-0 view (compat shim for callers/tests that want a single bar's
        rows). Delegates to the batched path and returns episode 0's dicts."""
        batch = self._rows_by_tf_batch(self._abs_idx())
        return {tf: rows[0] for tf, rows in batch.items()}

    # ── train/val start-window control ──────────────────────────────────────────
    def set_start_window(self, lo_frac: float = 0.0, hi_frac: float = 1.0) -> None:
        """Restrict reset()'s episode-start sampling to the dataset fraction
        [lo_frac, hi_frac). Used to keep training and evaluation on DISJOINT,
        chronological slices so the eval pass-rate is out-of-sample. Clamped to
        [0, 1] with lo < hi enforced. (0.0, 1.0) restores full-range sampling."""
        lo = min(max(float(lo_frac), 0.0), 1.0)
        hi = min(max(float(hi_frac), 0.0), 1.0)
        if hi <= lo:                       # guard against an empty/inverted window
            lo, hi = 0.0, 1.0
        self._start_lo_frac, self._start_hi_frac = lo, hi

    def _derive_half_target_t(self, target_pct_t: torch.Tensor) -> torch.Tensor:
        """Per-episode OK-tier half-target fraction (dd_classification_refine.md).

        When DAILY_HALF_TARGET_PCT is pinned, use that constant; when it is None,
        DERIVE it as half of THIS episode's (possibly randomized) target_pct so the
        OK threshold always sits at exactly 50% of the full target. Config-driven —
        nothing hardcodes 0.0125."""
        if self._half_target_explicit is None:
            return target_pct_t * 0.5
        return torch.full_like(target_pct_t, float(self._half_target_explicit))

    # ── reset ──────────────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        warmup = self.lkbk + 25
        # Full admissible start range [warmup, max_start). The active start-window
        # fraction then carves a chronological sub-range out of it (default = all).
        max_start = max(warmup + 1, self.T - self.ep_bars - 1)
        span = max_start - warmup
        lo = warmup + int(self._start_lo_frac * span)
        hi = warmup + int(self._start_hi_frac * span)
        lo = min(max(lo, warmup), max_start)
        hi = min(max(hi, lo + 1), max(warmup + 1, max_start))
        self._start = torch.randint(lo, max(lo + 1, hi),
                                    (self.B,), device=self.device)
        self._step_i.zero_()
        self._balance.fill_(self.initial_equity)
        self._equity.fill_(self.initial_equity)
        self._day_start_eq.fill_(self.initial_equity)
        self._day_high_eq.fill_(self.initial_equity)
        self._position.zero_()
        self._entry_px.zero_()
        self._dd_breached.zero_()
        self._trades_today.zero_()
        self._wins_today.zero_()
        self._losses_today.zero_()
        self._done.zero_()
        self._day_halted.zero_()
        self._day_passed.zero_()
        # NOTE: _pass_streak intentionally persists across episodes within a phase
        # (DESIGN_DECISIONS.md #7 — consecutive pass-days counter is phase-level).
        # The full Section-2 streak machine (consec_fail/fail_streak/last_was_pass/
        # signed_streak) ALSO persists with it so mulligan/momentum carry across the
        # episode boundary like the pass streak does. _best_streak is PER EPISODE
        # (it scores "the best run THIS episode" for the S4.2 composite bonus), so it
        # resets here.
        self._best_streak.zero_()
        self._equity_prev.fill_(self.initial_equity)
        self._gate_bars_today.zero_()
        self._day_idx.zero_()

        # Section 3 / 7 per-episode state (intra-day + speed). The multi-day peak is
        # phase-level (protects gains ACROSS episodes), so it is NOT reset here.
        self._day_progress_reward.zero_()
        self._day_progress_prev.zero_()
        self._intraday_high_eq.fill_(self.initial_equity)
        self._entry_bar.zero_()
        self._speed_pending.zero_()
        self._speed_armed.zero_()

        # ── PER-EPISODE FTMO INPUTS (target_aware_policy.md item 2) ──────────
        # DEFAULT: hold the constant scalar cfg values (so the obs still carries
        # target/DD and inference-time changes still shift behaviour). When
        # RANDOMIZE_FTMO_INPUTS is ON, sample a fresh (target_pct, max_dd_pct[,
        # account_size]) per EPISODE — the env uses these sampled values for THIS
        # episode's PASS/FAIL + DD-halt AND exposes them in the observation, which
        # is what teaches the policy to CONDITION on the inputs.
        d, B = self.device, self.B
        if self.randomize_ftmo:
            tlo, thi = self._target_range
            dlo, dhi = self._dd_range
            self._target_pct_t = (torch.rand(B, device=d) * (thi - tlo) + tlo)
            self._max_dd_pct_t = (torch.rand(B, device=d) * (dhi - dlo) + dlo)
            if self._randomize_account:
                choices = torch.tensor(
                    self.cfg.get("ACCOUNT_SIZE_CHOICES",
                                 [10_000., 25_000., 50_000., 100_000.]),
                    device=d, dtype=torch.float32)
                pick = choices[torch.randint(len(choices), (B,), device=d)]
                self._initial_equity_t = pick
                self._balance = pick.clone()
                self._equity = pick.clone()
                self._day_start_eq = pick.clone()
                self._day_high_eq = pick.clone()
                self._equity_prev = pick.clone()
                self._intraday_high_eq = pick.clone()
                self._multi_day_peak = pick.clone()
            else:
                self._initial_equity_t.fill_(self.initial_equity)
        else:
            self._target_pct_t.fill_(self.target_pct)
            self._max_dd_pct_t.fill_(self.max_dd_pct)
            self._initial_equity_t.fill_(self.initial_equity)
        # Fixed daily increment = initial_equity * target_pct, per episode (RULE 1).
        self._daily_increment_t = self._initial_equity_t * self._target_pct_t
        # OK-tier half target tracks the (possibly resampled) target per episode.
        self._half_target_pct_t = self._derive_half_target_t(self._target_pct_t)
        return self._get_state()

    # ── state ──────────────────────────────────────────────────────────────────
    def _get_state(self) -> torch.Tensor:
        abs_idx = self._abs_idx()
        offsets = torch.arange(self.lkbk - 1, -1, -1, device=self.device)
        win_idx = (abs_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp(0, self.T - 1)
        window = self.features[win_idx]                       # (B, lkbk, F)
        mu = window.mean(dim=(1, 2), keepdim=True)
        std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        norm = ((window - mu) / std).reshape(self.B, -1)      # (B, lkbk*F)

        close = self.features[abs_idx, COL["close"]]
        unrealised = torch.where(
            self._position != 0,
            (close - self._entry_px) * torch.sign(self._position),
            torch.zeros_like(close))
        init_eq = self._initial_equity_t
        eq_chg = (self._equity - init_eq) / (init_eq + 1e-8)
        # Daily target = day's opening equity + the FIXED increment (RULE 1), NOT
        # day_start * (1 + target_pct). The increment is a flat dollar amount.
        target_eq = self._day_start_eq + self._daily_increment_t
        gap = (target_eq - self._equity) / (init_eq + 1e-8)
        dd_used = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        dd_head = (self._max_dd_pct_t - dd_used).clamp(min=0.0)
        daily_ret = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8)
        # v1 position/account features (N_POSITION_FEATS).
        pos_feat = torch.stack([self._position, unrealised, eq_chg, gap, dd_head,
                                daily_ret], dim=1)

        # ── v2 TARGET/RISK-AWARE FEATURES (target_aware_policy.md item 1) ────
        # All finite, O(1)-scaled, and built from the PER-EPISODE FTMO inputs so a
        # randomized-training episode and an inference-time --target-pct/--max-dd
        # change both flow through identically. Each feature is commented below.
        #
        # (a) target_pct — the active daily profit target as a fraction of initial
        #     equity (already O(1), e.g. 0.025). Lets the net see "how big a day".
        f_target = self._target_pct_t
        # (b) max_dd_pct — the active daily trailing-DD limit as a fraction
        #     (e.g. 0.01). Lets the net see "how much risk budget the regime gives".
        f_maxdd = self._max_dd_pct_t
        # (c) difficulty = daily_increment / day_start_equity — today's required
        #     gain as a fraction of TODAY's opening balance (how hard today is;
        #     grows as the account compounds since the increment is fixed dollars).
        f_difficulty = self._daily_increment_t / (self._day_start_eq + 1e-8)
        # (d) progress_to_target = clamp((equity - day_start)/daily_increment,0,1)
        #     — how close to hitting today's target (1.0 = target reached).
        f_progress = ((self._equity - self._day_start_eq)
                      / (self._daily_increment_t + 1e-8)).clamp(min=0.0, max=1.0)
        # (e) dd_headroom = (equity - peak*(1-max_dd))/(peak*max_dd) — fraction of
        #     the DD budget still available right now (1.0 = full room, 0 = at
        #     breach). Clamped to [0,1] so a breach reads 0, not negative.
        peak = self._day_high_eq
        dd_floor = peak * (1.0 - self._max_dd_pct_t)
        f_dd_headroom = ((self._equity - dd_floor)
                         / (peak * self._max_dd_pct_t + 1e-8)).clamp(min=0.0, max=1.0)
        # (f) fraction_of_day_remaining = (bars_per_day - bars_elapsed_today)/
        #     bars_per_day — time pressure within the trading day (1.0 = day start).
        bars_elapsed = (self._step_i % self.bars_per_day).float()
        f_day_remaining = (1.0 - bars_elapsed / float(self.bars_per_day)).clamp(
            min=0.0, max=1.0)
        # (g) account_size (log-normalized) so size-relative risk is visible. We
        #     normalize log10(equity) around a $10k reference and /1.0 decade so a
        #     10k->100k span maps to ~[0,1]; finite and O(1).
        f_acct = (torch.log10(init_eq.clamp(min=1.0)) - 4.0)
        ftmo_feat = torch.stack([f_target, f_maxdd, f_difficulty, f_progress,
                                 f_dd_headroom, f_day_remaining, f_acct], dim=1)

        # ── v3 SESSION/CONTEXT FEATURES (Section 6) ──────────────────────────
        # All finite, O(1)-scaled, built from the synthetic CEST clock + state so
        # the policy can learn intraday timing, session edges, give-back protection,
        # and per-symbol cost. Each commented below.
        bars_elapsed_i = (self._step_i % self.bars_per_day).float()
        # minute-of-day on the synthetic CEST clock: day opens at SESSION_DAY_OPEN
        # and advances one minute per bar (M1), wrapping at 1440.
        minute_of_day = torch.remainder(
            self._session_day_open + bars_elapsed_i
            * (1440.0 / float(self.bars_per_day)), 1440.0)
        # (6.1) time-of-day in CEST, normalized to [0,1] (0=00:00, 1=24:00).
        s_tod = minute_of_day / 1440.0
        # (6.2) progress toward daily target (% achieved, clamped [0,1]). Same as
        #       the FTMO f_progress above but kept here as the Section-6 obs too so
        #       the schema is self-describing; cheap to duplicate.
        s_progress = f_progress
        # (6.3) remaining time in the trading day (1.0 at open -> 0 at close).
        s_remaining = (1.0 - bars_elapsed_i / float(self.bars_per_day)).clamp(0.0, 1.0)
        # (6.4) session label as a normalized code in [0,1] (overlap>NY>London>Asian;
        #       0 outside all sessions). Vectorized lookup over TRADING_SESSIONS.
        s_session = torch.zeros(self.B, device=self.device)
        for _name, start, end, code in self._sessions:
            in_sess = (minute_of_day >= float(start)) & (minute_of_day < float(end))
            # first matching row wins -> only fill where still 0 (overlap listed first)
            s_session = torch.where(in_sess & (s_session == 0),
                                    torch.full_like(s_session,
                                                    float(code) / self._n_sessions),
                                    s_session)
        # (6.5) DD budget remaining (% of max DD still available right now, [0,1]).
        #       Reuses f_dd_headroom (already the fraction of budget left).
        s_dd_budget = f_dd_headroom
        # (6.6) current streak length, SIGNED and normalized: +pass / -fail run,
        #       squashed by /10 then clamped to [-1.5,1.5] so a long run stays O(1).
        s_streak = (self._signed_streak.float() / 10.0).clamp(min=-1.5, max=1.5)
        # (6.7) commission cost for the CURRENT symbol at a reference 1.0 lot,
        #       normalized by day-start equity (so it is account-size invariant and
        #       O(1)). Forex EURUSD: $5 RT / $10k ≈ 0.0005. Lets the net price-in cost.
        ref_comm = self._commission_per_lot_round_trip(close)   # (B,) $ per 1.0 lot RT
        s_comm = ref_comm / (self._day_start_eq + 1e-8)
        session_feat = torch.stack([s_tod, s_progress, s_remaining, s_session,
                                    s_dd_budget, s_streak, s_comm], dim=1)
        return torch.cat([norm, pos_feat, ftmo_feat, session_feat], dim=1)

    # ── Section 8 lot curriculum mapping (head [0,1] -> [lot_lo, lot_hi]) ─────
    def _map_lot_curriculum(self, lot_raw: torch.Tensor) -> torch.Tensor:
        """Map the PPO lot head's raw [0,1] onto the active phase's curriculum
        window [lot_lo, lot_hi] (Section 8). Early phases narrow the EFFECTIVE size;
        later phases / beast widen toward the full [0.01, MAX_LOT] head. The window
        is resolved by _refresh_lot_window() on phase change."""
        lo, hi = self._lot_lo, self._lot_hi
        return lo + lot_raw.clamp(0, 1) * (hi - lo)

    # ── Section 5 commission helpers (vectorized for the active forex path) ───
    def _commission_per_lot_round_trip(self, price: torch.Tensor) -> torch.Tensor:
        """(B,) commission in $ for a 1.0-lot ROUND TRIP of THIS instrument at the
        given price. Forex: flat self._comm_value (price-independent). pct_notional
        (metals/crypto): value * (1.0*contract*price) * 2 sides. zero otherwise.
        Vectorized so it can feed the obs + the per-trade deduction on the hot path."""
        if self._comm_kind == "zero" or self._comm_value == 0.0:
            return torch.zeros_like(price)
        if self._comm_kind == "per_lot_round_trip":
            return torch.full_like(price, self._comm_value)
        if self._comm_kind == "pct_notional":
            contract = float(self.cfg.get("CONTRACT_SIZE", 100_000.0))
            return self._comm_value * contract * price * 2.0
        return torch.zeros_like(price)

    def _commission_for_lots(self, lots: torch.Tensor, price: torch.Tensor,
                             side: str) -> torch.Tensor:
        """(B,) commission in $ for trading `lots` of THIS instrument at `price` for
        one SIDE ("open"/"close"). Scales with lots (Section 5.2) and is charged at
        BOTH open and close (Section 5.3). per_lot_round_trip is split in half per
        side; pct_notional is one side of the notional cost."""
        if self._comm_kind == "zero" or self._comm_value == 0.0:
            return torch.zeros_like(lots)
        if self._comm_kind == "per_lot_round_trip":
            return 0.5 * self._comm_value * lots.abs()
        if self._comm_kind == "pct_notional":
            contract = float(self.cfg.get("CONTRACT_SIZE", 100_000.0))
            return self._comm_value * lots.abs() * contract * price
        return torch.zeros_like(lots)

    # ── action mask (RULE 12) — VECTORIZED (stall fix) ─────────────────────────
    #
    # PERFORMANCE NOTE (the 4-hour phase0 stall fix):
    #   The masks below are built from the PRECOMPUTED per-bar gate signal
    #   (self._gate_signal, length T) with pure tensor ops — a single gather plus
    #   a few torch.where calls, all O(B). The old implementation rebuilt pandas
    #   row-dicts every bar (to_dict("records")) and looped in Python over all B
    #   episodes; profiling showed ~90% of step time there. See
    #   core/env/gate_precompute.py for the precompute, and
    #   tests/unit/test_gate_precompute.py for the bar-for-bar parity proof that
    #   this fast path returns identical masks to the original scalar path.

    def _allow_all_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, DIRECTION_DIM) all-ones mask + all-False must_enter (free phase /
        no gating). Allocated fresh each call (callers may store it)."""
        m = torch.ones(self.B, DIRECTION_DIM, dtype=torch.float32, device=self.device)
        must = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        return m, must

    def current_mask_and_force(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-episode gate mask. Returns:
          dir_mask   : (B, DIRECTION_DIM) float — 1.0 allowed / 0.0 masked, computed
                       independently for EVERY one of the B episodes (each episode
                       has its own bar index and own position state).
          must_enter : (B,) bool — True where a force_in_and_gate/open_gate gate is
                       ON and the episode is flat (agent must open BUY or SELL).

        BUG-4 (still honored): the mask is per-episode — each episode uses its own
        current bar index (abs_idx) and its own flat/in-trade state.

        Implementation: gather the precomputed per-bar gate booleans at each
        episode's current bar, then derive the mask with tensor ops. No pandas.
        """
        sig = self._gate_signal
        is_flat = (self._position == 0)                      # (B,) bool
        if sig is None:
            return self._allow_all_mask()                    # free / ungated phase

        abs_idx = self._abs_idx()
        if sig["kind"] == "named":
            gate_on = sig["gate_on"][abs_idx]                # (B,) bool — per episode
            return self._named_mask_from_gate(gate_on, is_flat)

        # string-condition path: per-bar buy/sell triggers
        buy_on = sig["buy_on"][abs_idx]
        sell_on = sig["sell_on"][abs_idx]
        return self._string_mask_from_triggers(buy_on, sell_on)

    def _named_mask_from_gate(self, gate_on: torch.Tensor, is_flat: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized equivalent of compute_action_mask's force_in_and_gate /
        open_gate branch (they are identical), evaluated for all B episodes at
        once from the per-episode gate_on + is_flat booleans:

            gate ON  + flat      -> {BUY, SELL}        must_enter=True
            gate ON  + in-trade  -> {FLAT, BUY, SELL}  must_enter=False
            gate OFF + flat      -> {FLAT}             must_enter=False
            gate OFF + in-trade  -> {FLAT, BUY, SELL}  must_enter=False
        """
        B, dev = self.B, self.device
        # Column-wise allow flags (B,) for FLAT / BUY / SELL.
        on, off = gate_on, ~gate_on
        flat, intrade = is_flat, ~is_flat
        allow_flat = (on & intrade) | (off & flat) | (off & intrade)   # not (on&flat)
        allow_buy = (on) | (off & intrade)
        allow_sell = allow_buy
        mask = torch.zeros(B, DIRECTION_DIM, dtype=torch.float32, device=dev)
        mask[:, FLAT] = allow_flat.float()
        mask[:, BUY] = allow_buy.float()
        mask[:, SELL] = allow_sell.float()
        must_enter = on & flat
        return mask, must_enter

    def _string_mask_from_triggers(self, buy_on: torch.Tensor, sell_on: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized equivalent of compute_action_mask's string-condition branch:

            buy only  -> {BUY}
            sell only -> {SELL}
            both      -> {BUY, SELL}
            neither   -> {FLAT, BUY, SELL}  (allow all)

        must_enter is always False on the string path (matches the scalar code).
        """
        B, dev = self.B, self.device
        both = buy_on & sell_on
        neither = (~buy_on) & (~sell_on)
        only_buy = buy_on & (~sell_on)
        only_sell = sell_on & (~buy_on)
        mask = torch.zeros(B, DIRECTION_DIM, dtype=torch.float32, device=dev)
        mask[:, FLAT] = neither.float()
        mask[:, BUY] = (only_buy | both | neither).float()
        mask[:, SELL] = (only_sell | both | neither).float()
        must_enter = torch.zeros(B, dtype=torch.bool, device=dev)
        return mask, must_enter

    def current_direction_mask(self) -> torch.Tensor:
        """(B, DIRECTION_DIM) per-episode float mask for the PPO direction head."""
        mask, _ = self.current_mask_and_force()
        return mask

    def _gate_on_batch(self, abs_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B,) bool — True where the phase gate condition fires for that episode
        at bar `abs_idx` (defaults to the current bar), regardless of position.

        Vectorized: a direct gather into the precomputed per-bar gate tensor. For
        the string-condition path "gate active" means either side triggered. For a
        free/ungated phase nothing fires (all False)."""
        if abs_idx is None:
            abs_idx = self._abs_idx()
        sig = self._gate_signal
        if sig is None:
            return torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if sig["kind"] == "named":
            return sig["gate_on"][abs_idx]
        return sig["buy_on"][abs_idx] | sig["sell_on"][abs_idx]

    # ── step ────────────────────────────────────────────────────────────────────
    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Advance one bar for all B episodes with a PPO structured action.

        actions: a dict from PPOAgent.select_actions with keys:
          direction (B,) long {FLAT,BUY,SELL}, lot_raw (B,) float in [0,1],
          exit (B,) long {HOLD,REDUCE,CLOSE}.
        Returns (next_state, reward (B,), done (B,) bool, info dict).
        """
        direction = actions["direction"].to(self.device).long()
        lot_raw = actions["lot_raw"].to(self.device).float()
        exit_a = actions.get("exit")
        exit_a = (exit_a.to(self.device).long() if exit_a is not None
                  else torch.zeros(self.B, dtype=torch.long, device=self.device))

        # ── S6 LOUD ACTION VALIDATION (reject, never silently zero-lot) ──────────
        # The old code mapped only {BUY,SELL} to a side and left ANY other direction
        # code as 0 (flat) — a silent zero-trade for a corrupted/out-of-range action.
        # The action contract is direction∈{FLAT,BUY,SELL}, exit∈{HOLD,REDUCE,CLOSE},
        # lot_raw∈[0,1] finite. A violation is a programming error upstream, so we
        # FAIL FAST with a clear message instead of masking it as a no-op day.
        if not bool(((direction >= 0) & (direction < DIRECTION_DIM)).all().item()):
            bad = direction[(direction < 0) | (direction >= DIRECTION_DIM)]
            raise ValueError(
                f"invalid direction code(s) {bad.tolist()} — must be in "
                f"[0,{DIRECTION_DIM}) {{FLAT,BUY,SELL}} (silent zero-lot rejected)")
        if not bool(((exit_a >= 0) & (exit_a < EXIT_DIM)).all().item()):
            bad = exit_a[(exit_a < 0) | (exit_a >= EXIT_DIM)]
            raise ValueError(
                f"invalid exit code(s) {bad.tolist()} — must be in "
                f"[0,{EXIT_DIM}) {{HOLD,REDUCE,CLOSE}}")
        if not bool(torch.isfinite(lot_raw).all().item()):
            raise ValueError("lot_raw contains non-finite values (NaN/Inf)")

        abs_idx = self._abs_idx()
        close = self.features[abs_idx, COL["close"]]
        nxt_idx = (abs_idx + 1).clamp(0, self.T - 1)
        next_close = self.features[nxt_idx, COL["close"]]

        # Per-episode gate state for THIS bar (drives force-entry + gate counting).
        gate_on = self._gate_on_batch(abs_idx)            # (B,) bool

        # Direction sign (+1 buy / -1 sell / 0 flat) and per-trade lot (continuous).
        dirs = torch.zeros(self.B, device=self.device)
        dirs = torch.where(direction == BUY, torch.ones_like(dirs), dirs)
        dirs = torch.where(direction == SELL, -torch.ones_like(dirs), dirs)
        # map raw [0,1] -> [lot_lo, lot_hi] (Section 8 CURRICULUM clamp window for
        # the active phase; the head still emits the full [0,1] — the clamp narrows
        # the EFFECTIVE size early in the curriculum and widens later). The agent
        # still learns WHERE inside the window to size (contextual sizing, S8.3).
        lots = self._map_lot_curriculum(lot_raw)
        lots = torch.where(dirs != 0, lots, torch.zeros_like(lots))

        # Once the day is halted (DD hit), no new positions open this day.
        halted = self._day_halted
        dirs = torch.where(halted, torch.zeros_like(dirs), dirs)
        lots = torch.where(halted, torch.zeros_like(lots), lots)

        # ── EXIT head wired into position management (action_space EXIT_*) ──
        # EXIT_CLOSE flattens the open position; EXIT_REDUCE halves it; EXIT_HOLD
        # leaves it. Applied to the EXISTING position before any new entry. PnL on
        # the closed/reduced fraction is realized at the current close.
        had_pos0 = self._position != 0
        price_move0 = (close - self._entry_px) * torch.sign(self._position)
        pnl_per_unit = price_move0 * self._position.abs() * 100000.0
        do_close = had_pos0 & (exit_a == EXIT_CLOSE)
        do_reduce = had_pos0 & (exit_a == EXIT_REDUCE)
        # realize PnL on the closed (100%) and reduced (50%) fractions
        exit_realized = (do_close.float() * pnl_per_unit
                         + do_reduce.float() * pnl_per_unit * 0.5)
        self._balance = self._balance + exit_realized
        # ── WIN / LOSS tally on this exit (close or reduce realizes PnL) ──
        exited = do_close | do_reduce
        self._wins_today = self._wins_today + (exited & (exit_realized > 0)).long()
        self._losses_today = self._losses_today + (exited & (exit_realized < 0)).long()
        # ── Section 5 commission at trade CLOSE (charged on the lots being closed) ──
        # The CLOSE side of the round-trip cost is deducted here, scaled by the lots
        # actually closed (full for EXIT_CLOSE, half for EXIT_REDUCE). Agent feels
        # the real per-trade cost (5.3). Forex EURUSD: $2.50/std-lot per side.
        closed_lots = (do_close.float() * self._position.abs()
                       + do_reduce.float() * self._position.abs() * 0.5)
        close_comm = self._commission_for_lots(closed_lots, close, side="close")
        self._balance = self._balance - close_comm
        # ── Section 7 speed bonus: confirm/revoke on CLOSE ──
        # A position that armed the speed bonus (showed green within the window)
        # KEEPS its pending bonus only if it CLOSES in profit (price_move0>0); a
        # losing close REVOKES it. We fold the realized speed bonus into a per-step
        # accumulator applied to reward below.
        closed_now = do_close & had_pos0
        profitable_close = closed_now & (price_move0 > 0)
        speed_realized = torch.where(profitable_close & self._speed_armed,
                                     self._speed_pending,
                                     torch.zeros_like(self._speed_pending))
        # clear pending/armed for any fully-closed position (win or loss).
        self._speed_pending = torch.where(closed_now, torch.zeros_like(self._speed_pending),
                                          self._speed_pending)
        self._speed_armed = torch.where(closed_now, torch.zeros_like(self._speed_armed),
                                        self._speed_armed)
        # shrink/close the position per the exit action
        self._position = torch.where(do_close, torch.zeros_like(self._position),
                                     self._position)
        self._position = torch.where(do_reduce, self._position * 0.5, self._position)

        # Realize PnL on existing position when direction flips (new entry).
        had_pos = self._position != 0
        price_move = (close - self._entry_px) * torch.sign(self._position)
        realized = torch.where(had_pos, price_move * self._position.abs() * 100000.0,
                               torch.zeros_like(close))
        # open new position where a non-hold action is taken
        opening = dirs != 0
        self._trades_today = self._trades_today + opening.long()
        # ── WIN / LOSS tally when a flip closes the OLD position (realizes PnL) ──
        flip_closed = opening & had_pos
        self._wins_today = self._wins_today + (flip_closed & (realized > 0)).long()
        self._losses_today = self._losses_today + (flip_closed & (realized < 0)).long()
        self._balance = self._balance + torch.where(opening, realized,
                                                  torch.zeros_like(realized))
        self._position = torch.where(opening, dirs * lots, self._position)

        # ── FORCE-ENTRY ENFORCEMENT (gate invariant, DESIGN_DECISIONS.md #2) ──
        # A trade MUST be active on EVERY bar the gate is ON. The PPO direction
        # mask enforces this when the episode is flat at the START of the bar, but
        # the agent can still close an existing position (EXIT_CLOSE) and sample
        # FLAT in the SAME bar — going flat mid-gate. The rule is "re-enter that
        # same bar", so here we re-open any episode that is gate-ON, not halted,
        # and now flat. The code never PICKS the side: we reuse the agent's own
        # sampled direction, defaulting to BUY only when it sampled FLAT. The forced
        # lot uses the SAME Section-8 curriculum clamp as a normal entry.
        need_entry = gate_on & (~self._day_halted) & (self._position == 0)
        if bool(need_entry.any().item()):
            forced_dir = torch.where(direction == SELL,
                                     -torch.ones(self.B, device=self.device),
                                     torch.ones(self.B, device=self.device))
            forced_lot = self._map_lot_curriculum(lot_raw)
            self._trades_today = self._trades_today + need_entry.long()
            self._position = torch.where(need_entry, forced_dir * forced_lot,
                                         self._position)
            opening = opening | need_entry
        # Entry fills at the bar CLOSE, optionally worsened by entry friction
        # (half-spread + slippage in the trade direction). BUY pays UP (+), SELL is
        # filled DOWN (-); friction is 0 when ENTRY_FRICTION_ENABLED is off, so the
        # default path is the unchanged `close` fill. _position sign carries the
        # side (already set above for both agent entries and force-entries).
        entry_fill = close
        if self._entry_friction_enabled and self._friction_px != 0.0:
            entry_fill = close + torch.sign(self._position) * self._friction_px
        self._entry_px = torch.where(opening, entry_fill, self._entry_px)

        # ── Section 5 commission at trade OPEN + Section 7 entry-bar stamp ──
        # The OPEN side of the round-trip cost is deducted on any bar a NEW position
        # was opened (agent or force-entry), scaled by the opened lots. We also
        # stamp the entry bar and reset the speed-bonus arm flag for the new trade.
        opened_lots = torch.where(opening, self._position.abs(),
                                  torch.zeros_like(self._position))
        open_comm = self._commission_for_lots(opened_lots, close, side="open")
        self._balance = self._balance - open_comm
        self._entry_bar = torch.where(opening, self._step_i, self._entry_bar)
        self._speed_armed = torch.where(opening, torch.zeros_like(self._speed_armed),
                                        self._speed_armed)
        self._speed_pending = torch.where(opening, torch.zeros_like(self._speed_pending),
                                          self._speed_pending)

        # mark-to-market with next close
        # PnL = price_move (raw price units) * lots * contract_size (100000).
        # A 10-pip move (0.0010) on 0.10 lots = 0.0010 * 0.10 * 100000 = $10.00.
        # Leverage 1:100 affects margin requirement only, not PnL per lot.
        mtm = (next_close - self._entry_px) * torch.sign(self._position) * \
            self._position.abs() * 100_000.0
        # MARKED equity = REALIZED balance + open-position unrealized PnL, recomputed
        # fresh each bar from self._balance (which holds NO unrealized component). The
        # earlier code added mtm to self._equity and then wrote equity_now back into
        # self._equity, so the next bar's self._equity already contained this bar's
        # unrealized PnL and re-added it — compounding a held winner's equity every
        # bar (P0 double-count). Sourcing mtm from the realized balance fixes it.
        equity_now = self._balance + torch.where(self._position != 0, mtm,
                                                torch.zeros_like(mtm))
        self._day_high_eq = torch.maximum(self._day_high_eq, equity_now)
        # intra-day equity HIGH (Section 3.2 give-back-from-high) + multi-day PEAK
        # (Section 3.3 cross-day give-back). Both track on the mark-to-market equity.
        self._intraday_high_eq = torch.maximum(self._intraday_high_eq, equity_now)
        self._multi_day_peak = torch.maximum(self._multi_day_peak, equity_now)

        # ── Section 7 speed bonus: ARM within the window on unrealized profit ──
        # While a position is open and within SPEED_BONUS_MINUTES bars of entry, if
        # the trade shows unrealized profit NET of this symbol's (round-trip)
        # commission, accrue the pending speed bonus and mark it armed. The bonus is
        # only KEPT if the trade later closes in profit (confirmed on close above).
        if self._speed_bonus > 0.0:
            in_window = ((self._step_i - self._entry_bar) <= self._speed_window) \
                & (self._position != 0)
            rt_comm = self._commission_per_lot_round_trip(next_close) \
                * self._position.abs()
            unrealized_net = mtm - rt_comm
            arm_now = in_window & (unrealized_net > 0) & (~self._speed_armed)
            self._speed_armed = self._speed_armed | arm_now
            self._speed_pending = torch.where(
                arm_now, torch.full_like(self._speed_pending, self._speed_bonus),
                self._speed_pending)

        # ── vectorized DD + day boundary (no per-batch python loop) ──
        dd_used = (self._day_high_eq - equity_now) / (self._day_high_eq + 1e-8)
        breach_now = dd_used >= self._max_dd_pct_t
        self._dd_breached = self._dd_breached | breach_now

        # ── count gate-active bars this step, PER EPISODE — BEFORE the day reset ──
        # ROOT-CAUSE FIX (ftmo_rules_fix.md RULE 4): this MUST happen before the
        # new_day reset of _gate_bars_today below. Previously the counter was
        # incremented AFTER the reset, so on every calendar-day-close step the
        # reported/classified count was 0-or-1 — never the day's true total. That
        # made the old "gate_was_active" test always read False on day boundaries,
        # which is what funneled whole zero-trade days into the (now removed) SKIP
        # bucket instead of FAIL. gate_on is position-independent, so it is the
        # correct gate-active signal whether or not the episode holds a trade.
        self._gate_bars_today = self._gate_bars_today + gate_on.long()

        self._step_i = self._step_i + 1
        new_day = (self._step_i % self.bars_per_day) == 0
        # Save trades_today BEFORE resetting — info must report the closing day's count
        trades_today_closing = self._trades_today.clone()
        wins_today_closing = self._wins_today.clone()
        losses_today_closing = self._losses_today.clone()
        # Snapshot the CLOSING day's gate-bar count BEFORE the new_day reset so
        # classification/reporting see the whole day's total (see root-cause note).
        gate_bars_closing = self._gate_bars_today.clone()
        # Snapshot the CLOSING day's baseline/peak BEFORE rolling them forward.
        # The reward, PASS/FAIL classification, and the honest day report must all
        # measure the day that just ended against ITS OWN start equity — not the
        # post-reset baseline (which would make daily_ret ≈ 0 on every calendar
        # boundary and silently prevent PASS from ever firing there).
        day_start_closing = self._day_start_eq.clone()
        day_high_closing = self._day_high_eq.clone()
        # Snapshot the CLOSING day's intra-day equity HIGH + accrued progress reward
        # for the Section-3.2 give-back/wipeout (computed below), then reset them on
        # the new day so each day's give-back is measured within its own bounds.
        intraday_high_closing = self._intraday_high_eq.clone()
        progress_reward_closing = self._day_progress_reward.clone()
        # On a new day: roll baseline forward (vectorized)
        self._day_start_eq = torch.where(new_day, equity_now, self._day_start_eq)
        self._day_high_eq = torch.where(new_day, equity_now, self._day_high_eq)
        self._intraday_high_eq = torch.where(new_day, equity_now, self._intraday_high_eq)
        self._trades_today = torch.where(new_day,
                                         torch.zeros_like(self._trades_today),
                                         self._trades_today)
        self._wins_today = torch.where(new_day, torch.zeros_like(self._wins_today),
                                       self._wins_today)
        self._losses_today = torch.where(new_day, torch.zeros_like(self._losses_today),
                                         self._losses_today)
        # new day clears the intraday DD halt (fresh trading day, FTMO CEST)
        self._day_halted = torch.where(new_day, torch.zeros_like(self._day_halted),
                                       self._day_halted)
        self._dd_breached = torch.where(new_day, torch.zeros_like(self._dd_breached),
                                        self._dd_breached)
        self._gate_bars_today = torch.where(new_day,
                                            torch.zeros_like(self._gate_bars_today),
                                            self._gate_bars_today)
        # Reset the per-day accrued intra-day progress reward on the new day (S3.2).
        self._day_progress_reward = torch.where(new_day,
                                                torch.zeros_like(self._day_progress_reward),
                                                self._day_progress_reward)
        self._day_progress_prev = torch.where(new_day,
                                              torch.zeros_like(self._day_progress_prev),
                                              self._day_progress_prev)

        self._equity = equity_now

        # ── INTRADAY 1% DD ENDS THE TRADING DAY (DESIGN_DECISIONS.md #5) ──
        # When today's trailing DD first hits the limit, halt trading for the
        # rest of the day (positions flattened, no new entries until next day).
        #
        # ROOT-CAUSE FIX (every-other-day zero-trade bug): breach_now (computed at
        # the top of step) is measured against the CLOSING day's peak/equity. On a
        # calendar boundary the block above ALREADY rolled the day forward and
        # cleared _day_halted to False for the FRESH day. If we then OR breach_now
        # back into _day_halted unconditionally, a breach on the final bar of the
        # closing day (or simply a still-below-peak last bar) RE-HALTS the brand-new
        # day before it has traded a single bar — so the next day opens permanently
        # halted, produces ZERO trades, then clears at ITS boundary, and the cycle
        # repeats. That is the perfect odd/even alternating zero-trade signature.
        #
        # The closing day's breach is already captured in _dd_breached and the
        # classification snapshots taken BEFORE the reset, so the halt itself must
        # only apply to the day the breach actually belongs to — i.e. NOT on a
        # new_day boundary. We gate the halt with (~new_day): on a calendar rollover
        # the new day starts fresh (halt stays False); intraday breaches halt as
        # before. The position flatten is likewise scoped so a freshly-reset day is
        # never flattened on its opening bar by the previous day's breach.
        halt_breach = breach_now & (~new_day)
        newly_halted = halt_breach & (~self._day_halted)
        self._day_halted = self._day_halted | halt_breach
        # When the DD-halt flattens an OPEN position, its unrealized mark-to-market
        # must be REALIZED into the balance (a forced close), not silently dropped.
        # mtm is this bar's open-position PnL (0 when flat); fold it into _balance for
        # the episodes flattened THIS bar, then zero the position. Without this the
        # breach loss would vanish from the balance and the day would read flat.
        flatten = self._day_halted & (self._position != 0)
        self._balance = self._balance + torch.where(flatten, mtm,
                                                    torch.zeros_like(mtm))
        self._position = torch.where(self._day_halted, torch.zeros_like(self._position),
                                     self._position)

        # ════════════════════════════════════════════════════════════════════
        # REWARD — fully NORMALIZED (percent-of-start-equity) units so it is
        # account-size invariant and O(1) per step (NOT raw dollars, which the
        # PnL fix made ~10,000x larger and would blow up PPO). See learning_loop
        # _fix.md FIX 1. Every term below is a fraction of start-of-day equity,
        # tuned so a typical day's cumulative reward is roughly O(1).
        #
        #   daily_ret    = (equity - day_start_eq) / day_start_eq      [percent]
        #   target_pct   = +2.5% daily goal       max_dd_pct = 1% trailing DD
        #
        # Dense per-step shaping (points the gradient toward the FTMO objective
        # before the sparse terminal pass/fail ever fires):
        #   (1) step PnL    : Δequity / day_start_eq   — direct progress signal.
        #   (2) target pull : reward closing the gap to +2.5% (only while below
        #                     target and positive), scaled small so it nudges,
        #                     doesn't dominate.
        #   (3) DD proximity: penalty that grows as intraday DD approaches the 1%
        #                     limit (risk awareness) — quadratic in dd_used/limit.
        #   (4) overtrade   : tiny per-new-trade cost so it won't churn ~100
        #                     trades/day of noise, but small enough it still trades.
        # The terminal pass/fail/streak day bonus (below) is kept but is ALSO in
        # these normalized units now (see _day_reward_norm), so all terms share
        # one scale.
        # ════════════════════════════════════════════════════════════════════
        rw = self.cfg.get("REWARD", {}) or {}
        # Measure the day that JUST traded against its OWN baseline (the pre-reset
        # snapshot). On a calendar boundary the live _day_start_eq has already
        # rolled forward to equity_now, so using it here would zero out the day's
        # return; day_start_closing preserves it.
        daily_ret = (self._equity - day_start_closing) / (day_start_closing + 1e-8)
        reward = torch.zeros_like(daily_ret)

        # (1) step PnL in percent-of-day-start (NOT dollars).
        step_pnl_pct = (equity_now - self._equity_prev) / (day_start_closing + 1e-8)
        reward = reward + float(rw.get("step_pnl_scale", 1.0)) * step_pnl_pct
        self._equity_prev = equity_now.clone()

        # (2) target-progress pull: while below target and in profit, reward the
        #     fraction of the +2.5% goal achieved this bar (delta of clipped ratio).
        prog = (daily_ret / (self._target_pct_t + 1e-8)).clamp(min=0.0, max=1.0)
        reward = reward + float(rw.get("target_progress_scale", 0.0)) * step_pnl_pct \
            * (prog > 0).float()

        # (2b) Section 3.1 INTRA-DAY PROGRESSIVE reward: a smooth LINEAR pull as
        #      equity climbs toward today's target. We reward the per-bar INCREMENT
        #      of clamped progress (so it accrues once as the day advances toward the
        #      target, not every bar held). The accrued total is tracked so a FAIL
        #      day can WIPE it (S3.2 below). intraday_progress_scale weights it.
        progress_inc = (prog - self._day_progress_prev).clamp(min=0.0)
        intraday_progress_r = float(rw.get("intraday_progress_scale", 0.5)) * progress_inc
        reward = reward + intraday_progress_r
        self._day_progress_reward = self._day_progress_reward + intraday_progress_r
        self._day_progress_prev = prog

        # (3) drawdown-proximity penalty: 0 when flat-to-peak, grows quadratically
        #     as intraday DD eats into the 1% headroom; hard breach handled below.
        dd_used_now = (day_high_closing - equity_now) / (day_high_closing + 1e-8)
        dd_frac = (dd_used_now / (self._max_dd_pct_t + 1e-8)).clamp(min=0.0, max=2.0)
        reward = reward - float(rw.get("dd_proximity_scale", 0.02)) * dd_frac.pow(2)

        # (4) overtrade penalty: small cost per NEW position opened this bar.
        reward = reward - float(rw.get("overtrade_penalty", 0.0005)) * opening.float()

        # (5) Section 7 SPEED BONUS realized on a profitable close this bar (the
        #     pending bonus confirmed when the armed trade closed green).
        reward = reward + speed_realized

        # ════════════════════════════════════════════════════════════════════
        # DAY CLASSIFICATION — FIVE TIERS (dd_classification_refine.md; vectorized
        # mirror of core/reward/shaper.classify_day). Thresholds are measured
        # against INITIAL equity (FIXED-$ increments), with a NEW capital-loss guard
        # checked FIRST. final = equity_now (the closing OR DD-halt balance);
        # prior_day_balance == day_start_closing (yesterday's close == today's open).
        # Precedence (FIRST match wins — identical whether or not a DD breach
        # occurred; a breach merely ends the day early, it is NOT an auto-fail):
        #   1. final < prior_day_balance            -> FAIL_CAPITAL_LOSS  [NEW]
        #   2. elif final >= initial*(1+target_pct) -> PASS  (>= +2.5% of INITIAL)
        #   3. elif final >= initial*(1+half_pct)   -> OK    (>= +1.25%, >=50% tgt)
        #   4. else                                 -> FAIL  (< half, not < prior)
        # THEN: EXCEED = PASS AND strictly above the full target AND never breached;
        # SURVIVAL bonus only on a traded day that NEVER breached. A breached day,
        # even one whose halt balance is PASS/OK, can earn NEITHER EXCEED NOR
        # SURVIVAL (RESOLVED DECISION 2). A zero-trade day ends flat (final == start
        # == prior) -> not < prior, < half -> FAIL_UNDER_TARGET. The reward MAGNITUDE
        # math below still keys off `progress` (day-start-relative) for severity /
        # OK partial-credit / EXCEED-progressive — only the TIER DECISION moved to
        # absolute INITIAL-relative thresholds. Binary `passed`/`failed` are still
        # exported (passed == PASS|EXCEED; failed == every FAIL_*/OK) so the FTMO
        # guard, eval loop, and aggregation keep working unchanged. The day closes on
        # a calendar boundary (new_day) OR the moment a DD breach halts trading
        # (newly_halted).
        # ════════════════════════════════════════════════════════════════════
        day_closed = new_day | newly_halted
        traded_today = trades_today_closing > 0          # day actually traded
        # RULE 1: fixed-increment target measured off the CLOSING day's opening eq
        # (kept for the obs/info `daily_target`); the TIER decision uses absolute
        # INITIAL-relative thresholds below.
        daily_target = day_start_closing + self._daily_increment_t
        inc = self._daily_increment_t.clamp(min=1e-9)
        progress = (equity_now - day_start_closing) / inc
        breached = self._dd_breached                      # breached this day (any bar)

        # ── ABSOLUTE INITIAL-RELATIVE THRESHOLDS (dd_classification_refine.md) ──
        # Fixed-$ off INITIAL equity: full target == initial*(1+target_pct),
        # half == initial*(1+half_target_pct). prior_day == day_start_closing.
        full_target_eq = self._initial_equity_t * (1.0 + self._target_pct_t)
        half_target_eq = self._initial_equity_t * (1.0 + self._half_target_pct_t)
        prior_day_balance = day_start_closing

        # ── TIER MASKS (precedence: capital-loss FIRST, then PASS, then OK) ──
        is_capital_loss = equity_now < prior_day_balance               # 1. NEW guard
        at_or_above_full = equity_now >= full_target_eq
        at_or_above_half = equity_now >= half_target_eq
        is_pass = (~is_capital_loss) & at_or_above_full                 # 2. PASS
        # EXCEED: PASS, strictly above the full target, and DD never breached.
        is_exceed = is_pass & (equity_now > full_target_eq) & (~breached)
        is_ok = (~is_capital_loss) & (~is_pass) & at_or_above_half      # 3. OK
        # 4. FAIL_UNDER_TARGET: not capital-loss, below half target.
        is_fail_under = (~is_capital_loss) & (~is_pass) & (~is_ok)
        # `is_fail` keeps its old meaning for the reward MAGNITUDE path (any FAIL_*).
        is_fail = is_capital_loss | is_fail_under
        # binary compatibility: PASS or EXCEED counts as a "passed" day; OK and all
        # FAIL_* are non-passing (OK does NOT advance the pass-streak — RULE 3).
        passed = is_pass
        failed = ~passed
        self._day_passed = torch.where(day_closed & passed,
                                       torch.ones_like(self._day_passed), self._day_passed)

        # ── TIER REWARD (Section 1) — vectorized day_reward() ──
        pass_b = float(rw.get("pass_day_bonus", 2.0))
        fail_b = float(rw.get("fail_day_penalty", -2.0))
        ok_lo = float(rw.get("ok_partial_lo", 0.25))
        ok_hi = float(rw.get("ok_partial_hi", 0.95))
        exceed_scale = float(rw.get("exceed_scale", 1.0))
        survival_b = float(rw.get("survival_bonus", 1.5))
        red_scale = float(rw.get("red_day_scale", 1.0))
        dd_eff_w = float(rw.get("dd_efficiency_weight", 0.5))

        tier_reward = torch.zeros_like(progress)
        # FAIL: base penalty scaled LINEARLY by how negative (severity 1x at 0%
        # progress, growing as progress goes negative). max(1, 1-min(progress,0)).
        severity = (1.0 - progress.clamp(max=0.0)).clamp(min=1.0)
        tier_reward = torch.where(is_fail, fail_b * severity, tier_reward)
        # OK: linear partial credit ok_lo..ok_hi of pass_b as progress climbs 0.5->1.
        ok_frac = ok_lo + (ok_hi - ok_lo) * ((progress - 0.5) / 0.5)
        tier_reward = torch.where(is_ok, pass_b * ok_frac, tier_reward)
        # PASS / EXCEED: full pass_b, plus progressive (progress-1) for EXCEED.
        tier_reward = torch.where(is_pass, torch.full_like(progress, pass_b), tier_reward)
        tier_reward = torch.where(is_exceed,
                                  pass_b + exceed_scale * (progress - 1.0).clamp(min=0.0),
                                  tier_reward)

        # ── Section 1.3 DD-EFFICIENCY multiplier (positive days only) ──
        dd_today = (day_high_closing - equity_now) / (day_high_closing + 1e-8)
        used_frac = (dd_today / (self._max_dd_pct_t + 1e-8)).clamp(min=0.0, max=1.0)
        dd_eff_mult = 1.0 - dd_eff_w * used_frac
        positive_day = (is_ok | is_pass) & (tier_reward > 0)
        tier_reward = torch.where(positive_day, tier_reward * dd_eff_mult, tier_reward)

        # ── Section 1.2 RED-DAY linear penalty (on top of FAIL) ──
        loss_frac = ((day_start_closing - equity_now) / inc).clamp(min=0.0)
        red_pen = torch.where(equity_now < day_start_closing,
                              -red_scale * loss_frac, torch.zeros_like(progress))
        tier_reward = tier_reward + red_pen

        # ── Section 1.1 SURVIVAL bonus (stacked; traded AND never breached) ──
        survived = traded_today & (~breached)
        tier_reward = tier_reward + survived.float() * survival_b

        # ════════════════════════════════════════════════════════════════════
        # SECTION 2 — STREAK STATE MACHINE (vectorized StreakTracker mirror).
        # On each CLOSED day: momentum bias (from prior pass), positive/negative
        # exponential streak curve, mulligan (2 consecutive fails break the streak),
        # escalating consecutive-fail penalty, recovery bonus. Only the closed
        # episodes update; the rest carry state unchanged.
        # ════════════════════════════════════════════════════════════════════
        streak_base = float(rw.get("streak_base", 0.5))
        neg_mult = float(rw.get("negative_streak_mult", 1.5))
        escalation = float(rw.get("consec_fail_escalation", 0.5))
        recovery_b = float(rw.get("recovery_bonus", 3.0))
        momentum_b = float(rw.get("momentum_bonus", 0.2))
        a, b = self._streak_a, self._streak_b

        # Momentum: small positive bias on a closed day that FOLLOWS a pass.
        streak_reward = torch.where(day_closed & self._last_was_pass,
                                    torch.full_like(progress, momentum_b),
                                    torch.zeros_like(progress))

        # --- PASS branch (closed & passed) ---
        cp = day_closed & passed
        recovering = cp & (self._fail_streak > 0)
        new_pass_streak = torch.where(cp, self._pass_streak + 1, self._pass_streak)
        # positive streak reward = base + a*(exp(b*(s-1))-1)
        pos_extra = a * (torch.exp(b * (new_pass_streak.float() - 1.0)) - 1.0)
        streak_reward = streak_reward + torch.where(
            cp, streak_base + pos_extra, torch.zeros_like(progress))
        streak_reward = streak_reward + recovering.float() * recovery_b

        # --- FAIL branch (closed & failed) ---
        cf = day_closed & failed
        new_consec_fail = torch.where(cf, self._consec_fail + 1, self._consec_fail)
        new_fail_streak = torch.where(cf, self._fail_streak + 1, self._fail_streak)
        # negative streak mirrors the positive curve at neg_mult magnitude.
        neg_mag = streak_base + a * (torch.exp(b * (new_fail_streak.float() - 1.0)) - 1.0)
        streak_reward = streak_reward + torch.where(
            cf, -neg_mult * neg_mag, torch.zeros_like(progress))
        # escalating consecutive-fail penalty (in addition to the negative streak).
        streak_reward = streak_reward - cf.float() * escalation * new_consec_fail.float()

        # Commit the streak-machine state for closed episodes (mulligan: the pass
        # streak survives a SINGLE fail; a SECOND consecutive fail breaks it).
        broke = cf & (new_consec_fail > int(rw.get("mulligan_count", 1)))
        self._pass_streak = torch.where(cp, new_pass_streak,
                                        torch.where(broke,
                                                    torch.zeros_like(self._pass_streak),
                                                    self._pass_streak))
        self._fail_streak = torch.where(cp, torch.zeros_like(self._fail_streak),
                                        torch.where(cf, new_fail_streak,
                                                    self._fail_streak))
        self._consec_fail = torch.where(cp, torch.zeros_like(self._consec_fail),
                                        torch.where(cf, new_consec_fail,
                                                    self._consec_fail))
        self._last_was_pass = torch.where(cp, torch.ones_like(self._last_was_pass),
                                          torch.where(cf,
                                                      torch.zeros_like(self._last_was_pass),
                                                      self._last_was_pass))
        # signed streak for the observation (S6.6): +pass run / -fail run.
        self._signed_streak = torch.where(self._pass_streak > 0, self._pass_streak,
                                          -self._fail_streak)
        # best pass-streak achieved THIS episode (S4.2 composite episode bonus).
        self._best_streak = torch.maximum(self._best_streak, self._pass_streak)

        # ════════════════════════════════════════════════════════════════════
        # SECTION 3.2 / 3.3 — GIVE-BACK PENALTIES (applied on the closed day).
        #  • intra-day WIPEOUT on FAIL: erase this day's accrued progress reward and
        #    penalize the give-back from the intra-day HIGH to the close.
        #  • cross-day give-back: penalize any drop from the multi-day PEAK.
        # ════════════════════════════════════════════════════════════════════
        giveback = torch.zeros_like(progress)
        if bool(rw.get("intraday_wipeout", True)):
            from_high = ((intraday_high_closing - equity_now)
                         / (day_start_closing + 1e-8)).clamp(min=0.0)
            wipe = is_fail & day_closed
            giveback = giveback + wipe.float() * (
                -progress_reward_closing
                - float(rw.get("giveback_from_high_scale", 1.0)) * from_high)
        cross_drop = ((self._multi_day_peak - equity_now)
                      / (self._initial_equity_t + 1e-8)).clamp(min=0.0)
        giveback = giveback - day_closed.float() \
            * float(rw.get("cross_day_giveback_scale", 0.5)) * cross_drop

        # ── Compose the terminal day reward (only on closed days) ──
        day_reward = tier_reward + streak_reward + giveback
        reward = reward + day_closed.float() * day_reward

        # ════════════════════════════════════════════════════════════════════
        # REWARD SAFETY NET (self-healing) — sanitize + clamp the final reward.
        # Every term above is designed to be O(1) percent-of-equity, but if the
        # net ever emits a bad action / equity goes non-finite (e.g. after a
        # transient numerical hiccup), an unbounded reward would feed straight
        # into PPO and blow up the value loss. We:
        #   (1) replace any NaN/Inf reward with 0.0 (a neutral step), and
        #   (2) hard-clamp the per-step reward into [-REWARD_CLIP, +REWARD_CLIP].
        # REWARD_CLIP defaults to 10.0 (a whole well-played day is only a few
        # units, so this never touches legitimate rewards) and is config-driven.
        # This makes a single bad bar a no-op instead of a training-killer.
        # ════════════════════════════════════════════════════════════════════
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward_clip = float(self.cfg.get("REWARD_CLIP", 10.0))
        reward = reward.clamp(min=-reward_clip, max=reward_clip)

        # Advance the GLOBAL day index on the calendar new-day boundary only
        # (every bars_per_day bars, synchronized across all episodes). A DD-halt
        # closes an episode's day early for classification but does NOT advance
        # the calendar index — so the trainer still aggregates by aligned day.
        self._day_idx = self._day_idx + new_day.long()

        # episode termination: reached ep_bars or trade cap
        max_trades = int(self.cfg.get("MAX_TRADES_PER_DAY", 800))
        cap_hit = self._trades_today >= max_trades
        ep_end = (self._step_i >= self.ep_bars) | cap_hit
        self._done = self._done | ep_end

        pass_no_breach = passed & (~self._dd_breached)
        # SURVIVAL is exported as its own flag: a breached day can never earn it
        # (RESOLVED DECISION 2). It is the `survived` mask scoped to closed days.
        survival = (survived & day_closed)
        info = {
            "equity": self._equity.detach(),
            "passed": passed.detach(),       # PASS or EXCEED (binary-compatible)
            "failed": failed.detach(),       # complement of passed
            # ── 5-tier flags (Section 1 + dd_classification_refine). Exactly one of
            # fail_under/capital_loss/ok/pass is true on a closed day; tier_fail is
            # the union of both FAIL_* tiers (binary-compat); exceed is a subset of
            # pass; survival stacks (never on a breached day). All scoped to closed
            # days so non-closing bars read 0.
            "tier_fail":   (is_fail & day_closed).detach(),          # any FAIL_*
            "tier_fail_under":   (is_fail_under & day_closed).detach(),
            "tier_capital_loss": (is_capital_loss & day_closed).detach(),
            "tier_ok":     (is_ok & day_closed).detach(),
            "tier_pass":   (is_pass & day_closed).detach(),
            "tier_exceed": (is_exceed & day_closed).detach(),
            "survival":    survival.detach(),
            "dd_breached": self._dd_breached.detach(),
            "trades_today": trades_today_closing.detach(),
            # WON / LOST trades for the CLOSING day (realized-PnL count, not days).
            "wins_today": wins_today_closing.detach(),
            "losses_today": losses_today_closing.detach(),
            "pass_no_breach": pass_no_breach.detach(),
            "day_closed": day_closed.detach(),
            "day_halted": self._day_halted.detach(),
            "pass_streak": self._pass_streak.detach(),
            "fail_streak": self._fail_streak.detach(),
            "signed_streak": self._signed_streak.detach(),
            "best_streak": self._best_streak.detach(),
            # CLOSING day's true gate-bar total (pre-reset snapshot) — used by the
            # never-flat-through-gate test and diagnostics, NOT for classification
            # (a zero-trade day is a FAIL regardless of gate activity, RULE 2).
            "gate_bars_today": gate_bars_closing.detach(),
            "executed_direction": direction.detach(),
            # per-day snapshots for HONEST aggregation (Bug A): the closing day's
            # return and trailing DD as fractions, plus the global day index so
            # the trainer can group all episodes that closed the SAME calendar day.
            "daily_return": daily_ret.detach(),
            "daily_dd": dd_today.detach(),
            "daily_target": daily_target.detach(),    # fixed-increment target (RULE 1)
            "day_idx": self._day_idx.detach(),
            # report the CLOSING day's baseline so the trainer's day PnL
            # (equity - day_start_eq) reflects the day that just ended, not the
            # post-reset baseline (which would always read ~0).
            "day_start_eq": day_start_closing.detach(),
        }
        return self._get_state(), reward, self._done.clone(), info

    def get_status_dict(self) -> dict:
        """Scalar snapshot of batch item 0 (for dashboards/tests)."""
        return {
            "equity": float(self._equity[0].item()),
            "day_start_eq": float(self._day_start_eq[0].item()),
            "dd_breached": bool(self._dd_breached[0].item()),
            "trades_today": int(self._trades_today[0].item()),
            "step": int(self._step_i[0].item()),
        }


# ── multi-symbol loading (STEP 12) ───────────────────────────────────────────
def build_multi_symbol_env(features_by_symbol: Dict[str, "np.ndarray"], cfg: dict,
                           device: torch.device, phase: Optional[dict] = None,
                           policy: Optional[dict] = None):
    """
    Build one BatchedFTMOEnv per instrument, aligned by row count (inner-join on
    the shorter series). Returns dict[instrument -> BatchedFTMOEnv]. The trainer
    averages reward across instruments for a basket phase.
    """
    if not features_by_symbol:
        raise ValueError("features_by_symbol is empty")
    min_len = min(len(v) for v in features_by_symbol.values())
    envs = {}
    for sym, feats in features_by_symbol.items():
        envs[sym] = BatchedFTMOEnv(feats[:min_len], cfg, device, instrument=sym,
                                   phase=phase, policy=policy)
    return envs
