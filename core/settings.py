"""
core/settings.py
────────────────────────────────────────────────────────────────────────────
Central CFG dict. Agent is PURE PPO (DQN deprecated -> legacy/; see
DESIGN_DECISIONS.md #1). A100 tuning scales the parallel env batch and the
on-policy rollout length:

    BATCH_SIZE_ENV : parallel episodes on GPU (4 CPU / 32 T4 / 64 A100)
    ROLLOUT_STEPS  : on-policy steps per PPO update (tuned per device)

Paths are NOT hardcoded here for Drive — train.py / live_runner receive them as
CLI args or read them from config YAML (HARD RULE 11). The defaults below are
local/dev fallbacks only.

Device is auto-detected: CUDA (A100/T4) when available, else CPU. The same code
runs identically in Colab (GPU) and in CI/dev (CPU) — only speed differs.
"""
from __future__ import annotations

import torch



def get_device() -> torch.device:
    """Return CUDA device if available (Colab A100/T4), else CPU (dev/CI)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── GPU UTILIZATION TIERS (Section 10) ───────────────────────────────────────
# auto_tune_batch picks (BATCH_SIZE_ENV, ROLLOUT_STEPS) per the VRAM of whatever
# GPU is attached so we hit a HIGH utilization target (>=80% of the card) WITHOUT
# manual tuning, and never pay for idle silicon on a smaller/cheaper card. Each
# tier is a (min_vram_gb, batch, rollout) row; the first row whose min_vram_gb the
# card meets (scanning largest-first) wins. Adding a new card class is a one-line
# edit here — nothing downstream hardcodes 64/4096. The values are sized so the
# PPO rollout (time*env samples per update) keeps the SMs busy on each tier:
#   A100/H100 (>30GB)  -> 64 env  x 4096 steps  (big on-policy batch)
#   L4/A10/3090 (>20)  -> 48 env  x 3072
#   T4/V100    (>12)   -> 32 env  x 2048
#   small GPU  (<=12)  -> 16 env  x 1024  (cost-effective; never idle)
# CPU (dev/CI)         -> 4  env  x 64    (fast smoke tests)
GPU_TIERS = [
    # (min_vram_gb, BATCH_SIZE_ENV, ROLLOUT_STEPS, label)
    (30.0, 64, 4096, "A100/H100-class"),
    (20.0, 48, 3072, "L4/A10/3090-class"),
    (12.0, 32, 2048, "T4/V100-class"),
    (0.0,  16, 1024, "small-GPU (cost-effective)"),
]
# Target fraction of the attached GPU we aim to keep busy (Section 10.1). The
# profiling cell / auto_tune_batch print this so a human can confirm >=80%.
GPU_UTIL_TARGET = 0.80


def auto_tune_batch(cfg: dict, device: torch.device) -> dict:
    """
    Auto-scale batch sizes to the detected GPU so we hit the >=80% utilization
    target on ANY card (Section 10), with a cost-effective fallback for smaller
    GPUs (never pay for idle hardware). EXTENDS the original A100/T4/CPU policy
    by scanning the GPU_TIERS table above instead of a hardcoded if/else, so a new
    card class is a single table row. The PPO update is full-batch over time*env,
    so the (batch, rollout) pair keeps minibatch/epoch math coherent (n_epochs is
    independent of rollout length).

    Respects an explicitly caller-pinned BATCH_SIZE_ENV / ROLLOUT_STEPS only when
    cfg["AUTO_TUNE_GPU"] is False (default True) — otherwise the tier wins so the
    GPU is never left under-utilized by a stale value carried in from a checkpoint
    or a copy-pasted cfg.
    """
    cfg = dict(cfg)
    if not bool(cfg.get("AUTO_TUNE_GPU", True)):
        return cfg                       # honor caller-pinned sizes verbatim
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        for min_vram, batch, rollout, label in GPU_TIERS:
            if vram_gb >= min_vram:
                cfg["BATCH_SIZE_ENV"] = batch
                cfg["ROLLOUT_STEPS"] = rollout
                cfg["_GPU_TIER_LABEL"] = label
                # One-line, glanceable confirmation that the tier was applied and
                # what utilization target we are sizing for (Section 10.1).
                print(f"[gpu] {label}: VRAM {vram_gb:.1f}GB -> BATCH_SIZE_ENV="
                      f"{batch}, ROLLOUT_STEPS={rollout} "
                      f"(target >={GPU_UTIL_TARGET*100:.0f}% util)", flush=True)
                break
    else:                                # CPU dev/CI — tiny + fast for smoke tests
        cfg["BATCH_SIZE_ENV"] = 4
        cfg["ROLLOUT_STEPS"] = 64
        cfg["_GPU_TIER_LABEL"] = "CPU"
    return cfg


# ── The master CFG dict ──────────────────────────────────────────────────────
CFG = {
    # Data (paths supplied via CLI / YAML at runtime — never hardcoded to Drive)
    "DATA_CSV_EURUSD": None,
    "SYMBOL":          "EURUSD",

    # Episode
    "EPISODE_BARS":    43_200,    # ~30 trading days of 1m bars
    "BATCH_SIZE_ENV":  64,        # parallel episodes on GPU (auto-tuned per device)
    "LOOKBACK":        20,        # bars of history in state
    "TF_FACTORS":      [1, 15, 60, 1440],   # resample factors from 1m

    # ── FTMO / risk — RUNTIME-CONFIGURABLE RULE INPUTS ───────────────────────
    # (mirrors config/trading_policy.yaml; CLI flags --target-pct / --max-dd-pct /
    #  --daily-target-usd override these at runtime; YAML/CLI win.)
    #
    # daily_increment = INITIAL_EQUITY * DAILY_TARGET_PCT  (a FIXED dollar amount
    # computed once at account open — e.g. $250 on $10k @ 2.5%). It is the unit for
    # the EXCEED progressive bonus and the diagnostic daily_target.
    #
    # REFINED CLASSIFICATION (dd_classification_refine.md): the PASS/OK tier
    # thresholds are measured against INITIAL equity (fixed-$ off the original
    # account), NOT the day's opening balance:
    #     PASS iff final-or-halt >= INITIAL_EQUITY * (1 + DAILY_TARGET_PCT)
    #     OK   iff final-or-halt >= INITIAL_EQUITY * (1 + DAILY_HALF_TARGET_PCT)
    #     FAIL_CAPITAL_LOSS iff final-or-halt < prior-day close (checked FIRST)
    # A DD breach HALTS the day but is NOT an auto-fail (the halt balance is
    # classified the same way). See the single-source-of-truth principles block in
    # core/env/environment.py and core/reward/shaper.classify_day.
    #
    # HONESTY NOTE (rules vs policy): changing these at RUNTIME correctly changes
    # RULE ENFORCEMENT (PASS/FAIL + the DD halt) immediately, everywhere they are
    # used — no retraining needed for the RULES. BUT the trained POLICY was
    # optimized for the target/risk it trained on; running live at very different
    # values still classifies correctly but may degrade the agent's behaviour
    # until retrained. "Rules are config-driven (instant); policy is learned
    # (needs retraining for big changes)."
    "DAILY_TARGET_PCT":   0.025,  # daily profit target as a fraction of INITIAL equity
    # ── HALF-TARGET ("OK" tier) THRESHOLD (dd_classification_refine.md) ────────
    # The OK tier fires when the day's FINAL/halt balance reaches >= 50% of the
    # full target measured AGAINST INITIAL equity. Default 0.0125 == target/2.
    # Set to None to DERIVE it as DAILY_TARGET_PCT/2 automatically (so an override
    # of the target keeps OK at exactly half unless this is pinned). Either way the
    # OK and PASS thresholds are config-driven — NOTHING hardcodes 0.0125 / 250 / 125.
    "DAILY_HALF_TARGET_PCT": 0.0125,  # OK tier: final >= INITIAL*(1+this); None -> target/2
    "DAILY_MAX_DD_PCT":   0.010,  # 1% trailing intraday DD (breach halts the day)
    # ── ACCOUNT SIZE (learning_loop_fix.md FIX 3) ────────────────────────────
    # Default $10,000 so numbers are comprehensible. Override via the CLI flag
    # --account-size or by setting CFG["ACCOUNT_SIZE"] (10000/25000/50000/100000).
    # The env reads ACCOUNT_SIZE first, then falls back to INITIAL_EQUITY. FTMO
    # +2.5%/1% rules are percentage-based off start-of-day equity, and reward is
    # normalized, so everything scales automatically with account size.
    "ACCOUNT_SIZE":       10_000.0,
    "INITIAL_EQUITY":     10_000.0,
    "ACCOUNT_SIZE_CHOICES": [10_000.0, 25_000.0, 50_000.0, 100_000.0],
    "RANDOMIZE_ACCOUNT_SIZE": False,   # FUTURE HOOK — disabled (see env.reset)

    # ── TARGET/RISK-AWARE POLICY (target_aware_policy.md) ────────────────────
    # The policy OBSERVES the active target_pct / max_dd_pct / account-size and is
    # meant to act PURSUANT to them (sizing, timing, risk). To learn a policy that
    # GENERALIZES across target/risk, train with --randomize-ftmo (default OFF):
    # then each EPISODE samples a (target_pct, max_dd_pct[, account_size]) from the
    # ranges below, the env uses them for that episode's classification/DD AND
    # exposes them in the observation. With it OFF the fixed cfg values are used
    # and STILL appear (constant) in the obs, so inference-time changes still shift
    # behaviour via the observation. See item 2 of target_aware_policy.md.
    "RANDOMIZE_FTMO_INPUTS":   False,                 # domain-randomization mode (default OFF)
    "RANDOMIZE_TARGET_RANGE":  [0.01, 0.05],          # per-episode target_pct sample range
    "RANDOMIZE_DD_RANGE":      [0.005, 0.02],         # per-episode max_dd_pct sample range
    "RANDOMIZE_FTMO_ACCOUNT":  False,                 # also sample account_size from CHOICES

    # ── PROPORTIONAL ADAPTATION TO TRAINED BASELINE (item 6) ─────────────────
    # The deterministic, no-retrain scaler layered ON TOP of the agent's chosen lot
    # at inference/eval/live (NEVER forcing direction/exit). It tracks how the
    # CURRENT target/DD differ from what the policy trained on:
    #   target_ratio = current_target_pct / TRAINED_TARGET_PCT
    #   dd_ratio     = current_max_dd_pct / TRAINED_MAX_DD_PCT
    #   effective_lot_scale = clamp(dd_ratio * f(target_ratio), lo, hi)
    # where tighter DD (dd_ratio<1) scales exposure DOWN and a higher target
    # (target_ratio>1) permits more aggression. At baseline (ratios==1) it is
    # EXACTLY 1.0 (no behaviour change). Bounds are configurable. The baseline
    # itself is persisted in the checkpoint metadata (TRAINED_TARGET_PCT /
    # TRAINED_MAX_DD_PCT); these cfg values are the fallback default (0.025/0.01),
    # or the MIDPOINT of the randomization ranges when --randomize-ftmo is used.
    "PROPORTIONAL_SCALER":     True,                  # toggle the item-6 scaler (default ON)
    "PROPORTIONAL_SCALE_LO":   0.25,                  # lower clamp on effective_lot_scale
    "PROPORTIONAL_SCALE_HI":   2.0,                   # upper clamp on effective_lot_scale
    "TRAINED_TARGET_PCT":      0.025,                 # baseline the policy trained at (fallback)
    "TRAINED_MAX_DD_PCT":      0.010,                 # baseline the policy trained at (fallback)
    "MAX_TRADES_PER_DAY": 800,
    "LEVERAGE":           100,    # 1:100 FTMO leverage. Affects margin only —
                                  # PnL per lot is always price_move * lots * 10
                                  # (EURUSD: 100000 units * 0.0001 pip = $10/pip/lot)

    # Agent / training — PURE PPO (DQN deprecated; see DESIGN_DECISIONS.md #1)
    "STATE_DIM":       None,      # filled at runtime from env.state_dim
    "HIDDEN":          256,
    "LR":              3e-4,
    "GAMMA":           0.95,
    "ROLLOUT_STEPS":   2048,      # on-policy steps collected before each PPO update
    "PPO": {                      # PPO hyperparameters (see core/agent/ppo.py)
        "learning_rate": 3e-4,
        "gamma":         0.95,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        # ent_coef raised 0.01 -> 0.02 to keep exploration alive early: the old
        # ~$0 do-nothing policy was a collapse symptom (learning_loop_fix.md FIX
        # 1.5). lot_log_std is also floored in ppo.py so the sizing head can't
        # collapse to a deterministic 0-variance lot.
        "ent_coef":      0.02,
        "vf_coef":       0.5,
        "n_epochs":      4,
        "max_grad_norm": 0.5,
        "lot_log_std_init": -0.5,   # initial log-std for the continuous lot head
        "lot_log_std_min":  -2.0,   # floor so exploration on lot size never dies
    },

    # torch.compile / AMP toggles (auto-disabled on CPU).
    # compile uses mode="default" — NOT "reduce-overhead" which uses CUDA Graphs
    # and overwrites rollout buffer tensors, crashing torch.stack() in update().
    "USE_TORCH_COMPILE": True,
    "USE_AMP":           True,
    # ── COMPILE WARMUP VISIBILITY (compile_warmup_visibility fix) ────────────
    # torch.compile(mode="default") compiles LAZILY on the FIRST forward pass —
    # which happens INSIDE the rollout step loop, AFTER the PHASE banner. That
    # first compile BLOCKS the main thread for ~10-15 min on an A100, during
    # which the wall-clock heartbeat (it lives further down the same loop) never
    # gets a chance to fire. The result was a Colab cell frozen with NO output,
    # indistinguishable from a crash. To make warmup PROVABLY ALIVE we (a) print
    # an explicit announcement right before the first forward, (b) emit an
    # immediate step-0 heartbeat (printed + on disk) before the block, (c) print
    # a "compile finished in Ns" marker right after the first forward returns,
    # and (d) run a stdlib-ONLY watchdog thread (NO torch/CUDA calls — pure
    # time.sleep + print) that prints "still compiling… Ns" every
    # COMPILE_WATCHDOG_SECS until the first forward returns. The watchdog is a
    # daemon, never raises, and is cleanly joined after the first forward.
    "COMPILE_WATCHDOG_ENABLED": True,   # the still-compiling… ticker (default ON)
    "COMPILE_WATCHDOG_SECS":    30,     # ticker cadence during the compile block

    # Curriculum
    "PHASE":                  0,
    "MAX_EPISODES_PER_PHASE": 500,
    "CHECKPOINT_EVERY":       10,
    "EVAL_EVERY":             50,

    # ── Console / heartbeat (learning_loop_fix.md FIX 2) ─────────────────────
    # Heartbeat is WALL-CLOCK time-based (default 300s = 5 min), one-liner.
    "HEARTBEAT_SECS":         300,
    "BARS_PER_DAY":           1440,

    # ── Feature/indicator cache (learning_loop_fix.md FIX 4) ─────────────────
    # On first build the TF indicators + feature matrix are cached to disk keyed
    # by CSV path+mtime+feature-config hash; restart loads in seconds. Dir is
    # configurable (default next to the checkpoint/data Drive dir). None = auto.
    "FEATURE_CACHE_DIR":      None,
    "USE_FEATURE_CACHE":      True,

    # ══════════════════════════════════════════════════════════════════════════
    # REWARD SYSTEM (redesign — see reward_redesign_plan.md, Sections 1-4 & 7).
    # ──────────────────────────────────────────────────────────────────────────
    # ALL reward weights are normalized / percent-of-day-start units so the system
    # is ACCOUNT-SIZE INVARIANT and stays O(1) per step (raw dollars would blow up
    # PPO). Dense per-step shaping + the sparse terminal DAY bonus share one scale.
    #
    # The weights below ENCODE THE PRIORITY STACK (what the bot optimizes, in
    # order) so the gradient points the agent at the right thing:
    #   1. STREAKS (consecutive pass days)  — the #1 signal (exponential curve S2)
    #   2. DD EFFICIENCY (hit target on minimal DD budget)  — S1.3 multiplier
    #   3. DAILY TARGET HIT RATE            — the 5-tier day bonus (S1)
    #   4. IMPROVEMENT OVER TIME            — episode improvement multiplier (S4.3)
    #   5. PROFIT BEYOND TARGET             — EXCEED progressive bonus (S1, no cap)
    #
    # NOTE on the 5 TIERS (Section 1): a day is classified by where its ENDING (or
    # DD-HALT) balance lands vs the daily target (= day_start + fixed increment):
    #   FAIL     ending < 50% of target progress (and/or DD breached low)
    #   OK       50% <= ending < 100% of target   (linear partial credit)
    #   PASS     ending >= 100% of target          (full pass_day_bonus)
    #   EXCEED   ending >  100% AND never breached  (PASS + progressive, no cap)
    #   SURVIVAL traded all day, NEVER breached DD  (big bonus stacked on the tier)
    # A breached day can never earn SURVIVAL or EXCEED (RESOLVED DECISION 2).
    "REWARD": {
        # ── Section 1: tier bonuses/penalties (terminal, applied at day close) ──
        "pass_day_bonus":    2.0,    # full bonus at PASS (ending >= 100% of target)
        "fail_day_penalty":  -2.0,   # full FAIL penalty (ending < 50% of target)
        # OK tier earns a LINEAR partial credit between these fractions of
        # pass_day_bonus as ending climbs 50%->99% of target (S1.1).
        "ok_partial_lo":     0.25,   # credit fraction at 50% of target
        "ok_partial_hi":     0.95,   # credit fraction just under 100% of target
        # EXCEED: progressive bonus per +100% of target beyond it (NO cap, S1.1).
        # bonus = pass_day_bonus + exceed_scale * (excess / daily_increment).
        "exceed_scale":      1.0,
        # SURVIVAL: big bonus stacked on top of whatever tier earned, ONLY when the
        # day traded and NEVER breached the trailing DD (S1.1).
        "survival_bonus":    1.5,
        # Section 1.2 — RED-DAY linear penalty: a negative-PnL day is punished in
        # PROPORTION to loss magnitude (lose $50 = 5x lose $10), ON TOP of the FAIL
        # tier penalty. Scaled in percent-of-day-start units (loss_pct / target_pct).
        "red_day_scale":     1.0,
        # Section 1.3 — DD-EFFICIENCY multiplier: on OK/PASS/EXCEED days multiply
        # the (positive) day reward by an efficiency factor derived from how little
        # of the DD budget was used. 1.0 when DD≈0, falling toward (1-weight) when
        # the full budget was consumed. Encodes priority #2 (DD efficiency).
        "dd_efficiency_weight": 0.5,
        # legacy low-DD sweetener kept for the Φ/diagnostics path (still harmless).
        "low_dd_threshold":  0.005,
        "low_dd_bonus":      0.3,

        # ── Section 2: streak system (the #1 priority signal) ──────────────────
        # Positive streak reward = a*(exp(b*(streak-1))-1) fit to the anchors
        # Day1=base only (0 extra), Day3=+0.3, Day5=+1.0, Day10=+5.0, Day15=+12.0.
        # The fitted (a,b) are computed in shaper.fit_streak_curve(); these are the
        # cached defaults (overridable). See the Day1/3/5/10/15 table in the report.
        "streak_curve_a":    0.616998,  # fitted amplitude (see fit_streak_curve)
        "streak_curve_b":    0.221749,  # fitted growth rate
        "streak_base":       0.5,    # flat Day-1 base added to a passing day
        "mulligan_count":    1,      # free fails per streak (two CONSECUTIVE break it)
        "negative_streak_mult": 1.5, # neg streak mirrors positive curve at 1.5x (S2.3)
        "consec_fail_escalation": 0.5,  # extra penalty PER consecutive fail (S2.4)
        "recovery_bonus":    3.0,    # flat bonus for a PASS that breaks a fail streak (S2.5)
        "momentum_bonus":    0.2,    # small positive bias the day AFTER a pass (S2.6)

        # ── Section 3: intra-day progressive + give-back protection ────────────
        # Linear intra-day pull as equity approaches the daily target (S3.1). The
        # per-bar increment of clamped progress is scaled by this.
        "intraday_progress_scale": 0.5,
        # On a FAIL day, retroactively WIPE that day's accrued intra-day progress
        # and add a penalty proportional to the give-back from the intra-day HIGH
        # to the close (S3.2): teaches "don't give back gains".
        "intraday_wipeout":         True,
        "giveback_from_high_scale": 1.0,
        # Cross-day give-back (S3.3): if equity falls from the multi-day PEAK,
        # penalize the drop (protects multi-day gains). Percent-of-initial units.
        "cross_day_giveback_scale": 0.5,

        # ── Dense per-step shaping (percent-of-day-start units), kept from before ──
        "step_pnl_scale":        1.0,     # weight on Δequity/day_start_eq each bar
        "target_progress_scale": 0.5,     # extra pull while below+toward target
        "dd_proximity_scale":    0.02,    # quadratic penalty as DD nears the cap
        "overtrade_penalty":     0.0005,  # small cost per new trade (DO NOT raise —
                                          # trade VOLUME is intentional/phase-driven)

        # ── Section 7: speed bonus ─────────────────────────────────────────────
        # If a trade shows unrealized profit (NET of this symbol's commission)
        # within SPEED_BONUS_MINUTES of entry, accrue a PENDING speed bonus; it is
        # KEPT only if the trade eventually CLOSES in profit, else revoked (S7).
        "speed_bonus":          0.3,
    },

    # ── Section 7: speed-bonus window (Colab-tunable) ────────────────────────────
    # Minutes after entry within which an unrealized-profit trade earns the pending
    # speed bonus. 1 bar == 1 minute on M1 data, so this is also the bar window.
    "SPEED_BONUS_MINUTES":   3,

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — TRADING SESSIONS (FTMO CEST clock) + session filter.
    # ──────────────────────────────────────────────────────────────────────────
    # Our 1m bars carry no real timestamp (alignment is by integer row position),
    # so the env derives a SYNTHETIC CEST time-of-day from the bar-of-day index and
    # maps it to a SESSION via this table. Each row is
    #   (name, start_minute, end_minute, code)  in CEST minutes-of-day [0,1440).
    # `code` is the integer session id; the observation exposes code/len so the four
    # sessions map to ~{0.25,0.5,0.75,1.0} (0.0 == outside/closed). These are the
    # standard FX sessions in CEST (summer): Asian 02:00-10:00, London 09:00-18:00,
    # NY 14:00-23:00, with the London/NY OVERLAP 14:00-18:00 flagged separately as
    # the highest-liquidity window. Overlap is checked first so it wins.
    "TRADING_SESSIONS": [
        # (name, start_min, end_min, code)  — CEST minutes of day
        ("london_ny_overlap", 14 * 60, 18 * 60, 4),   # 14:00-18:00 (highest liq)
        ("asian",              2 * 60, 10 * 60, 1),    # 02:00-10:00
        ("london",             9 * 60, 18 * 60, 2),    # 09:00-18:00
        ("ny",                14 * 60, 23 * 60, 3),    # 14:00-23:00
    ],
    "N_SESSIONS": 4,            # normalizer for the session-code obs (code/N_SESSIONS)
    # Minute-of-day the SYNTHETIC clock assigns to the FIRST bar of each trading
    # day (so bar 0 == this CEST minute). FTMO's day rolls at ~00:00 CEST; we open
    # the synthetic clock at the Asian session start so early bars land in a real
    # session rather than the dead 00:00-02:00 window.
    "SESSION_DAY_OPEN_MIN":  2 * 60,   # 02:00 CEST == first bar of the day
    # Optional SESSION FILTER (Colab-tunable): when set to a session NAME, the env
    # only force-enters / favors trading inside that session (the agent still
    # observes the session code regardless). None == trade all sessions (default).
    # This is an OBSERVATION/curriculum aid only — it never overrides the strategy
    # gate. Left None so existing behaviour is unchanged.
    "SESSION_FILTER":        None,

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — TRANSACTION COSTS (multi-asset framework; EURUSD active now).
    # ──────────────────────────────────────────────────────────────────────────
    # Commission is charged at BOTH trade OPEN and trade CLOSE (the agent feels the
    # real per-trade round-trip cost) and SCALES WITH LOT SIZE. The table is keyed
    # by ASSET CLASS; resolve_commission() in environment.py maps a SYMBOL to its
    # class and returns the per-SIDE cost for a given lot. Forex is the active path
    # (EURUSD): $5.00/standard lot ROUND TRIP == $2.50/lot/side. So 0.5 lot EURUSD
    # costs $2.50 round trip ($1.25/side); 2.0 lot costs $10 round trip.
    #   • "per_lot_round_trip": flat $ per standard (1.0) lot for the full round
    #     trip (forex). Per-side = per_lot_round_trip/2 * lots.
    #   • "pct_notional": fraction of NOTIONAL (lots*contract*price) charged PER
    #     SIDE (metals, crypto). round trip = 2x.
    # Everything else (indices/oils/agriculture) is $0 (no commission).
    "COMMISSION": {
        "forex":       {"kind": "per_lot_round_trip", "value": 5.00},   # $5/std lot RT
        "indices":     {"kind": "zero",               "value": 0.0},    # *.cash
        "metals":      {"kind": "pct_notional",        "value": 0.000014},  # 0.0014%/side
        "oils":        {"kind": "zero",               "value": 0.0},
        "agriculture": {"kind": "zero",               "value": 0.0},
        "crypto":      {"kind": "pct_notional",        "value": 0.00065},   # 0.065%/side
    },
    # Symbol -> asset-class routing. EURUSD (and any *USD / 6-char FX pair) -> forex.
    # The explicit lists below cover the spec's named instruments; the fallback in
    # resolve_commission() classifies anything else heuristically (e.g. trailing
    # ".cash" -> indices) and defaults to forex for 6-letter pairs.
    "COMMISSION_SYMBOLS": {
        "metals":      ["XAUUSD", "XPDUSD", "XPTUSD"],
        "oils":        ["UKOIL.cash", "USOIL.cash"],
        "agriculture": ["COCOA.c", "COFFEE.c", "SOYBEAN.c", "WHEAT.c"],
        # crypto/indices identified heuristically (BTC*/ETH* -> crypto; *.cash ->
        # indices) in resolve_commission(); listed here only for documentation.
    },
    "CONTRACT_SIZE": 100_000.0,   # standard FX lot units (EURUSD); notional = lots*this*price

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — LOT-SIZING CURRICULUM (narrow -> wide, configurable PER PHASE).
    # ──────────────────────────────────────────────────────────────────────────
    # The PPO lot HEAD always emits a raw [0,1] mapped onto [0.01, MAX_LOT] (the
    # head is unchanged). On TOP of the head we apply a CURRICULUM CLAMP that starts
    # NARROW (so the bot learns direction/timing before size) and WIDENS as it
    # advances through the strategy phases. The clamp is purely a [lo, hi] lot
    # window; the agent still learns WHERE inside it to size (S8.3 contextual
    # sizing is learned, not hardcoded). DO NOT add overtrade penalties (S8 / the
    # hard CONSTRAINT — trade volume is intentional).
    #   • LOT_CURRICULUM maps a strategy-phase NAME to its [lo, hi] lot window.
    #     "_default" is used for any phase not listed; "beast" (or BEAST_MODE on)
    #     removes the hard cap (clamps to MAX_LOT only, no narrowing).
    #   • Early phases trade small (0.1-0.5); later phases open up to the full head.
    "LOT_CURRICULUM_ENABLED": True,
    "MAX_LOT":            2.0,    # head ceiling (kept); curriculum clamps within it
    "BEAST_MODE":         False,  # when True (or live_improve): no narrowing, cap=MAX_LOT
    "BEAST_MAX_LOT":      2.0,    # normal-mode cap is configurable; beast lifts narrowing
    "LOT_CURRICULUM": {
        "_default":            [0.10, 0.50],   # narrow default (learn direction first)
        "phase1_cci_align":    [0.10, 0.50],
        "phase0_cci_extreme":  [0.10, 0.75],
        "phase2":              [0.10, 1.00],
        "phase3":              [0.10, 1.25],
        "phase4":              [0.10, 1.50],
        "phase5":              [0.10, 1.75],
        "phase6":              [0.01, 2.00],
        "phase7_full_ftmo":    [0.01, 2.00],   # full head by the final FTMO phase
        "live_improve":        [0.01, 2.00],   # beast / live: full range
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — EXPLORATION SCHEDULE (entropy annealing high -> normal).
    # ──────────────────────────────────────────────────────────────────────────
    # Start exploration HIGH so the agent samples a wide spread of direction/lot
    # early, then anneal LINEARLY back to the stable PPO ent_coef by episode
    # ENTROPY_ANNEAL_EPISODES (default 20). The annealed coefficient is applied in
    # the PPO loss; all params are config-driven (S9.2). At/after the anneal end it
    # equals PPO.ent_coef exactly (no residual perturbation).
    "ENTROPY_ANNEAL_ENABLED":  True,
    "ENTROPY_START_COEF":      0.10,   # high exploration at episode 0
    "ENTROPY_ANNEAL_EPISODES": 20,     # linearly reach the stable ent_coef by here

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11 — STRATEGY-PHASE GATE (advance on a pass-streak).
    # ──────────────────────────────────────────────────────────────────────────
    # Advance to the NEXT strategy phase (config/phases.yaml) once the bot reaches
    # PHASE_ADVANCE_STREAK consecutive passing days WITHIN A SINGLE EPISODE. This
    # advances through the STRATEGY phases (NOT a reward phase — the reward system
    # is universal). The required streak and the STARTING phase are Colab inputs
    # (--phase-advance-streak / --start-phase). Default 10 (S11.1).
    "PHASE_ADVANCE_STREAK":  10,

    # GPU auto-tune master toggle (Section 10). When True (default) auto_tune_batch
    # picks the (batch, rollout) tier for the attached card so it is never idle.
    "AUTO_TUNE_GPU":  True,

    # ── Potential-based reward shaping (Φ) ───────────────────────────────────
    # Φ = (pass_rate × avg_ret_normalised) / (1 + λ × avg_dd_normalised)
    # SHAPE_WARMUP REMOVED (Section 4.1): the Φ consistency bonus is active from
    # EPISODE 1 (warmup=0). The key is kept at 0 so any stale reader stays safe.
    "SHAPE_ALPHA":   0.006,
    "SHAPE_CLIP":    0.006,
    "SHAPE_LAMBDA":  5.0,
    "SHAPE_WARMUP":  0,      # Section 4.1: phi active from episode 1 (was 150)
    "WEEKLY_BONUS":  0.02,    # weekly_consistency_bonus when 7-day pass rate improves
    "PASS_NO_BREACH_BONUS": 0.01,   # bonus when target hit AND no DD breach

    # ══════════════════════════════════════════════════════════════════════════
    # INTERPRETABILITY + DASHBOARD I/O (post-hoc only — ZERO training-loop cost
    # except the cheap action-distribution logger below, which is toggleable).
    # ──────────────────────────────────────────────────────────────────────────
    # Where the Colab "💾 Save Snapshot" cell writes params snapshots + the master
    # snapshot_log.json, and where the training results writer (PART 1) looks them
    # up by params_hash to append a results block. Config-driven (never hardcoded
    # downstream) so a non-Colab run can point it at a local dir. The default mirrors
    # the spec's Drive path; tests override it with a tmp dir.
    "SNAPSHOT_DIR":  "/content/drive/MyDrive/snapshots/params",

    # ── ACTION-DISTRIBUTION LOGGER (PART 3) ──────────────────────────────────
    # Lightweight, NO SHAP: every LOG_ACTION_DIST_EVERY rollout steps the train
    # loop appends the current batch's mean action-prob distribution + market state
    # to {metrics_dir}/action_distributions.csv. Toggleable + interval-configurable
    # so it has negligible overhead and can be turned off entirely. The optional
    # per-N-episode "shift" print is gated by LOG_ACTION_DIST_EPISODE_SUMMARY.
    "LOG_ACTION_DIST":            True,   # master toggle for the CSV logger
    "LOG_ACTION_DIST_EVERY":      100,    # rollout steps between CSV rows
    "LOG_ACTION_DIST_EPISODE_SUMMARY": True,   # print the per-episode shift line

    # ── SHAP (PART 2/6) — post-hoc, OPTIONAL, import-guarded ─────────────────
    # RUN_SHAP gates the (slow, ~60-120s) SHAP pass in the interpretability cell;
    # default OFF so Run-All stays fast. SHAP background/explain sample sizes are
    # config-driven (200-500 bg, <=500 explained) per the spec. The always-on fast
    # path (saliency -> action-dist -> report) never reads these.
    "RUN_SHAP":               False,
    "SHAP_BACKGROUND_SAMPLES": 256,   # background obs for GradientExplainer (200-500)
    "SHAP_EXPLAIN_SAMPLES":    200,   # obs explained per head (<=500)
}
