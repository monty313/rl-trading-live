# SPEC_STRATEGY.md — Locked Curriculum & Indicator Spec (source of truth)

## AUTHORITATIVE: 8-phase curriculum (config/training_config.yaml)

The canonical curriculum is the **8-phase** system from REPO1's
`config/training_config.yaml` (phase0..phase7), now ported into
`config/phases.yaml` with named masks in `core/env/conditions_engine.py`
`MASK_REGISTRY`. This supersedes the earlier 4-phase PDF exploration below
(kept for reference). CCI(300) and CCI(900) are intentionally NOT computed
(too slow on 1m data).

| Phase | Name | mask_type | gate TFs |
|---|---|---|---|
| 0 | CCI Extreme Gate | force_in_and_gate | 1m, 15m |
| 1 | CCI Directional Alignment | open_gate | 1m, 15m |
| 2 | Lagged High/Low SMA — Trend | force_in_and_gate | 1m, 30m |
| 3 | Lagged High/Low SMA — Counter-TF | force_in_and_gate | 1m, 15m |
| 4 | Bollinger Band Position | force_in_and_gate | 1m, 15m |
| 5 | SMA(2) Stack Alignment | force_in_and_gate | 1m, 60m |
| 6 | ATR Expansion | force_in_and_gate | 1m, 60m |
| 7 | Full FTMO — No Mask | free | — |

`force_in_and_gate`: when the gate is TRUE the agent MUST be in a trade (forced
entry if flat); when FALSE, opening new trades is blocked (existing positions
may stay open). `open_gate`: gates opening only; agent learns when to exit.
`free`: no masking. Reward weights (pass/ok/fail + streak + low-DD) come from
the `REWARD` block of `training_config.yaml`.

---

## Earlier 4-phase PDF exploration (reference only — superseded above)

Captured verbatim from the user's locked design (sessions 2026-05-28). Retained
for traceability; the 8-phase `training_config.yaml` is what the code implements.

## Timeframes
Indicators are computed on **1D, 1H, 15m, 1m**. Phase gating uses specific TF
pairs (Phase 1 & 3: 1m + 15m; Phase 2: 1m + 1H).

## Indicators required for the 4-phase gates (per asset, per timeframe)
- **CCI(30)** and `SMA(1) shift +2` applied to CCI(30)
- **CCI(100)** and `SMA(1) shift +2` applied to CCI(100)
- **ATR(14)** and `SMA(1) shift +2` applied to ATR(14)
- **BB(20)** and **BB(200)** — upper band, middle line
- `SMA(4) shift +8` applied to the BB **upper band** (per TF)

(Broader STRAT-001…011 indicator library also defined; the 4 phases need the above.)

## Phase 1 — CCI alignment (gate TFs: 1m + 15m)
- New trades allowed ONLY when, on **both** 1m and 15m, in the **same direction**:
  - both above:  `CCI30 > SMA1_s2(CCI30)` AND `CCI100 > SMA1_s2(CCI100)`, OR
  - both below:  `CCI30 < SMA1_s2(CCI30)` AND `CCI100 < SMA1_s2(CCI100)`
  - …and that same "both-above" or "both-below" holds on 15m too (same direction).
- Agent chooses buy/sell + lot.
- **Mask mode = ENTRY_ONLY**: when the condition is FALSE, **no new entries**, but
  **existing positions may remain open** and the agent must learn when to close
  them (HOLD + exit/close allowed; only opening a NEW position is masked).

## Phase 2 — BB-upper SMA regime (gate TFs: 1m + 1H)
- Prior (Phase 1) conditions OFF.
- Define `SMA_upper4_s8 = SMA(4) shift +8` applied to BB upper band.
- Condition (same direction on both TFs):
  - above-on-both: `price_1m > SMA_upper4_s8_1m` AND `price_1H > SMA_upper4_s8_1H`, OR
  - below-on-both: `price_1m < SMA_upper4_s8_1m` AND `price_1H < SMA_upper4_s8_1H`
- **Mask mode = MUST_HOLD**: when condition holds, the agent **must be in an active
  trade** (if flat, env forces entry; agent still chooses direction + lot).
  Outside the condition, flat is allowed.

## Phase 3 — BB midline regime (gate TFs: 1m + 15m)
- Prior conditions OFF. Use **middle line** of BB(200) and BB(20).
- Condition (same direction on both TFs):
  - above-on-both: `price > BB200_mid` AND `price > BB20_mid` on 1m AND on 15m, OR
  - below-on-both: `price < BB200_mid` AND `price < BB20_mid` on 1m AND on 15m
- **Mask mode = MUST_HOLD** (forced entry when condition holds; flat allowed outside).

## Phase 4 — Free FTMO
- All gates OFF. Trade freely. Goal: **+2.5%/day** with **≤1% trailing daily DD**.
- Mask mode = FREE.

## Reward — progressive cross-day consistency (all phases)
- Per day: `pass` if `r_d ≥ 2.5%` AND `dd_d ≤ 1%`; `acceptable` if `0 ≤ r_d < 2.5%`
  AND `dd_d ≤ 1%`; `fail` if `dd_d > 1%` or `r_d < 0`.
- `reward_day = base(pass/ok/fail) + α·streak_len + β·max(0, 1% − dd_d)`
  - streak bonus rewards consecutive passing days; low-DD bonus rewards days well
    under the 1% limit. Lower DD and more passing days in a row count for more.

## Mask-mode enum (env + conditions_engine must support)
- `FREE`        — no masking (Phase 4 / live_improve)
- `ENTRY_ONLY`  — when gate false: mask opening NEW positions; allow HOLD + exits
                  (Phase 1)
- `MUST_HOLD`   — when gate true: force being in a trade (mask HOLD/flat); agent
                  chooses direction + lot. When gate false: flat allowed (Phase 2, 3)
- Same-direction multi-TF gating: a gate is a structured condition, not a single
  buy/sell string — it evaluates "both-above or both-below across the listed TFs."
