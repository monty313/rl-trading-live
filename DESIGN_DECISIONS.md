# DESIGN_DECISIONS.md — Locked answers (source of truth)

These resolve the open design questions. They OVERRIDE earlier assumptions in
the codebase. Future changes must not silently reintroduce deprecated paths.

## 1. Agent = PURE PPO (DQN deprecated)
- The single agent is **PPO**. It outputs: **direction** (BUY / SELL / FLAT),
  **lot size**, and **exit actions** (close / reduce / hold).
- DQN is deprecated: no training or inference path depends on it. The old
  `DQNAgent` is moved to `legacy/` (kept for reference, not imported by the live
  system). Remove DQN-specific config (epsilon, replay buffer, target-net sync).
- Action encoding lives in ONE place (`core/agent/action_space.py`): a clearly
  defined discrete/continuous split — discrete direction + exit, continuous (or
  bucketed) lot size.
- Checkpoints, eval, self-healing/rollback reference **PPO only**.

## 2. Forced entry: force a trade, NEVER choose the direction
- **When ANY strategy gate is ACTIVE, FLAT/HOLD is NOT an option — in EVERY**
  **phase** (force_in_and_gate AND open_gate alike). The agent must always hold
  a position while the strategy is active; it may flip BUY<->SELL but can never
  go flat. (User: "when any strategy is active hold is not an option during any
  phase.")
- The direction mask therefore allows only {BUY, SELL} while a gate is active.
  `must_enter` is True when the agent is currently flat (forced to open now).
- The code **NEVER** picks BUY vs SELL — the model learns direction entirely on
  its own (corrects the old code which defaulted to long).
- The agent chooses lot size freely (PPO continuous head). When the gate flips
  INACTIVE while a position is open, the position is NOT force-closed — the agent
  manages the exit (FLAT becomes available again only when the gate is inactive).

## 3. talib REQUIRED everywhere (single indicator source of truth)
- `TA-Lib` is a hard dependency (in requirements; installed in the Colab notebook
  via `apt-get install -y ta-lib && pip install TA-Lib`; Windows via wheel).
- All indicators come from talib. The pure-numpy path is an explicit
  degraded/TEST-ONLY mode and must NEVER be mixed with talib in a parity run.
- Rationale: numpy != talib bit-for-bit (esp. ATR Wilder smoothing affects the
  phase-6 ATR-expansion gate threshold crossings).

## 4. State vector = 4-timeframe stack
- State = `lkbk × F × len(TF_FACTORS) + 6`, `TF_FACTORS=[1,15,60,1440]`,
  per-window mean/std normalization (matches existing checkpoints' input layer).

## 5. PASS/FAIL (single definition) + intraday DD stop
- `pass` = `daily_return >= 2.5%` AND `daily_dd <= 1%`.
- `fail` = trailing DD >= 1% OR end-of-day balance < 2.5% of initial.
- When intraday trailing DD hits 1%, the **trading day ends immediately**.
- This one definition governs BOTH phase advancement and the day reward.

## 6. Episode length = 30 days (~1 month)
- `EPISODE_BARS` = 30 trading days of 1m bars.

## 7. Phase advancement
- `consecutive_pass` is a **phase-level** counter that PERSISTS across episodes
  within a phase (does NOT reset each episode). Advance at
  `advance_consecutive_pass_days = 5` (or `max_episodes_per_phase` cap).

## 8. Rewards: keep BOTH Φ shaping AND progressive day bonuses
- Φ potential shaping (consistency) AND pass/ok/fail + streak + low-DD day
  bonuses are BOTH active simultaneously.

## 9. Multi-symbol from phase 0
- All 4 symbols (EURUSD, GBPUSD, XAUUSD, US30) trained from phase 0 onward.
- One env per symbol; reward averaged across symbols each step.
- Live: bot trades all MT5 symbols while attached to one chart.

## 10. Time: FTMO CEST end-of-day
- Daily boundary = **Europe/Berlin (CET/CEST)** midnight, matching FTMO.

## 11. Checkpoint transfer
- With the 4-TF stacked state restored, `state_dim` aligns with the existing
  checkpoint's input layer. At load, READ the checkpoint's first-layer shape:
  transfer input+hidden when it matches; else transfer hidden-only (logged).
- NOTE: moving DQN->PPO means the action head differs; transfer is best-effort on
  shared feature layers only.

## Scope note
These files generalize the policy/strategy so the user's real Drive
checkpoints/data slot in. Match principles & interfaces over hardcoded specifics.
