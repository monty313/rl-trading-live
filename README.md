# rl-trading-live

GPU-optimized (NVIDIA A100) reinforcement-learning forex trading system with a
756-action DQN, YAML-driven strategy phases, FTMO + Beast risk modes, an MT5 live
runner, and **Jordan** — a read-only, permission-gated Wolf-of-Wall-Street coach
with a Streamlit dashboard.

> Repo alias: the project is also referred to as `jordan-rl-mt5`. The canonical
> name used throughout the code and the Colab notebook is **`rl-trading-live`**.

Code lives in GitHub; **data and checkpoints live in Google Drive** (Colab is
ephemeral). The Colab notebook (`rl_trading_colab.ipynb`) is the one entry point.

---

## Prerequisites

- Google Colab with **A100** access (Colab Pro recommended for guaranteed A100)
- Google Drive with data files at `/MyDrive/RL-Trading-Data/` (see `core/pipeline.py`
  / the notebook for exact paths)
- For live trading: a **Windows** machine with the MetaTrader 5 terminal installed
- A GitHub account (to clone this repo)

## Colab Training Setup (copy-paste)

1. Go to https://colab.research.google.com
2. Upload `rl_trading_colab.ipynb`
3. Runtime → Change runtime type → GPU → **A100**
4. Run cells **1 → 6** in order:
   - CELL 1 GPU check (confirms A100, >30GB VRAM)
   - CELL 2 mount Drive (confirms primary data file)
   - CELL 3 install deps
   - CELL 4 clone/pull repo (also clears stale module cache)
   - CELL 5 `inspect_system.py` (aborts if anything is red)
   - CELL 6 training (resumes from your last checkpoint; runs indefinitely)
5. Training checkpoints auto-save to Drive — close the tab any time.

## Local / CI validation (no GPU needed)

```bash
pip install -r requirements.txt
python inspect_system.py            # the one command — exit 0 if all green
python tests/run_all_tests.py       # COMPLETE SYSTEM TEST (unit + integration)
```

`inspect_system.py` marks GPU/Drive-only checks as ⚠️ SKIP on CPU/CI (never faked
as passing). On the A100 in Colab those become ✅.

---

## Adding a New Strategy (NO CODE REQUIRED)

1. Open `config/phases.yaml`.
2. Add a block:
   ```yaml
   - name: your_strategy_name
     order: 3                       # phases run in ascending order
     instruments: [EURUSD]
     entry_conditions:
       buy:  "cci_14 < -150 and close > ema_20"
       sell: "cci_14 > 150 and close < ema_20"
     max_episodes: 500
     advance_criteria:
       consecutive_pass_days: 5
   ```
3. Run `python scripts/validate_phases_yaml.py` — confirm **PASS**.
4. Commit + push. Re-run Colab CELL 4 (pull) and CELL 6 (training resumes + new phase).

If you reference a variable the system doesn't recognize, the validator prints
**exact** IRAC instructions on what to add and where. The allowed variables are in `core/env/conditions_engine.py` `VARIABLE_REGISTRY`
(derived from `core/env/indicators.py` `FEATURE_COLUMNS`) and include the
multi-timeframe gate variables: `cci30, cci100, cci30_sma1_sh8, cci100_sma1_sh8,
bb20_upper/mid, bb200_upper/mid, high_sma4_sh8, low_sma4_sh8, atr14,
atr14_sma1_sh8, atr45, atr45_sma1_sh8, bb20_upper_sma4_sh8` plus the basics
(`close, open, high, low, volume, sma_20, ema_20, cci_14, atr_14`). CCI(300) and
CCI(900) are intentionally excluded (too slow on 1m).

### Curriculum (authoritative)
The canonical curriculum is the **8-phase** system in `config/training_config.yaml`
and `config/phases.yaml` (phase0 CCI Extreme → … → phase7 Full FTMO → infinite
`live_improve`), with `force_in_and_gate` / `open_gate` / `free` mask semantics.
Indicators are computed per timeframe (1m resampled to 15m/30m/1H/1D internally).
If `talib` is installed it is used; otherwise a numpy fallback computes the same
columns so the repo runs clone-and-run on Colab/CI. See `SPEC_STRATEGY.md`.

## Promoting a Model to Live

1. Open the Jordan dashboard (Colab CELL 7 → localtunnel/ngrok URL).
2. Go to **Training Control**.
3. Verify `best_eval.pt` metrics (Φ, PASS rate).
4. Click **Promote Best Eval → Live** and confirm.
5. `live_trading.pt` is updated on Drive; `live_runner.py` loads it on the next bar.

## Live Trading (Windows MT5 machine)

```bash
git clone https://github.com/monty313/rl-trading-live
pip install -r requirements.txt
pip install MetaTrader5            # Windows only
copy .env.example .env             # then fill in MT5 + GROK + Telegram values
python -m broker.live_runner       # uses live_trading.pt from Drive
```

## Crash Recovery

1. Reconnect to Colab.
2. Run CELL 1 → CELL 4.
3. Run CELL 8 (crash recovery) — verifies checkpoints, finds best resume.
4. Run CELL 6 — training resumes from the best checkpoint.

## Emergency Halt

- **A:** Jordan dashboard sidebar → red **EMERGENCY HALT** button.
- **B:** `python -c "from core.risk.daily_guard import DailyGuard; DailyGuard('ftmo',100000,{}).force_halt()"`
- **C:** Close the MetaTrader 5 terminal on the Windows machine.

---

## What you need to fill in (`.env`)

| Setting | Why |
|---|---|
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | live MT5 trading (Windows) |
| `GROK_API_KEY` | Jordan's Grok personality (fallback one-liners fire without it) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram alerts (optional) |

Credentials are read via `os.getenv()` only — **never** hardcoded (enforced by
`inspect_system.py`).

## Jordan is read-only

Jordan never writes code, never trades, never deploys. Its only file output is
`logs/jordan_reports/pending_patch_*.md`, and only after **two** user approvals
(`jordan/consent_flow.py`). Jordan can also read the **complete system test**
results (`tests/run_all_tests.py` → `jordan_summary()`) to report build health.

## Repo layout

```
config/        phases.yaml (strategies) · trading_policy.yaml · jordan_sources.yaml
core/          env (indicators, fills, environment, conditions) · agent (dqn, action_space)
               reward (shaper) · risk (sizer, daily_guard, trade_gate) · pipeline · settings
training/      train.py · checkpoint_manager.py · eval_loop.py
backtest/      engine.py (md5 parity assertion)
broker/        broker_base · mt5_adapter · live_runner · account_manager · stubs
jordan/        irac_engine · persona · policy_inspector · consent_flow · vitals_daemon · repo_indexer
dashboard/     app.py + pages/ (FTMO HQ, Training Control, Jordan, Performance, Beast Mode)
monitoring/    flatline_detector · alert_dispatcher
scripts/       validate_phases_yaml · crash_recovery · smoke_train/backtest/infer
tests/         unit/ · integration/ · fixtures/ · mocks/ · run_all_tests.py (one-test-tester)
inspect_system.py   rl_trading_colab.ipynb   AUDIT.md   INTERFACES.md
```

---

## Known issues & operational caveats

These are real, expected behaviors — not bugs:

- **torch.compile warmup:** the first ~10-15 min / first few episodes are slow.
  This is normal warmup, **not a crash** — let it run.
- **Colab localtunnel is flaky:** if the dashboard URL won't load, use the ngrok
  fallback in CELL 7.
- **FAISS index build:** Jordan's first full-codebase index can take 2-3 minutes —
  it may look stuck but isn't (a keyword fallback runs if FAISS is unavailable).
- **MT5 symbol suffixes:** brokers vary (`EURUSD`, `EURUSD.sim`, `EURUSDm`, `.r`,
  `.pro`, `.ecn`). `mt5_adapter` auto-resolves via the alias list in
  `trading_policy.yaml` — extend it if your broker uses another suffix.
- **Transfer learning (7→756 actions):** on first resume from an old checkpoint,
  the output layer is re-initialized and epsilon is raised (`TRANSFER_EPSILON`)
  for `TRANSFER_EPISODES`. Verify the `[transfer]` log line prints on first run.
- **MQL5 EA (separate from this Python repo):** if you also run the MetaTrader EA,
  prior builds hit per-symbol bar tracking, margin-aware lot sizing, `FILE_COMMON`
  path mismatch, Windows-path `\U` unicode escapes, and `.sim` suffix issues — all
  resolved in the EA; keep symbol lists as base names and let `ResolveSymbol` map
  suffixes.

## Hard rules (enforced)

Every order calls `trade_gate.approve()` (no bypass) · credentials only via
`.env`/`os.getenv` · indicators + fills are md5-parity-checked across train /
backtest / live · checkpoints `latest.pt` / `best_eval.pt` / `live_trading.pt` /
`transfer_start.pt` are never deleted · training always resumes unless
`--force-fresh` · Jordan is read-only and double-consent-gated.
