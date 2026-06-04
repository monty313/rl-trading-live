# COLAB RUNBOOK — READ FIRST (humans & LLMs)

This is the **single source of truth** for how to run training on Google Colab
end-to-end, and how to fix the errors you are most likely to hit. The Colab
notebook (`rl_trading_colab.ipynb`) and the training crash banner both point
here.

Code lives in **GitHub**; data + checkpoints live in **Google Drive** (Colab
storage is ephemeral). The notebook clones the repo fresh each session and reads
data/checkpoints from your mounted Drive.

> ⚠️ **THE #1 GOTCHA:** If you **restart the runtime** or it **times out**,
> Google Drive **UNMOUNTS**. You **MUST re-run Cell 2 (MOUNT DRIVE)** before
> Cell 6 (training), or training dies with `FileNotFoundError` on the CSV.
> **That error is an unmounted Drive — NOT a code bug.** Do not start rewriting
> the data loader; re-mount Drive and verify the file (see Troubleshooting).

---

## Run order (top to bottom)

Run every cell **in order**. For each step: what it does, the success signal to
look for, and the most common failure + its one-line fix.

### 1. GPU CHECK
- **Does:** Asserts an A100 GPU with >30 GB VRAM is attached.
- **Success:** `GPU: NVIDIA A100-SXM4-40GB | VRAM: 40.0GB`
- **Common failure → fix:** Not an A100 (or no GPU) →
  `Runtime → Change runtime type → A100 GPU`, then re-run.

### 2. MOUNT DRIVE  ← re-run this after EVERY restart/timeout
- **Does:** Mounts Google Drive and **verifies the EURUSD CSV is visible**. If
  the file isn't visible (a stale mount from a previous session), it
  automatically retries once with `force_remount=True`, then asserts the file
  exists with a message naming the exact expected path.
- **Success:** `Drive mounted. Primary data file confirmed:`
- **Common failure → fix:** `PRIMARY DATA FILE NOT FOUND` →
  1. Open the **RL-Trading-Data** folder in Drive and confirm the CSV is present
     and named **exactly** `EURUSD_M1_202101131130_202605270000_2020_2026.csv`.
  2. If it's there but still not seen, force a clean remount in a cell:
     ```python
     from google.colab import drive
     drive.mount('/content/drive', force_remount=True)
     ```
  3. Re-run the cell.

### 3. CLONE / UPDATE REPO  ← must run BEFORE install
- **Does:** Clones the repo on a fresh runtime, or hard-resets to
  `origin/master` on a re-run (discarding any local patches). `cd`s into the
  repo root.
- **Success:** `Repo cloned.` or `Repo hard-reset to origin/master.` followed by
  the latest commits.
- **Common failure → fix:** Transient network/git error → re-run the cell.
- **Why order matters:** Install (Cell 4) reads this repo's `requirements.txt`,
  so the repo directory must exist first. Installing before cloning fails on a
  fresh runtime.

### 4. INSTALL DEPENDENCIES
- **Does:** `pip install -r requirements.txt` from the cloned repo, then verifies
  `import talib`.
- **Success:** `Dependencies installed. TA-Lib <version> import OK.`
- **Common failure → fix:** `ModuleNotFoundError: talib` or a pip error → make
  sure Cell 3 ran first (the repo dir must exist), then re-run Cell 3 → Cell 4.

### 4b. SANITY IMPORT
- **Does:** Imports `core.settings`, `core.pipeline`, `training.train` so a
  broken dependency fails loudly here instead of deep in the training loop.
- **Success:** `Core modules import OK — ready to train.`
- **Common failure → fix:** Import error → re-run Cell 3 + Cell 4.

### 5. CLEAN MANIFEST
- **Does:** Removes stale checkpoint entries (e.g. deleted DQN files) from
  `gpu/manifest.json` so resume doesn't try to load a missing checkpoint.
- **Success:** `Manifest cleaned: N → M entries.`
- **Common failure → fix:** `No manifest found` is harmless — it's created fresh
  on the first training run.

### 6. SYSTEM INSPECTION
- **Does:** Runs `inspect_system.py`, a full preflight (imports, smoke train /
  backtest / infer, pytest). Aborts if any check is ❌.
- **Success:** All checks ✅ or ⚠️ SKIP, process exits 0.
- **Common failure → fix:** Any ❌ → read the printed IRAC block for that check,
  fix the root cause, re-run.

### 7. RUN TRAINING
- **Does:** Resumes from Drive checkpoints (unless none exist) and trains,
  streaming one aggregated `DAY` line per calendar day and an `Episode` summary.
- **Success:** Lines like `DAY  1 🟢 PnL $... equity ...` and
  `Episode 1 [phase] pass ...% ...` stream live.
- **Common failure → fix:**
  - `FileNotFoundError` on the CSV → **Drive unmounted — re-run Cell 2.**
  - `no checkpoint found — fresh start` → normal on a brand-new run.
- **Note:** The `--manifest` path **must live in the `gpu/` dir** (next to the
  `--checkpoint-dir` checkpoints) so `find_best_resume()` can locate them.
- **Note — `torch.compile` warmup is NOW VISIBLE (not a freeze):** The first
  rollout step compiles the model and **blocks ~10–15 min on an A100**. You used
  to see ZERO output during this and it looked frozen. It is now provably alive —
  the startup output appears **in this exact order**:
  1. `[train] === PHASE phase1_cci_align ===`
  2. `[train] 🛠  torch.compile warming up (mode=default) — first step compiles…`
     (the announcement; only when compile is ON)
  3. `  ⏱  heartbeat  step      0/…  0.0 steps/s  elapsed      0s  phase …  (loop entry)`
     (an **immediate** heartbeat, printed **and** written to
     `heartbeat_training.txt` on disk, BEFORE the blocking compile)
  4. `  ⏳ still compiling… 30s elapsed (torch.compile warmup, phase …) — NORMAL, not a crash`
     repeating every 30s **throughout** the block (the compile watchdog)
  5. `[train] ✅ torch.compile finished in 612s — training is now running fast.`
     (the compile-done marker, with how long it took)
  6. Then the normal `DAY …` / `Episode …` lines stream as usual.
  If you see steps 2–4 with no further output yet, that is **warmup, not a
  crash.** **To skip warmup entirely** (faster startup, slightly slower
  steady-state), uncheck **`USE_TORCH_COMPILE` / COMPILE_MODEL** in the Cell 7
  ⚡ GPU panel (or set `USE_TORCH_COMPILE: false` in `core/settings.py`). The
  heartbeat cadence and watchdog ticker are also configurable there
  (`HEARTBEAT_SECS`, `COMPILE_WATCHDOG_ENABLED`).

### 8–10. (optional) Dashboard / Crash recovery / GPU profiling
- **8 Dashboard:** Streamlit UI over localtunnel (ngrok fallback in the cell).
- **9 Crash recovery:** Finds the best valid checkpoint after a crash, then
  re-run Cell 7.
- **10 GPU profiling:** Profiles a forward pass; warns if GPU utilization <50%.

---

## Troubleshooting — error → cause → fix

| Error message (substring) | Real cause | Exact fix |
|---|---|---|
| `FileNotFoundError` / `PRIMARY DATA FILE NOT FOUND` on the **CSV** | **Drive not mounted** (fresh/restarted runtime), wrong path/filename, or the file isn't in the folder. **Not a code bug.** | (a) Re-run **Cell 2**. If still failing: `drive.mount('/content/drive', force_remount=True)`. (b) `!ls -la /content/drive/MyDrive/RL-Trading-Data/`. (c) Confirm the filename matches the `--csv` arg exactly. (d) Re-run Cell 6. |
| `ModuleNotFoundError: No module named 'talib'` | Install cell ran before the repo was cloned, or the install didn't complete. | Run **Cell 3** (clone) first, then re-run **Cell 4** (install). |
| `subprocess.CalledProcessError` from pip / `ResolutionImpossible` | pip dependency conflict (e.g. numpy pin) or install run from the wrong dir. | Re-run **Cell 3** then **Cell 4** from the repo root. `requirements.txt` pins are deliberate — don't loosen them blindly. |
| `inspect_system.py failed` (non-zero exit from Cell 6) | A preflight check found real code/config breakage. | Read the ❌ check's IRAC block in the output, fix that specific issue, re-run Cell 6. |
| `no checkpoint found` / no-checkpoint-to-resume | No prior checkpoints on Drive yet (or manifest was cleaned). | This is **normal** on a first run — training starts fresh and writes new checkpoints. |
| `FileNotFoundError ... eurusd_gpu` (checkpoint, not CSV) | Manifest lists a deleted DQN checkpoint. | Run **Cell 5** (clean manifest), then re-run Cell 7. |
| `No such file or directory: 'training/train.py'` / `No module named 'training'` | Working directory isn't the repo root. | Re-run **Cell 3** (it `cd`s into the repo), then Cell 7. |
| `CUDAGraphs that has been overwritten` | `torch.compile` CUDA-graph buffer reuse. | Re-run Cell 3 (reset to latest code) then Cell 7; if it persists, set `USE_TORCH_COMPILE: false` in `core/settings.py`. |
| `assert vram_gb > 30` (Cell 1) | Runtime isn't an A100. | `Runtime → Change runtime type → A100 GPU`. |

---

## For a future LLM debugging a crash

When you see a `FileNotFoundError` (or our `DataFileNotFoundError`) naming the
EURUSD CSV: **stop.** This is an environment/state problem, almost always an
unmounted Drive in a fresh or restarted Colab session. The data loader
(`core/pipeline.py::load_ohlcv_csv`) is fine — it now guards with
`os.path.exists` and raises a friendly, actionable error before pandas. **Do not
rewrite the loader or the pipeline.** The fix is operational: re-mount Drive
(Cell 2, `force_remount=True`) and `ls` the `RL-Trading-Data` folder to confirm
the file is present and correctly named.
