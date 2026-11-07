# rl-trading-live

## Project mission
Develop and operate a trading/market intelligence system focused on rl trading live.

## What this repository currently contains
Top-level structure:
- `.env.example` (file)
- `.gitignore` (file)
- `AUDIT.md` (file)
- `backtest` (dir)
- `broker` (dir)
- `config` (dir)
- `core` (dir)
- `dashboard` (dir)
- `DESIGN_DECISIONS.md` (file)
- `docs` (dir)
- `evaluate.py` (file)
- `HANDOFF_IRAC.md` (file)
- `inspect_system.py` (file)
- `INTERFACES.md` (file)
- `jordan` (dir)
- `legacy` (dir)
- `monitoring` (dir)
- `README.md` (file)
- `requirements.txt` (file)
- `rl_trading_colab.ipynb` (file)

Primary languages/tooling detected: Python (133), Markdown (8), YAML (4), Jupyter Notebook (1), Shell (1)

Likely key implementation files:
- `README.md`

## Quickstart (operator + LLM friendly)
Use this sequence first, then adapt to repo-specific details.

1. Clone and enter repo:
   ```bash
   git clone https://github.com/monty313/rl-trading-live.git
   cd rl-trading-live
   ```
2. Setup environment/dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   python main.py  # adjust to project entrypoint
   ```
4. Validate changes:
   ```bash
   pytest -q
   ```

## Operating intent (CEO-mode guide for future LLMs)
When continuing this project, optimize for business outcome first:

- Preserve or improve strategy quality, reliability, and reproducibility.
- Prefer small, testable, reversible changes over broad rewrites.
- Keep docs synchronized with code behavior.
- Add explicit assumptions, constraints, and success criteria in every major change.

## Delivery checklist
- [ ] Environment setup is deterministic.
- [ ] Primary run path is documented and works.
- [ ] Validation/test path is documented and works.
- [ ] Key risks and TODOs are tracked in the handoff file.

## Branch and release notes
- Default branch: `master`
- Keep PRs focused and attach proof-of-run or proof-of-test for behavior changes.
