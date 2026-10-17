# rl-trading-live — IRAC handoff (2026-10-17)

## Issue
A100 PPO forex system (direction/lot/exits), FTMO+Beast risk, MT5 live runner, Jordan read-only coach dashboard.
Central question: No evidence of passing; last work fixing NaN crashes.

## Rule
README, SPEC_STRATEGY.md, DESIGN_DECISIONS.md, INTERFACES.md, AUDIT.md. Doctrine: scalping-only; 5m CCI 30/100 burst entry; 1R=0.25% equity; max 4 concurrent; daily floor -3R; pyramid winners OK, add-to-losers banned.

## Application
47 commits, last 080683f 2026-06-04. e9e22f6 'self-healing PPO (stop NaN-logits crash)'. 8817af8 reward refine. c5cbf31 audit PASS-2. Data on Drive (not verifiable). Tests in tests/ not run.

## Conclusion
Status: abandoned. Stale since 2026-06-04. Blockers: training stability, no results.
Next: 1) Archive or run inspect_system.py + one eval to get a baseline.
