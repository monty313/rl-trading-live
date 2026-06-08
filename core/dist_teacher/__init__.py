# ═══════════════════════════════════════════════════════
# [DIST PRE-PHASE] TEMPORARY PACKAGE — REMOVE AT GRADUATION
# ═══════════════════════════════════════════════════════
# This package contains the DQN→PPO direction distillation
# pre-phase machinery. It is TEMPORARY by design.
#
# >>> FULL LIFECYCLE + REVERT INSTRUCTIONS live at the TOP of
# >>> core/dist_phase/dist_phase_manager.py. Read THAT file first.
#
#
# WHEN TO REMOVE:
#   Only after dist_graduation_record.json exists AND
#   contains "dist_graduation_complete": true AND
#   shows 10 consecutive passing days on record.
#
# HOW TO REVERT (full instructions):
#   1. Delete entire core/dist_teacher/ directory
#   2. Delete entire core/dist_phase/ directory
#   3. In training/train.py: remove DistPrePhaseWrapper
#      wrapping (restore env = base_env)
#   4. In rl_trading_colab.ipynb: comment out or delete
#      the [DIST] cells (probe, init)
#   5. In core/settings.py: remove the [DIST PRE-PHASE]
#      block (search for the bookend comments)
#   6. In PPO actor-critic: if state_dim was expanded by 3,
#      reduce back to the original base obs_dim
#   7. Confirm: pytest tests/ — all original tests pass
#   8. KEEP dist_graduation_record.json — it is your proof
#
# WHAT IS NOT CHANGED AND NEEDS NO REVERT:
#   - core/env/environment.py  (untouched)
#   - core/agent/ppo.py        (untouched)
#   - training/train.py base loop (untouched, only wrap site)
#   - config/phases.yaml       (untouched)
#   - All existing tests       (untouched)
# ═══════════════════════════════════════════════════════

from core.dist_teacher.dist_dqn_teacher import DistDQNTeacher
from core.dist_teacher.dist_obs_adapter import DistObsAdapter
from core.dist_teacher.dist_prephase_wrapper import DistPrePhaseWrapper

__all__ = ["DistDQNTeacher", "DistObsAdapter", "DistPrePhaseWrapper"]
