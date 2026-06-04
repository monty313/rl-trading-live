"""
core/seeding.py
────────────────────────────────────────────────────────────────────────────
Single source of truth for RUN reproducibility (PASS-2 S7). One call seeds
Python's `random`, NumPy, and Torch (CPU + all CUDA devices) from one integer
so a fixed --seed makes two runs with identical data/config produce identical
rollouts. Kept tiny and dependency-free so train.py, evaluate.py, and tests can
all share it (zero drift in how a "seed" is interpreted).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> int:
    """Seed random / numpy / torch (CPU + CUDA) from one integer. Returns the
    seed used. When `deterministic` is True, also requests cuDNN determinism
    (slower, but bit-reproducible on GPU); off by default so training keeps the
    fast non-deterministic kernels unless a test/eval explicitly wants exactness."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
