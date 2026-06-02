"""Pytest config: repo root importable + allow numpy-indicator fallback in tests.

talib is the production source of truth (DESIGN_DECISIONS.md #3); CI here has no
talib, so we set RL_ALLOW_NUMPY_INDICATORS=1 so indicators import via the numpy
fallback. This flag is TEST-ONLY and never set in production.
"""
import os, sys
os.environ.setdefault("RL_ALLOW_NUMPY_INDICATORS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
