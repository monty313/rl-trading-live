"""
scripts/smoke_backtest.py
Smoke test: run 1 day of backtest on synthetic data; assert the 6-key dict. Exit 0.
    python scripts/smoke_backtest.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import CFG, get_device, auto_tune_batch  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array  # noqa: E402

def main() -> int:
    device = get_device()
    cfg = auto_tune_batch(dict(CFG), device)
    cfg.update({"EPISODE_BARS": 240, "BARS_PER_DAY": 60,
                "USE_AMP": False, "USE_TORCH_COMPILE": False})
    out = run_backtest(None, cfg=cfg, device=device, n_days=1,
                       features=make_synthetic_ohlcv_array(n=600))
    required = {"daily_returns", "pass_fail", "phi", "total_pass_days",
                "total_fail_days", "max_drawdown_pct"}
    assert required.issubset(out.keys()), f"missing keys: {required - set(out)}"
    print("[smoke_backtest] keys OK:", sorted(out.keys()))
    print("SMOKE_BACKTEST OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
