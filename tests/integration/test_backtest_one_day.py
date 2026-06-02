"""Integration: one day backtest returns the 6-key dict."""
import torch
from core.settings import CFG, auto_tune_batch
from backtest.engine import run_backtest
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array


def test_backtest_one_day():
    c = auto_tune_batch(dict(CFG), torch.device("cpu"))
    c.update({"EPISODE_BARS": 120, "BARS_PER_DAY": 60,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    out = run_backtest(None, cfg=c, device=torch.device("cpu"), n_days=1,
                       features=make_synthetic_ohlcv_array(n=400))
    assert {"daily_returns", "pass_fail", "phi", "total_pass_days",
            "total_fail_days", "max_drawdown_pct"}.issubset(out)
