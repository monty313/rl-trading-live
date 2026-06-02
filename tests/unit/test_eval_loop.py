"""Unit test for eval_loop output contract."""
import torch
from core.pipeline import build_pipeline
from core.settings import CFG
from training.eval_loop import run_eval
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array

DEV = torch.device("cpu")


def test_eval_returns_four_keys():
    c = dict(CFG)
    c.update({"FEATURES": make_synthetic_ohlcv_array(n=800),
              "EPISODE_BARS": 240, "BARS_PER_DAY": 60,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    env, agent, *_ = build_pipeline(c, DEV,
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    m = run_eval(env, agent, c, n_days=2)
    assert set(m) == {"pass_rate", "phi", "avg_daily_return", "avg_daily_dd"}
    assert 0.0 <= m["pass_rate"] <= 1.0
