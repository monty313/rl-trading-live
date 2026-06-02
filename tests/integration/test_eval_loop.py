"""Integration: eval loop returns valid metrics dict."""
import torch
from core.settings import CFG, auto_tune_batch
from core.pipeline import build_pipeline
from training.eval_loop import run_eval
from tests.fixtures.sample_candles import make_synthetic_ohlcv_array


def test_eval_loop_integration():
    c = auto_tune_batch(dict(CFG), torch.device("cpu"))
    c.update({"FEATURES": make_synthetic_ohlcv_array(n=800),
              "EPISODE_BARS": 240, "BARS_PER_DAY": 60,
              "USE_AMP": False, "USE_TORCH_COMPILE": False})
    env, agent, *_ = build_pipeline(c, torch.device("cpu"),
        phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    m = run_eval(env, agent, c, n_days=2)
    assert set(m) == {"pass_rate", "phi", "avg_daily_return", "avg_daily_dd"}
