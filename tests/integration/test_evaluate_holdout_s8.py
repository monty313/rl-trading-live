"""
tests/integration/test_evaluate_holdout_s8.py
────────────────────────────────────────────────────────────────────────────
PASS-2 STEP 8 — honest holdout evaluation (evaluate.py). Verifies:
  • a checkpoint runs end-to-end over a holdout CSV and writes JSON + CSV that
    carry the checkpoint SHA-256;
  • the aggregates INCLUDE bad days (the flat/always-FLAT policy => zero-trade
    days => FAIL, never silently skipped);
  • always-buy / always-sell deterministic policies trade and are scored;
  • a random-seeded policy is reproducible (same seed => same metrics);
  • overfitting is made visible (train-vs-holdout gap when --train-pass-rate given).

We patch the agent's deterministic action selector to impose a fixed policy so
the test does not depend on a trained model — the env, commission, fills, tier
classification and the WHOLE aggregation path are the real ones.
"""
from __future__ import annotations

import json
import os

import torch

import evaluate as E
from core.agent.action_space import BUY, SELL, FLAT, EXIT_HOLD
from core.settings import CFG
from tests.fixtures.sample_candles import write_synthetic_csv

DEV = torch.device("cpu")


def _cfg():
    c = dict(CFG)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False,
              "BARS_PER_DAY": 60, "INITIAL_EQUITY": 10_000.0})
    return c


def _force_policy(direction_code):
    """Return a select_actions_eval replacement that always emits `direction_code`
    (and a mid lot), ignoring the network — to script always-buy/sell/flat. Patched
    onto the class, so it receives `self` as the first positional arg."""
    def _fn(self, state, mask=None, lot_scale=1.0):
        b = state.shape[0]
        return {"direction": torch.full((b,), direction_code, dtype=torch.long),
                "exit": torch.full((b,), EXIT_HOLD, dtype=torch.long),
                "lot_raw": torch.full((b,), 0.5),
                "value": torch.zeros(b)}
    return _fn


def _run(monkeypatch, tmp_path, direction_code, **kw):
    csv = write_synthetic_csv(str(tmp_path / "holdout.csv"), n=400, seed=11)
    # Build a real checkpoint from a freshly-constructed agent.
    from core.pipeline import build_pipeline
    cfg = _cfg(); cfg["FEATURES"] = None; cfg["DATA_CSV_EURUSD"] = csv
    cfg["BATCH_SIZE_ENV"] = 1
    _env, agent, *_ = build_pipeline(cfg, DEV,
                                     phase={"name": "live_improve", "mask": None,
                                            "mask_type": "none"})
    ckpt = str(tmp_path / "ck.pt")
    agent.save(ckpt, extra={"phase": "live_improve", "episode": 1})
    if direction_code is not None:
        monkeypatch.setattr("core.agent.ppo.PPOAgent.select_actions_eval",
                            _force_policy(direction_code))
    return E.evaluate(ckpt, csv, _cfg(), DEV, str(tmp_path / "out"), **kw)


def test_flat_policy_is_zero_trade_and_all_fail(monkeypatch, tmp_path):
    m = _run(monkeypatch, tmp_path, FLAT)
    assert m["days_evaluated"] > 0
    assert m["trade_count"] == 0, "flat policy should place no trades"
    assert m["zero_trade_days"] == m["days_evaluated"], \
        "every flat day must be a zero-trade day"
    assert m["pass_rate"] == 0.0, "a zero-trade holdout must be a 0% pass (FAIL)"
    # the bad days were INCLUDED in the aggregate, not skipped.
    assert m["fails"] == m["days_evaluated"]


def test_always_buy_trades_and_is_scored(monkeypatch, tmp_path):
    m = _run(monkeypatch, tmp_path, BUY)
    assert m["trade_count"] > 0, "always-buy must open trades"
    assert m["days_evaluated"] > 0
    assert 0.0 <= m["pass_rate"] <= 1.0
    assert m["avg_lot"] > 0.0


def test_always_sell_trades_and_is_scored(monkeypatch, tmp_path):
    m = _run(monkeypatch, tmp_path, SELL)
    assert m["trade_count"] > 0, "always-sell must open trades"
    assert m["invalid_action_count"] == 0


def test_outputs_written_with_checkpoint_hash(monkeypatch, tmp_path):
    m = _run(monkeypatch, tmp_path, BUY)
    assert os.path.exists(m["_json_path"]) and os.path.exists(m["_csv_path"])
    blob = json.load(open(m["_json_path"]))
    assert "metrics" in blob and "days" in blob
    assert len(blob["metrics"]["checkpoint_sha256"]) == 64
    # CSV has one row per evaluated day (plus header).
    lines = open(m["_csv_path"]).read().strip().splitlines()
    assert len(lines) - 1 == m["days_evaluated"]


def test_random_seeded_eval_is_reproducible(monkeypatch, tmp_path):
    # No forced policy: the real (untrained) net drives actions; the seed must make
    # two evaluations of the SAME checkpoint produce identical headline metrics.
    csv = write_synthetic_csv(str(tmp_path / "h.csv"), n=400, seed=5)
    from core.pipeline import build_pipeline
    cfg = _cfg(); cfg["DATA_CSV_EURUSD"] = csv; cfg["BATCH_SIZE_ENV"] = 1
    _env, agent, *_ = build_pipeline(cfg, DEV,
                                     phase={"name": "live_improve", "mask": None,
                                            "mask_type": "none"})
    ckpt = str(tmp_path / "ck.pt")
    agent.save(ckpt, extra={"phase": "live_improve", "episode": 1})
    m1 = E.evaluate(ckpt, csv, _cfg(), DEV, str(tmp_path / "o1"), seed=7)
    m2 = E.evaluate(ckpt, csv, _cfg(), DEV, str(tmp_path / "o2"), seed=7)
    for k in ("pass_rate", "avg_daily_return", "trade_count", "net_pnl"):
        assert m1[k] == m2[k], f"seeded eval not reproducible on {k}"


def test_overfit_gap_reported(monkeypatch, tmp_path):
    m = _run(monkeypatch, tmp_path, FLAT, train_pass_rate=0.80)
    assert "overfit_gap" in m
    # flat holdout pass rate is 0 => gap == train pass rate.
    assert abs(m["overfit_gap"] - 0.80) < 1e-9
