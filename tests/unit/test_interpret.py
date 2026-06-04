"""
tests/unit/test_interpret.py
────────────────────────────────────────────────────────────────────────────
PART 7 tests for the interpretability + dashboard suite. Cover, with a TINY net
(no GPU, no real data, fast):

  • dashboard_utils: build_params nesting, coerce/clamp, sanitize_label,
    params_hash stability, diff_from_defaults, params_to_cli contract, and the
    SOURCE-OF-TRUTH sync of widget defaults against the live settings.py / shaper.
  • results_writer: field set, params_hash match, run_index append (no overwrite),
    interrupt-partial metrics persistence.
  • action_logger: probabilities sum to ~1, CSV header/format + market-state cols.
  • saliency: non-zero grads, ranking length == feature count, runs on CPU.
  • shap_explain: graceful skip when shap absent; wrappers + cache when present.
  • policy_report: all sections present, valid stats, writes .txt + .json.
"""
import csv
import json
import os

import numpy as np
import torch

from core.settings import CFG, auto_tune_batch
from core.agent.ppo import PPOAgent
from core.agent.action_space import DIRECTION_DIM, EXIT_DIM

DEV = torch.device("cpu")
STATE_DIM = 20 + 20 * 3      # lkbk? -> use a width that yields a clean feature map


def _cfg():
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False})
    return c


def _tiny_agent(state_dim=STATE_DIM):
    return PPOAgent(state_dim, _cfg(), DEV)


def _tiny_checkpoint(tmp_path, state_dim=STATE_DIM, extra=None):
    agent = _tiny_agent(state_dim)
    p = str(tmp_path / "ckpt.pt")
    payload = {"cfg": _cfg()}
    if extra:
        payload.update(extra)
    agent.save(p, extra=payload)
    return p, agent


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  dashboard_utils                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_build_params_nests_reward():
    from core.interpret.dashboard_utils import (build_params, default_params,
                                                reward_keys)
    params = build_params(default_params())
    assert "REWARD" in params and isinstance(params["REWARD"], dict)
    for k in reward_keys():
        assert k in params["REWARD"]
        assert k not in params           # reward keys are NOT top-level
    # A top-level (non-reward) key stays top-level.
    assert "DAILY_TARGET_PCT" in params


def test_build_params_fills_missing_with_defaults():
    from core.interpret.dashboard_utils import build_params, default_params
    out = build_params({})               # empty values -> all defaults
    defs = default_params()
    flat = dict(out)
    flat.update(flat.pop("REWARD", {}))
    for k, v in defs.items():
        assert flat[k] == v


def test_coerce_and_clamp_types_and_ranges():
    from core.interpret.dashboard_utils import coerce_and_clamp
    # float clamp high
    v, clamped, skipped = coerce_and_clamp("DAILY_TARGET_PCT", 99.0)
    assert skipped is False and clamped is True and v == 0.10
    # int coercion + clamp low
    v, clamped, skipped = coerce_and_clamp("PHASE_ADVANCE_STREAK", 0)
    assert v == 1 and clamped is True and isinstance(v, int)
    # dropdown not in options -> skipped
    v, clamped, skipped = coerce_and_clamp("ACCOUNT_SIZE", 12345.0)
    assert skipped is True
    # dropdown in options
    v, clamped, skipped = coerce_and_clamp("ACCOUNT_SIZE", 25000.0)
    assert v == 25000.0 and skipped is False
    # checkbox -> bool
    v, clamped, skipped = coerce_and_clamp("BEAST_MODE", 1)
    assert v is True and skipped is False
    # unknown key -> skipped
    _, _, skipped = coerce_and_clamp("NOPE", 1)
    assert skipped is True


def test_apply_checkpoint_cfg_unpacks_reward_and_reports():
    from core.interpret.dashboard_utils import apply_checkpoint_cfg
    cfg = {"DAILY_TARGET_PCT": 99.0,            # clamp
           "ACCOUNT_SIZE": 50000.0,             # ok dropdown
           "REWARD": {"pass_day_bonus": 3.0},   # nested reward applied flat
           "TOTALLY_UNKNOWN": 1}                # skipped
    res = apply_checkpoint_cfg(cfg)
    assert res["applied"]["ACCOUNT_SIZE"] == 50000.0
    assert res["applied"]["pass_day_bonus"] == 3.0
    assert "DAILY_TARGET_PCT" in res["clamped"]
    assert any(k == "TOTALLY_UNKNOWN" for k, _ in res["skipped"])


def test_sanitize_label():
    from core.interpret.dashboard_utils import sanitize_label
    assert sanitize_label("my run 1") == "my_run_1"
    assert sanitize_label("a/b\\c:d") == "abcd"
    assert sanitize_label("***") == "unnamed"
    assert sanitize_label("") == "unnamed"
    assert sanitize_label("  spaced  ") == "spaced"


def test_params_hash_stable_and_order_independent():
    from core.interpret.dashboard_utils import params_hash
    a = {"x": 1, "y": 2, "REWARD": {"b": 1, "a": 2}}
    b = {"REWARD": {"a": 2, "b": 1}, "y": 2, "x": 1}
    assert params_hash(a) == params_hash(b)
    assert len(params_hash(a)) == 8


def test_diff_from_defaults():
    from core.interpret.dashboard_utils import (build_params, default_params,
                                                diff_from_defaults)
    defs = default_params()
    vals = dict(defs)
    vals["MAX_LOT"] = defs["MAX_LOT"] + 1.0
    vals["pass_day_bonus"] = defs["pass_day_bonus"] + 1.0   # reward key
    params = build_params(vals)
    diff = diff_from_defaults(params, defs)
    assert set(diff.keys()) == {"MAX_LOT", "pass_day_bonus"}
    assert diff["MAX_LOT"]["saved"] == defs["MAX_LOT"] + 1.0
    assert diff_from_defaults(build_params(defs), defs) == {}


def test_params_to_cli_contract():
    from core.interpret.dashboard_utils import (build_params, default_params,
                                                params_to_cli)
    defs = default_params()
    # Untouched dashboard -> valued flags None (omitted), store_true False.
    cli = params_to_cli(build_params(defs))
    assert cli["account-size"] is None
    assert cli["target-pct"] is None
    assert cli["randomize-ftmo"] is False
    # Change a few -> they appear.
    vals = dict(defs)
    vals["ACCOUNT_SIZE"] = 50000.0
    vals["DAILY_TARGET_PCT"] = 0.03
    vals["RANDOMIZE_FTMO_INPUTS"] = True
    cli = params_to_cli(build_params(vals))
    assert cli["account-size"] == 50000.0
    assert abs(cli["target-pct"] - 0.03) < 1e-9
    assert cli["randomize-ftmo"] is True


def test_widget_defaults_synced_to_code():
    """SOURCE OF TRUTH = THE CODE: every reward widget default must equal the live
    CFG['REWARD'] value, and every top-level widget default that exists in CFG must
    match. (PPO/GPU-nested keys not present at the flat CFG level are exempt — they
    live under CFG['PPO'] / module constants and are verified separately.)"""
    from core.interpret.dashboard_utils import widget_specs
    rw = CFG.get("REWARD", {})
    exempt = {"PPO_EPOCHS", "GAE_LAMBDA", "CLIP_EPS", "GPU_UTIL_TARGET",
              "START_PHASE"}
    for key, spec in widget_specs().items():
        if spec.get("reward"):
            assert key in rw, f"reward key {key} missing from CFG['REWARD']"
            assert rw[key] == spec["default"], f"reward default drift: {key}"
        elif key in CFG:
            assert CFG[key] == spec["default"], f"top-level default drift: {key}"
        else:
            assert key in exempt, f"phantom dashboard key not in CFG: {key}"


def test_obs_feature_names_length_and_labels():
    from core.interpret.dashboard_utils import obs_feature_names
    lkbk, n_ind = 20, 3
    names = obs_feature_names(lkbk, n_ind, ["cci", "rsi", "atr"])
    assert len(names) == lkbk * n_ind + 20
    assert names[0] == "cci@t-19"        # oldest lag first
    assert "session_code" in names
    assert "ftmo_target_pct" in names


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  results_writer (PART 1)                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _write_snapshot(snap_dir, params, label="t"):
    from core.interpret.dashboard_utils import params_hash
    os.makedirs(snap_dir, exist_ok=True)
    h = params_hash(params)
    path = os.path.join(snap_dir, f"params_snapshot_20260101_000000_{label}.json")
    with open(path, "w") as f:
        json.dump({"snapshot_meta": {"timestamp": "2026-01-01T00:00:00",
                                     "params_hash": h},
                   "params": params}, f)
    return path, h


def test_results_writer_matches_and_writes_fields(tmp_path):
    from core.interpret.results_writer import record_training_results
    from core.interpret.dashboard_utils import build_params, default_params
    snap_dir = str(tmp_path / "snaps")
    params = build_params(default_params())
    path, _ = _write_snapshot(snap_dir, params)
    metrics = {"pass_rate": 0.8, "best_phi": 1.23, "episodes_trained": 50,
               "final_equity": 10500.0, "best_streak": 7,
               "dd_efficiency_avg": 0.4, "phase_reached": "ph2"}
    out = record_training_results({}, params, metrics, snapshot_dir=snap_dir)
    assert out == path
    blob = json.load(open(path))
    assert len(blob["results"]) == 1
    block = blob["results"][0]
    for f in ("pass_rate", "best_phi", "episodes_trained", "final_equity",
              "best_streak", "dd_efficiency_avg", "phase_reached",
              "timestamp_completed"):
        assert f in block
    assert block["run_index"] == 0
    assert block["pass_rate"] == 0.8


def test_results_writer_appends_run_index_no_overwrite(tmp_path):
    from core.interpret.results_writer import record_training_results
    from core.interpret.dashboard_utils import build_params, default_params
    snap_dir = str(tmp_path / "snaps")
    params = build_params(default_params())
    _write_snapshot(snap_dir, params)
    m = {"pass_rate": 0.5, "best_phi": 0.1, "episodes_trained": 1,
         "final_equity": 1.0, "best_streak": 1, "dd_efficiency_avg": 0.0,
         "phase_reached": "ph0"}
    p1 = record_training_results({}, params, m, snapshot_dir=snap_dir)
    p2 = record_training_results({}, params, dict(m, pass_rate=0.9),
                                 snapshot_dir=snap_dir)
    assert p1 == p2
    blob = json.load(open(p1))
    assert [b["run_index"] for b in blob["results"]] == [0, 1]
    assert blob["results"][0]["pass_rate"] == 0.5      # first run NOT overwritten
    assert blob["results"][1]["pass_rate"] == 0.9


def test_results_writer_partial_on_interrupt(tmp_path):
    from core.interpret.results_writer import record_training_results
    from core.interpret.dashboard_utils import build_params, default_params
    snap_dir = str(tmp_path / "snaps")
    params = build_params(default_params())
    _write_snapshot(snap_dir, params)
    partial = {"pass_rate": 0.0, "best_phi": 0.0, "episodes_trained": 3,
               "final_equity": 9999.0, "best_streak": 0, "dd_efficiency_avg": 0.0,
               "phase_reached": "ph0", "interrupted": True}
    path = record_training_results({}, params, partial, snapshot_dir=snap_dir)
    block = json.load(open(path))["results"][0]
    assert block["interrupted"] is True
    assert block["episodes_trained"] == 3


def test_results_writer_no_match_returns_none(tmp_path):
    from core.interpret.results_writer import record_training_results
    snap_dir = str(tmp_path / "snaps")
    os.makedirs(snap_dir, exist_ok=True)
    out = record_training_results({}, {"x": 1}, {"pass_rate": 1.0},
                                  snapshot_dir=snap_dir)
    assert out is None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  action_logger (PART 3)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_action_distribution_probs_sum_to_one():
    from core.interpret.action_logger import action_distribution
    B = 16
    dist = action_distribution(torch.randn(B, DIRECTION_DIM),
                               torch.randn(B, EXIT_DIM),
                               torch.rand(B))
    assert abs(dist["dir_BUY"] + dist["dir_SELL"] + dist["dir_FLAT"] - 1.0) < 1e-5
    assert abs(dist["exit_HOLD"] + dist["exit_REDUCE"] + dist["exit_CLOSE"]
               - 1.0) < 1e-5
    assert dist["lot_std"] >= 0.0


def test_action_logger_csv_header_and_market_cols(tmp_path):
    from core.interpret.action_logger import (action_distribution, append_row,
                                              CSV_HEADER)
    csv_path = str(tmp_path / "metrics" / "action_distributions.csv")
    dist = action_distribution(torch.randn(8, DIRECTION_DIM),
                               torch.randn(8, EXIT_DIM), torch.rand(8))
    append_row(csv_path, dist, bar_index=100, cest_time="12:00:00",
               equity=10250.5, streak=3, dd_budget_remaining=0.0075)
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert "equity" in CSV_HEADER and "dd_budget_remaining" in CSV_HEADER
    assert "cest_time" in CSV_HEADER and "streak" in CSV_HEADER
    assert rows[1][0] == "100"           # bar_index
    assert rows[1][1] == "12:00:00"      # cest_time


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  saliency (PART 4)                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_saliency_nonzero_and_ranking_length():
    from core.interpret.saliency import compute_saliency
    agent = _tiny_agent()
    obs = torch.randn(64, STATE_DIM)
    sal = compute_saliency(agent.net, obs, top_k=5)
    assert set(sal.keys()) == {"direction", "exit", "lot"}
    for head, res in sal.items():
        assert len(res["importances"]) == STATE_DIM
        assert float(np.sum(np.abs(res["importances"]))) > 0.0   # non-zero grads
        assert len(res["ranking"]) == 5


def test_saliency_from_checkpoint_cpu(tmp_path):
    from core.interpret.saliency import saliency_from_checkpoint
    p, _ = _tiny_checkpoint(tmp_path)
    obs = torch.randn(32, STATE_DIM)
    sal = saliency_from_checkpoint(p, obs, _cfg(), device=DEV)
    assert len(sal["direction"]["importances"]) == STATE_DIM


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  shap_explain (PART 2)                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_shap_available_is_bool_and_guard():
    from core.interpret import shap_explain
    avail = shap_explain.shap_available()
    assert isinstance(avail, bool)
    if not avail:
        # explain_heads must raise a clear RuntimeError when shap is missing.
        agent = _tiny_agent()
        try:
            shap_explain.explain_heads(agent.net, torch.randn(8, STATE_DIM),
                                       torch.randn(4, STATE_DIM))
            raise AssertionError("expected RuntimeError when shap missing")
        except RuntimeError:
            pass


def test_shap_wrappers_single_output():
    from core.interpret.shap_explain import (DirectionHeadWrapper,
                                            ExitHeadWrapper, LotHeadWrapper)
    agent = _tiny_agent()
    x = torch.randn(5, STATE_DIM)
    assert DirectionHeadWrapper(agent.net)(x).shape == (5, DIRECTION_DIM)
    assert ExitHeadWrapper(agent.net)(x).shape == (5, EXIT_DIM)
    assert LotHeadWrapper(agent.net)(x).shape[0] == 5


def test_shap_file_hash_stable(tmp_path):
    from core.interpret.shap_explain import file_hash
    p, _ = _tiny_checkpoint(tmp_path)
    assert file_hash(p) == file_hash(p)
    assert len(file_hash(p)) == 12


def test_shap_run_and_cache_if_available(tmp_path):
    import pytest
    from core.interpret import shap_explain
    if not shap_explain.shap_available():
        pytest.skip("shap not installed — guarded path tested elsewhere")
    p, _ = _tiny_checkpoint(tmp_path)
    metrics_dir = str(tmp_path / "metrics")
    bg = torch.randn(32, STATE_DIM)
    ex = torch.randn(16, STATE_DIM)
    res = shap_explain.run_shap(p, bg, ex, _cfg(), metrics_dir, device=DEV)
    for head in ("direction", "exit", "lot"):
        assert len(res[head]["importances"]) == STATE_DIM
        assert res[head]["cached"] is False
    # second call hits the cache
    res2 = shap_explain.run_shap(p, bg, ex, _cfg(), metrics_dir, device=DEV)
    assert res2["direction"]["cached"] is True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  policy_report (PART 5)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_policy_report_sections_and_files(tmp_path):
    from core.interpret.policy_report import generate_policy_report
    p, _ = _tiny_checkpoint(tmp_path)
    obs = torch.randn(64, STATE_DIM)
    metrics_dir = str(tmp_path / "metrics")
    daily_log = [
        {"end_equity": 10250.0, "day_start_equity": 10000.0,
         "daily_increment": 250.0, "dd_breached": False, "traded": True,
         "dd_used_pct": 0.3},
        {"end_equity": 9900.0, "day_start_equity": 10000.0,
         "daily_increment": 250.0, "dd_breached": False, "traded": True,
         "dd_used_pct": 0.6},
    ]
    out = generate_policy_report(p, obs, _cfg(), metrics_dir,
                                 daily_log=daily_log, device=DEV)
    rep = out["report"]
    for section in ("trading_personality", "session_behavior",
                    "feature_reliance", "consistency"):
        assert section in rep
    # personality direction prefs sum to ~1
    dp = rep["trading_personality"]["direction_pref"]
    assert abs(sum(dp.values()) - 1.0) < 1e-5
    # files written
    assert os.path.exists(out["paths"]["txt"])
    assert os.path.exists(out["paths"]["json"])
    assert "POLICY SUMMARY REPORT" in out["text"]
    # consistency tier rates present
    tr = rep["consistency"]["tier_rates"]
    for tier in ("FAIL", "OK", "PASS", "EXCEED", "SURVIVAL"):
        assert tier in tr
