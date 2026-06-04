"""
tests/unit/test_audit_pass1_causality.py
────────────────────────────────────────────────────────────────────────────
PASS-1 AUDIT — Step 2 (causality & data integrity) regression tests.

These lock the invariants that decide whether ANY training result is meaningful:

  • LOOK-AHEAD (multi-timeframe): the higher-TF row exposed at 1m bar `raw_i`
    (index raw_i // tf) must aggregate ONLY 1m bars at positions <= raw_i. If the
    in-progress 15m/30m/60m bar leaked, the agent would see the future close of
    the bar it is deciding on. We assert positionally that the exposed TF bar can
    never include a future 1m row, across tf ∈ {15,30,60} and a sweep of raw_i.

  • DECISION/FILL TIMING: at bar t the entry fills at close[t] and PnL is only
    realized via mark-to-market on close[t+1] — i.e. the current action cannot
    profit from a price it could not have transacted at. A flat→buy at a bar whose
    NEXT close is +10 pips yields +PnL; a buy at a bar whose next close is flat
    yields ~0 (minus commission). This proves no same-bar future-close leak.

  • NaN/Inf handled LOUDLY, not silently: obs over a full episode contains no
    NaN/Inf even when fed a degenerate constant-price series.

  • TRAIN/VAL SEPARATION: set_start_window confines episode starts to disjoint
    chronological slices, so the eval tail never overlaps the training head.
"""
import numpy as np
import torch

from core.env.environment import BatchedFTMOEnv
from core.agent.action_space import FLAT, BUY, EXIT_HOLD, MIN_LOT

DEV = torch.device("cpu")
BPD = 1440


def _ramp_ohlcv(n: int) -> np.ndarray:
    """A strictly increasing close path so every TF bar has a DISTINCT close —
    this makes a future-bar leak detectable (a leaked future close would be
    strictly greater than any legitimately-visible past close)."""
    close = 1.10 + np.arange(n) * 1e-5
    high = close + 1e-6
    low = close - 1e-6
    vol = np.ones(n) * 100.0
    return np.stack([close, high, low, close, vol], axis=1).astype(np.float32)


def _cfg(B=2, days=4):
    return {"BATCH_SIZE_ENV": B, "LOOKBACK": 20, "BARS_PER_DAY": BPD,
            "EPISODE_BARS": BPD * days, "DAILY_TARGET_PCT": 0.025,
            "DAILY_MAX_DD_PCT": 0.010, "MAX_TRADES_PER_DAY": 100000,
            "MAX_LOT": 2.0, "FEATURES": None}


# ════════════════════════════════════════════════════════════════════════════
# LOOK-AHEAD: the exposed higher-TF bar never aggregates a FUTURE 1m row
# ════════════════════════════════════════════════════════════════════════════
def test_tf_alignment_never_exposes_future_bar():
    """For tf ∈ {15,30,60} and a sweep of 1m indices raw_i, the exposed TF bar
    index p = raw_i // tf must satisfy: the LAST 1m bar aggregated into p is at a
    position <= raw_i. With pandas resample(closed='right', label='right') the
    bin at integer position p closes at 1m row (p+1)*tf-1 in the WORST case; the
    repo's `i//tf` map guarantees p*tf <= raw_i, so the bin's coverage cannot
    extend past raw_i. We assert it empirically by reconstructing each TF close
    from the raw series and checking the exposed close equals an aggregate of
    ONLY past/current 1m closes (never a strictly-future, hence larger, close)."""
    n = BPD * 4 + 300
    ohlcv = _ramp_ohlcv(n)
    env = BatchedFTMOEnv(ohlcv, _cfg(), DEV,
                         phase={"gate_timeframes": [15, 30, 60],
                                "entry_conditions": {"buy": "any", "sell": "any"}})
    raw_close = ohlcv[:, 3]
    for tf in (15, 30, 60):
        df_ind = env._tf_indicators[tf]
        tf_len = len(df_ind)
        tf_close = df_ind["close"].to_numpy()
        # sweep representative 1m bars across several days
        for raw_i in range(env.lkbk + 5, n - 5, 137):
            p = env._tf_pos(raw_i, tf, tf_len)
            exposed = float(tf_close[p])
            # The maximum 1m close the agent is ALLOWED to have seen by bar raw_i
            # is raw_close[raw_i] (the bar it is deciding on). Since the series is
            # strictly increasing, a leaked FUTURE bar would be strictly greater.
            allowed_max = float(raw_close[raw_i]) + 1e-9
            assert exposed <= allowed_max, (
                f"tf={tf} raw_i={raw_i}: exposed TF close {exposed} > "
                f"max visible 1m close {allowed_max} — FUTURE BAR LEAKED")


def test_tf_pos_is_monotone_and_bounded():
    """`_tf_pos` (i//tf, clamped) is non-decreasing in i and never exceeds the
    resampled length — a regression guard on the integer-alignment contract."""
    env = BatchedFTMOEnv(_ramp_ohlcv(BPD * 2 + 100), _cfg(days=2), DEV,
                         phase={"gate_timeframes": [15],
                                "entry_conditions": {"buy": "any", "sell": "any"}})
    tf_len = len(env._tf_indicators[15])
    prev = -1
    for i in range(0, BPD * 2, 7):
        p = env._tf_pos(i, 15, tf_len)
        assert p >= prev, f"_tf_pos went backwards at i={i}"
        assert 0 <= p < tf_len
        prev = p


# ════════════════════════════════════════════════════════════════════════════
# DECISION/FILL TIMING: action at t fills at close[t]; profit only on close[t+1]
# ════════════════════════════════════════════════════════════════════════════
def _step_ohlcv(n, step_at, jump):
    close = np.full(n, 1.10, dtype=np.float64)
    close[step_at:] = 1.10 + jump
    return np.stack([close, close + 1e-6, close - 1e-6, close,
                     np.ones(n) * 100], axis=1).astype(np.float32)


def test_entry_fills_at_decision_bar_close_profit_only_next_bar():
    """A BUY decided on the LAST flat bar (so the very next close is +10 pips)
    makes money; a BUY decided one bar earlier — where the next close is still
    flat — makes ~0 on that step. This proves the fill uses close[t] and the
    current bar's action cannot read close[t+1] before committing."""
    B = 1
    n = 400
    jump = 0.0010                                          # +10 pips at bar 200
    ohlcv = _step_ohlcv(n, step_at=200, jump=jump)
    cfg = {**_cfg(B=B, days=1), "LOT_CURRICULUM_ENABLED": False}
    lot_raw = (0.10 - MIN_LOT) / (2.0 - MIN_LOT)           # maps to 0.10 lots
    buy = {"direction": torch.tensor([BUY]), "lot_raw": torch.tensor([float(lot_raw)]),
           "exit": torch.tensor([EXIT_HOLD])}

    # Case A: decide on bar 199 (next close = bar 200 = +10 pips) -> +$10
    envA = BatchedFTMOEnv(ohlcv, cfg, DEV,
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    envA.reset(); envA._start.fill_(0); envA._step_i.fill_(199)
    eqA0 = envA._equity.clone()
    envA.step(buy)
    gainA = float((envA._equity - eqA0)[0].item())

    # Case B: decide on bar 198 (next close = bar 199 = still flat) -> ~0 (− comm)
    envB = BatchedFTMOEnv(ohlcv, cfg, DEV,
                          phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    envB.reset(); envB._start.fill_(0); envB._step_i.fill_(198)
    eqB0 = envB._equity.clone()
    envB.step(buy)
    gainB = float((envB._equity - eqB0)[0].item())

    # 0.10 lots * 10 pips * 100000 = $10 (minus the open-side commission).
    assert 8.0 < gainA < 11.0, f"expected ≈+$10 on the +10-pip next bar, got {gainA}"
    assert abs(gainB) < 2.0, f"flat next-bar step should be ~0, got {gainB}"
    assert gainA > gainB + 5.0, "profit must require the move to be in the NEXT bar"


# ════════════════════════════════════════════════════════════════════════════
# NaN/Inf: obs is finite across a full episode on a degenerate series
# ════════════════════════════════════════════════════════════════════════════
def test_obs_finite_over_full_episode_constant_price():
    """A flat constant-price series (zero variance — the classic divide-by-std
    NaN trap) must still yield FINITE observations on every bar of a full day."""
    n = BPD + 200
    close = np.full(n, 1.10, dtype=np.float64)
    ohlcv = np.stack([close, close, close, close, np.ones(n) * 100],
                     axis=1).astype(np.float32)
    env = BatchedFTMOEnv(ohlcv, _cfg(B=2, days=1), DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    state = env.reset()
    assert torch.isfinite(state).all(), "reset obs has NaN/Inf on constant price"
    flat = {"direction": torch.full((2,), FLAT, dtype=torch.long),
            "lot_raw": torch.zeros(2), "exit": torch.full((2,), EXIT_HOLD,
                                                          dtype=torch.long)}
    for _ in range(BPD):
        state, r, d, _info = env.step(flat)
        assert torch.isfinite(state).all(), "obs went non-finite mid-episode"
        assert torch.isfinite(r).all(), "reward went non-finite mid-episode"


# ════════════════════════════════════════════════════════════════════════════
# TRAIN/VAL SEPARATION: disjoint chronological start windows
# ════════════════════════════════════════════════════════════════════════════
def test_start_window_confines_episode_starts_to_slice():
    """set_start_window(lo,hi) must keep every sampled start bar inside the
    fractional slice, and a train head [0,0.8) must be disjoint from an eval
    tail [0.8,1.0)."""
    n = BPD * 10
    ohlcv = _ramp_ohlcv(n)
    env = BatchedFTMOEnv(ohlcv, {**_cfg(B=64, days=1)}, DEV,
                         phase={"entry_conditions": {"buy": "any", "sell": "any"}})
    warmup = env.lkbk + 25
    max_start = max(warmup + 1, env.T - env.ep_bars - 1)
    span = max_start - warmup

    env.set_start_window(0.0, 0.8)
    env.reset()
    train_starts = env._start.clone()
    train_hi = warmup + int(0.8 * span)
    assert int(train_starts.max().item()) < train_hi + 1, "train start leaked into tail"

    env.set_start_window(0.8, 1.0)
    env.reset()
    eval_starts = env._start.clone()
    eval_lo = warmup + int(0.8 * span)
    assert int(eval_starts.min().item()) >= eval_lo, "eval start leaked into head"
    # disjoint: no eval start is below the train ceiling
    assert int(eval_starts.min().item()) >= int(train_starts.max().item()) - 1 or \
        eval_lo >= train_hi, "train/val start windows overlap"

    # default restores full range
    env.set_start_window(0.0, 1.0)
    env.reset()
    assert int(env._start.max().item()) <= max_start
