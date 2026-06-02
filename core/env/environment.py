"""
core/env/environment.py
────────────────────────────────────────────────────────────────────────────
BatchedFTMOEnv — B parallel trading episodes stepped in lockstep on GPU tensors.
Ported from gpu_rl_trading/env/environment.py (REPO1) with these changes:

  (a) Wires core/env/intrabar_fills.compute_fill for entry/SL/TP on each trade.
  (b) Adds a (B, DIRECTION_DIM) direction mask applied to PPO logits before
      argmax. The mask is produced by conditions_engine from the active phase.
  (c) Multi-symbol: load EURUSD / GBPUSD / XAUUSD / US30 (or aligned baskets).
  (d) PASS/FAIL rule (HARD RULE 7):
        PASS  = end_balance >= initial_balance * 1.025 (regardless of DD)
        FAIL  = DD breach occurred AND end_balance < initial_balance * 1.025
        PASS_NO_BREACH = target hit AND no DD breach -> +0.01 reward bonus
  (e) Entire feature tensor preloaded to device at __init__; episodes index slices.
  (f) All tensors live on cfg["device"]; day-boundary logic is vectorized
      (no Python per-batch loop in the hot path — fixes bottleneck #1).

State layout: a normalized lookback window of the feature matrix plus 6 FTMO
position/account features (position, unrealised, equity change, gap-to-target,
dd-headroom, daily-return). state_dim is computed in __init__.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

import pandas as pd

from core.agent.action_space import (DIRECTION_DIM, FLAT, BUY, SELL, HOLD,
                                     EXIT_REDUCE, EXIT_CLOSE, map_lot)
from core.env.indicators import (build_feature_matrix, NUM_FEATURES, COL,
                                 feature_row_dict, compute_indicators,
                                 resample_ohlcv)
from core.env import conditions_engine

_NEG_INF = -1e9


class BatchedFTMOEnv:
    """Vectorized FTMO trading environment over B parallel episodes."""

    def __init__(self, features: "np.ndarray | torch.Tensor", cfg: dict,
                 device: torch.device, instrument: str = "EURUSD",
                 phase: Optional[dict] = None, policy: Optional[dict] = None):
        self.cfg = cfg
        self.device = device
        self.instrument = instrument
        self.phase = phase or {"entry_conditions": {"buy": "any", "sell": "any"}}
        self.policy = policy or {}

        self.B = int(cfg.get("BATCH_SIZE_ENV", 4))
        self.lkbk = int(cfg.get("LOOKBACK", 20))
        self.target_pct = float(cfg.get("DAILY_TARGET_PCT", 0.025))
        self.max_dd_pct = float(cfg.get("DAILY_MAX_DD_PCT", 0.010))
        self.initial_equity = float(cfg.get("INITIAL_EQUITY", 100_000.0))
        self.bars_per_day = int(cfg.get("BARS_PER_DAY", 1440))
        self.max_lot = float(cfg.get("MAX_LOT", 2.0))
        self.direction_dim = DIRECTION_DIM

        # ── Preload the feature matrix to device (built if raw OHLCV passed) ──
        feat = self._ensure_feature_matrix(features)
        self.features = feat.to(device=device, dtype=torch.float32)
        self.T, self.F = self.features.shape

        # episode length: bounded by data; default ~ a few days for dev/CPU
        self.ep_bars = min(int(cfg.get("EPISODE_BARS", 43_200)),
                           max(self.bars_per_day, self.T - self.lkbk - 2))

        self.state_dim = self.lkbk * self.F + 6
        self._alloc_episode_tensors()

        # ── Per-timeframe indicator rows for phase gating ────────────────────
        # Phase masks gate on TF pairs (e.g. [1m,15m]). We precompute the full
        # indicator DataFrame per gate timeframe from the raw 1m OHLCV so the
        # conditions engine can read named columns per bar. Built lazily from
        # the raw series passed in (or skipped when only a feature matrix exists).
        self._tf_indicators: Dict[int, pd.DataFrame] = {}
        self._raw_ohlcv = self._extract_raw_ohlcv(features)
        if self._raw_ohlcv is not None:
            self._build_tf_indicators()
            print(f"[env] TF indicators built for timeframes: {sorted(self._tf_indicators.keys())} "
                  f"({len(self._raw_ohlcv)} bars raw OHLCV)", flush=True)
        else:
            print("[env] WARNING: no raw OHLCV available — phase gate masks DISABLED. "
                  "Pass raw (N,5) OHLCV, not a prebuilt feature matrix.", flush=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ensure_feature_matrix(self, features) -> torch.Tensor:
        """Accept either a prebuilt (N,F) feature matrix or raw (N,5) OHLCV."""
        t = features if isinstance(features, torch.Tensor) else torch.as_tensor(
            np.asarray(features, dtype=np.float32))
        if t.ndim == 2 and t.shape[1] == NUM_FEATURES:
            return t                                   # already a feature matrix
        if t.ndim == 2 and t.shape[1] >= 5:            # raw OHLCV -> build features
            o, h, l, c, v = (t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4])
            return build_feature_matrix(o, h, l, c, v, self.device)
        raise ValueError(f"Unexpected features shape {tuple(t.shape)}")

    def _alloc_episode_tensors(self):
        d, B = self.device, self.B
        self._start = torch.zeros(B, dtype=torch.long, device=d)
        self._step_i = torch.zeros(B, dtype=torch.long, device=d)
        self._equity = torch.full((B,), self.initial_equity, device=d)
        self._day_start_eq = torch.full((B,), self.initial_equity, device=d)
        self._day_high_eq = torch.full((B,), self.initial_equity, device=d)
        self._position = torch.zeros(B, device=d)       # +lots buy / -lots sell / 0
        self._entry_px = torch.zeros(B, device=d)
        self._dd_breached = torch.zeros(B, dtype=torch.bool, device=d)
        self._trades_today = torch.zeros(B, dtype=torch.long, device=d)
        self._done = torch.zeros(B, dtype=torch.bool, device=d)
        # PPO/day-reward state
        self._day_halted = torch.zeros(B, dtype=torch.bool, device=d)  # DD-ended day
        self._day_passed = torch.zeros(B, dtype=torch.bool, device=d)
        self._pass_streak = torch.zeros(B, dtype=torch.long, device=d)
        self._equity_prev = torch.full((B,), self.initial_equity, device=d)
        # tracks bars where the gate was active this day (condition TRUE)
        self._gate_bars_today = torch.zeros(B, dtype=torch.long, device=d)

    def _abs_idx(self) -> torch.Tensor:
        return (self._start + self._step_i).clamp(0, self.T - 1)

    # ── multi-timeframe indicator support ────────────────────────────────────
    def _extract_raw_ohlcv(self, features):
        """Return a 1m OHLCV DataFrame from raw (N,>=5) input, or None if a
        prebuilt feature matrix was passed (no raw OHLC available).

        The DataFrame carries a plain RangeIndex (0..N-1). A DatetimeIndex is
        synthesized ONLY transiently inside _build_tf_indicators for pandas'
        resample(); TF-to-1m alignment is done by INTEGER row position
        (1m row i -> TF bar i // tf), never by timestamp searchsorted. This
        avoids the fake-timestamp drift that produced impossible base_ts values.
        """
        t = features if isinstance(features, torch.Tensor) else torch.as_tensor(
            np.asarray(features, dtype=np.float32))
        if t.ndim == 2 and t.shape[1] >= 5 and t.shape[1] != NUM_FEATURES:
            arr = t[:, :5].detach().cpu().numpy()
            return pd.DataFrame(arr, columns=["open", "high", "low", "close",
                                              "volume"])
        return None

    def _build_tf_indicators(self):
        """Build indicator DataFrames for all required timeframes by resampling
        the 1m OHLCV bars. The raw 1m data from the CSV is the single source of
        truth — all higher timeframes (15m, 30m, 60m, etc.) are derived from it.
        No separate higher-TF CSV is needed.

        pandas' resample() needs a DatetimeIndex, so we attach a synthetic 1-min
        DatetimeIndex *transiently* just for the resample step, then drop it: the
        resulting per-TF indicator frames carry a plain RangeIndex so alignment
        back to the 1m bar is done purely by integer position (see _rows_by_tf)."""
        tfs = set([1])
        gt = (self.phase or {}).get("gate_timeframes", []) or []
        tfs.update(gt)
        tfs.update([15, 30, 60])   # all phase gate TFs used across the curriculum
        # Transient DatetimeIndex (resample-only); never used for alignment.
        dt_raw = self._raw_ohlcv.copy()
        dt_raw.index = pd.date_range("2020-01-01", periods=len(dt_raw), freq="1min")
        for tf in tfs:
            try:
                df_tf = resample_ohlcv(dt_raw, tf)
                if len(df_tf) > 0:
                    ind = compute_indicators(df_tf).reset_index(drop=True)
                    self._tf_indicators[tf] = ind
            except Exception as e:
                print(f"[env] WARNING: failed to build TF{tf} indicators: {e}", flush=True)

    def _tf_pos(self, raw_i: int, tf: int, tf_len: int) -> int:
        """Integer-based TF alignment: 1m row `raw_i` belongs to the higher-TF
        bar covering it, i.e. position `raw_i // tf` (most recent completed bar
        at/under the current 1m bar). Clamped to the resampled length. No
        timestamp searchsorted — alignment is purely positional, so it can never
        drift to an impossible date."""
        pos = raw_i // max(tf, 1)
        return max(0, min(pos, tf_len - 1))

    def _rows_by_tf_batch(self, abs_idx: torch.Tensor) -> Dict[int, list]:
        """Return {tf_minutes: [feature_row_dict per episode]} aligned PER EPISODE
        to each episode's current 1m bar via integer row indexing. Length of each
        list == B. Empty dict if no raw OHLCV / TF indicators are available."""
        if not self._tf_indicators:
            return {}
        raw_idx = abs_idx.clamp(0, len(self._raw_ohlcv) - 1).detach().cpu().tolist()
        out: Dict[int, list] = {}
        for tf, df_ind in self._tf_indicators.items():
            tf_len = len(df_ind)
            recs = df_ind.to_dict("records")   # list of row dicts, indexed positionally
            out[tf] = [recs[self._tf_pos(int(ri), tf, tf_len)] for ri in raw_idx]
        return out

    def _rows_by_tf(self) -> Dict[int, dict]:
        """Episode-0 view (compat shim for callers/tests that want a single bar's
        rows). Delegates to the batched path and returns episode 0's dicts."""
        batch = self._rows_by_tf_batch(self._abs_idx())
        return {tf: rows[0] for tf, rows in batch.items()}

    # ── reset ──────────────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        warmup = self.lkbk + 25
        max_start = max(warmup + 1, self.T - self.ep_bars - 1)
        self._start = torch.randint(warmup, max(warmup + 2, max_start),
                                    (self.B,), device=self.device)
        self._step_i.zero_()
        self._equity.fill_(self.initial_equity)
        self._day_start_eq.fill_(self.initial_equity)
        self._day_high_eq.fill_(self.initial_equity)
        self._position.zero_()
        self._entry_px.zero_()
        self._dd_breached.zero_()
        self._trades_today.zero_()
        self._done.zero_()
        self._day_halted.zero_()
        self._day_passed.zero_()
        # NOTE: _pass_streak intentionally persists across episodes within a phase
        # (DESIGN_DECISIONS.md #7 — consecutive pass-days counter is phase-level).
        self._equity_prev.fill_(self.initial_equity)
        self._gate_bars_today.zero_()
        return self._get_state()

    # ── state ──────────────────────────────────────────────────────────────────
    def _get_state(self) -> torch.Tensor:
        abs_idx = self._abs_idx()
        offsets = torch.arange(self.lkbk - 1, -1, -1, device=self.device)
        win_idx = (abs_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp(0, self.T - 1)
        window = self.features[win_idx]                       # (B, lkbk, F)
        mu = window.mean(dim=(1, 2), keepdim=True)
        std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        norm = ((window - mu) / std).reshape(self.B, -1)      # (B, lkbk*F)

        close = self.features[abs_idx, COL["close"]]
        unrealised = torch.where(
            self._position != 0,
            (close - self._entry_px) * torch.sign(self._position),
            torch.zeros_like(close))
        eq_chg = (self._equity - self.initial_equity) / self.initial_equity
        target_eq = self._day_start_eq * (1.0 + self.target_pct)
        gap = (target_eq - self._equity) / (self.initial_equity + 1e-8)
        dd_used = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        dd_head = (self.max_dd_pct - dd_used).clamp(min=0.0)
        daily_ret = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8)
        pos_feat = torch.stack([self._position, unrealised, eq_chg, gap, dd_head,
                                daily_ret], dim=1)
        return torch.cat([norm, pos_feat], dim=1)

    # ── action mask (RULE 12) ──────────────────────────────────────────────────
    def current_mask_and_force(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-episode gate mask. Returns:
          dir_mask   : (B, DIRECTION_DIM) float — 1.0 allowed / 0.0 masked, computed
                       independently for EVERY one of the B episodes (each episode
                       has its own bar index and own position state).
          must_enter : (B,) bool — True where a force_in_and_gate/open_gate gate is
                       ON and the episode is flat (agent must open BUY or SELL).

        BUG-4 FIX: previously this read only self._position[0] / abs_idx[0] (episode
        0) and broadcast that single mask to all 64 episodes, so 63/64 episodes got
        the wrong mask. Now the gate condition + force-entry is evaluated per episode.
        """
        abs_idx = self._abs_idx()
        is_flat = (self._position == 0)                      # (B,) bool tensor
        rows_batch = self._rows_by_tf_batch(abs_idx)
        if not rows_batch:
            # string-condition / no-raw-OHLCV fallback: build the compact feature
            # row per episode from the feature matrix.
            rows_batch = {1: [feature_row_dict(self.features[i])
                              for i in abs_idx.detach().cpu().tolist()]}
        return conditions_engine.compute_action_mask_batch(
            self.phase, rows_batch, self.B, self.device, is_flat=is_flat)

    def current_direction_mask(self) -> torch.Tensor:
        """(B, DIRECTION_DIM) per-episode float mask for the PPO direction head."""
        mask, _ = self.current_mask_and_force()
        return mask

    def _gate_on_batch(self, abs_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B,) bool — True where the phase gate condition fires for that episode
        at bar `abs_idx` (defaults to the current bar), regardless of its position.
        Computed by masking AS-IF-FLAT: when the gate is ON the FLAT direction is
        masked, so mask[:,FLAT]==0 is the per-episode gate-on signal."""
        if abs_idx is None:
            abs_idx = self._abs_idx()
        rows_batch = self._rows_by_tf_batch(abs_idx)
        if not rows_batch:
            rows_batch = {1: [feature_row_dict(self.features[i])
                              for i in abs_idx.detach().cpu().tolist()]}
        forced_flat = torch.ones(self.B, dtype=torch.bool, device=self.device)
        mask, _ = conditions_engine.compute_action_mask_batch(
            self.phase, rows_batch, self.B, self.device, is_flat=forced_flat)
        return mask[:, FLAT] == 0.0

    # ── step ────────────────────────────────────────────────────────────────────
    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Advance one bar for all B episodes with a PPO structured action.

        actions: a dict from PPOAgent.select_actions with keys:
          direction (B,) long {FLAT,BUY,SELL}, lot_raw (B,) float in [0,1],
          exit (B,) long {HOLD,REDUCE,CLOSE}.
        Returns (next_state, reward (B,), done (B,) bool, info dict).
        """
        direction = actions["direction"].to(self.device).long()
        lot_raw = actions["lot_raw"].to(self.device).float()
        exit_a = actions.get("exit")
        exit_a = (exit_a.to(self.device).long() if exit_a is not None
                  else torch.zeros(self.B, dtype=torch.long, device=self.device))

        abs_idx = self._abs_idx()
        close = self.features[abs_idx, COL["close"]]
        nxt_idx = (abs_idx + 1).clamp(0, self.T - 1)
        next_close = self.features[nxt_idx, COL["close"]]

        # Per-episode gate state for THIS bar (drives force-entry + gate counting).
        gate_on = self._gate_on_batch(abs_idx)            # (B,) bool

        # Direction sign (+1 buy / -1 sell / 0 flat) and per-trade lot (continuous).
        dirs = torch.zeros(self.B, device=self.device)
        dirs = torch.where(direction == BUY, torch.ones_like(dirs), dirs)
        dirs = torch.where(direction == SELL, -torch.ones_like(dirs), dirs)
        # map raw [0,1] -> [MIN_LOT, max_lot] (same mapping as action_space.map_lot)
        lots = 0.01 + lot_raw.clamp(0, 1) * (self.max_lot - 0.01)
        lots = torch.where(dirs != 0, lots, torch.zeros_like(lots))

        # Once the day is halted (DD hit), no new positions open this day.
        halted = self._day_halted
        dirs = torch.where(halted, torch.zeros_like(dirs), dirs)
        lots = torch.where(halted, torch.zeros_like(lots), lots)

        # ── EXIT head wired into position management (action_space EXIT_*) ──
        # EXIT_CLOSE flattens the open position; EXIT_REDUCE halves it; EXIT_HOLD
        # leaves it. Applied to the EXISTING position before any new entry. PnL on
        # the closed/reduced fraction is realized at the current close.
        had_pos0 = self._position != 0
        price_move0 = (close - self._entry_px) * torch.sign(self._position)
        pnl_per_unit = price_move0 * self._position.abs() * 100000.0
        do_close = had_pos0 & (exit_a == EXIT_CLOSE)
        do_reduce = had_pos0 & (exit_a == EXIT_REDUCE)
        # realize PnL on the closed (100%) and reduced (50%) fractions
        exit_realized = (do_close.float() * pnl_per_unit
                         + do_reduce.float() * pnl_per_unit * 0.5)
        self._equity = self._equity + exit_realized
        # shrink/close the position per the exit action
        self._position = torch.where(do_close, torch.zeros_like(self._position),
                                     self._position)
        self._position = torch.where(do_reduce, self._position * 0.5, self._position)

        # Realize PnL on existing position when direction flips (new entry).
        had_pos = self._position != 0
        price_move = (close - self._entry_px) * torch.sign(self._position)
        realized = torch.where(had_pos, price_move * self._position.abs() * 100000.0,
                               torch.zeros_like(close))
        # open new position where a non-hold action is taken
        opening = dirs != 0
        self._trades_today = self._trades_today + opening.long()
        self._equity = self._equity + torch.where(opening, realized,
                                                  torch.zeros_like(realized))
        self._position = torch.where(opening, dirs * lots, self._position)

        # ── FORCE-ENTRY ENFORCEMENT (gate invariant, DESIGN_DECISIONS.md #2) ──
        # A trade MUST be active on EVERY bar the gate is ON. The PPO direction
        # mask enforces this when the episode is flat at the START of the bar, but
        # the agent can still close an existing position (EXIT_CLOSE) and sample
        # FLAT in the SAME bar — going flat mid-gate. The rule is "re-enter that
        # same bar", so here we re-open any episode that is gate-ON, not halted,
        # and now flat. The code never PICKS the side: we reuse the agent's own
        # sampled direction, defaulting to BUY only when it sampled FLAT.
        need_entry = gate_on & (~self._day_halted) & (self._position == 0)
        if bool(need_entry.any().item()):
            forced_dir = torch.where(direction == SELL,
                                     -torch.ones(self.B, device=self.device),
                                     torch.ones(self.B, device=self.device))
            forced_lot = 0.01 + lot_raw.clamp(0, 1) * (self.max_lot - 0.01)
            self._trades_today = self._trades_today + need_entry.long()
            self._position = torch.where(need_entry, forced_dir * forced_lot,
                                         self._position)
            opening = opening | need_entry
        self._entry_px = torch.where(opening, close, self._entry_px)

        # mark-to-market with next close
        # PnL = price_move (raw price units) * lots * contract_size (100000).
        # A 10-pip move (0.0010) on 0.10 lots = 0.0010 * 0.10 * 100000 = $10.00.
        # Leverage 1:100 affects margin requirement only, not PnL per lot.
        mtm = (next_close - self._entry_px) * torch.sign(self._position) * \
            self._position.abs() * 100_000.0
        equity_now = self._equity + torch.where(self._position != 0, mtm,
                                                torch.zeros_like(mtm))
        self._day_high_eq = torch.maximum(self._day_high_eq, equity_now)

        # ── vectorized DD + day boundary (no per-batch python loop) ──
        dd_used = (self._day_high_eq - equity_now) / (self._day_high_eq + 1e-8)
        breach_now = dd_used >= self.max_dd_pct
        self._dd_breached = self._dd_breached | breach_now

        self._step_i = self._step_i + 1
        new_day = (self._step_i % self.bars_per_day) == 0
        # Save trades_today BEFORE resetting — info must report the closing day's count
        trades_today_closing = self._trades_today.clone()
        # On a new day: roll baseline forward (vectorized)
        self._day_start_eq = torch.where(new_day, equity_now, self._day_start_eq)
        self._day_high_eq = torch.where(new_day, equity_now, self._day_high_eq)
        self._trades_today = torch.where(new_day,
                                         torch.zeros_like(self._trades_today),
                                         self._trades_today)
        # new day clears the intraday DD halt (fresh trading day, FTMO CEST)
        self._day_halted = torch.where(new_day, torch.zeros_like(self._day_halted),
                                       self._day_halted)
        self._dd_breached = torch.where(new_day, torch.zeros_like(self._dd_breached),
                                        self._dd_breached)
        self._gate_bars_today = torch.where(new_day,
                                            torch.zeros_like(self._gate_bars_today),
                                            self._gate_bars_today)

        self._equity = equity_now

        # ── INTRADAY 1% DD ENDS THE TRADING DAY (DESIGN_DECISIONS.md #5) ──
        # When today's trailing DD first hits the limit, halt trading for the
        # rest of the day (positions flattened, no new entries until next day).
        newly_halted = breach_now & (~self._day_halted)
        self._day_halted = self._day_halted | breach_now
        self._position = torch.where(self._day_halted, torch.zeros_like(self._position),
                                     self._position)

        # ── per-day reward (progressive consistency) at each day boundary ──
        daily_ret = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8)
        reward = torch.zeros_like(daily_ret)
        # small step signal = change in unrealised/realised equity this bar
        reward = reward + (equity_now - self._equity_prev) / (self.initial_equity + 1e-8)
        self._equity_prev = equity_now.clone()

        # ── count gate-active bars this step, PER EPISODE ──
        # Reuse the per-episode gate_on computed for this bar (force-entry above):
        # it is independent of the live position, so it is the correct gate-active
        # signal whether or not the episode currently holds a trade.
        self._gate_bars_today = self._gate_bars_today + gate_on.long()

        # day-end (new_day) OR newly-halted -> classify PASS/OK/FAIL for the day
        day_closed = new_day | newly_halted
        # Use the pre-reset count so day close reports the correct trade count
        traded_today = trades_today_closing > 0
        # Gate was meaningfully active today if it fired on >5% of bars
        gate_was_active = self._gate_bars_today > (self.bars_per_day // 20)
        dd_today = (self._day_high_eq - equity_now) / (self._day_high_eq + 1e-8)
        passed = traded_today & (daily_ret >= self.target_pct) & (dd_today <= self.max_dd_pct)
        failed = traded_today & ((dd_today > self.max_dd_pct) | (daily_ret < 0))
        # No-trade penalty: gate was active but agent never opened a position
        no_trade_penalty = (~traded_today) & gate_was_active
        self._day_passed = torch.where(day_closed & passed,
                                       torch.ones_like(self._day_passed), self._day_passed)
        # progressive day bonus (pass/ok/fail + streak) applied at day close
        pass_b    = float(self.cfg.get("REWARD", {}).get("pass_day_bonus",    2.0))
        ok_b      = float(self.cfg.get("REWARD", {}).get("ok_day_bonus",      0.5))
        fail_b    = float(self.cfg.get("REWARD", {}).get("fail_day_penalty",  -2.0))
        no_trade_b = float(self.cfg.get("REWARD", {}).get("no_trade_penalty", -1.0))
        streak_s  = float(self.cfg.get("REWARD", {}).get("streak_scale",      0.1))
        self._pass_streak = torch.where(day_closed & passed, self._pass_streak + 1,
                                        torch.where(day_closed & (passed | failed | no_trade_penalty),
                                                    torch.zeros_like(self._pass_streak),
                                                    self._pass_streak))
        day_reward = (passed.float() * pass_b
                      + failed.float() * fail_b
                      + no_trade_penalty.float() * no_trade_b
                      + (traded_today & ~passed & ~failed).float() * ok_b
                      + streak_s * self._pass_streak.float())
        reward = reward + day_closed.float() * day_reward

        # episode termination: reached ep_bars or trade cap
        max_trades = int(self.cfg.get("MAX_TRADES_PER_DAY", 800))
        cap_hit = self._trades_today >= max_trades
        ep_end = (self._step_i >= self.ep_bars) | cap_hit
        self._done = self._done | ep_end

        pass_no_breach = passed & (~self._dd_breached)
        info = {
            "equity": self._equity.detach(),
            "passed": passed.detach(),
            "dd_breached": self._dd_breached.detach(),
            "trades_today": trades_today_closing.detach(),
            "pass_no_breach": pass_no_breach.detach(),
            "day_closed": day_closed.detach(),
            "day_halted": self._day_halted.detach(),
            "pass_streak": self._pass_streak.detach(),
            "no_trade_penalty": no_trade_penalty.detach(),
            "gate_bars_today": self._gate_bars_today.detach(),
            "executed_direction": direction.detach(),
        }
        return self._get_state(), reward, self._done.clone(), info

    def get_status_dict(self) -> dict:
        """Scalar snapshot of batch item 0 (for dashboards/tests)."""
        return {
            "equity": float(self._equity[0].item()),
            "day_start_eq": float(self._day_start_eq[0].item()),
            "dd_breached": bool(self._dd_breached[0].item()),
            "trades_today": int(self._trades_today[0].item()),
            "step": int(self._step_i[0].item()),
        }


# ── multi-symbol loading (STEP 12) ───────────────────────────────────────────
def build_multi_symbol_env(features_by_symbol: Dict[str, "np.ndarray"], cfg: dict,
                           device: torch.device, phase: Optional[dict] = None,
                           policy: Optional[dict] = None):
    """
    Build one BatchedFTMOEnv per instrument, aligned by row count (inner-join on
    the shorter series). Returns dict[instrument -> BatchedFTMOEnv]. The trainer
    averages reward across instruments for a basket phase.
    """
    if not features_by_symbol:
        raise ValueError("features_by_symbol is empty")
    min_len = min(len(v) for v in features_by_symbol.values())
    envs = {}
    for sym, feats in features_by_symbol.items():
        envs[sym] = BatchedFTMOEnv(feats[:min_len], cfg, device, instrument=sym,
                                   phase=phase, policy=policy)
    return envs
