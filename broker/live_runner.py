"""
broker/live_runner.py
────────────────────────────────────────────────────────────────────────────
Per-bar live inference loop (runs on the Windows MT5 machine, NOT in Colab).

On each new M1 bar close:
  1. fetch latest candle (MT5 or mock)
  2. build observation via core/env/indicators (PARITY REQUIRED)
  3. agent.select_action(obs, deterministic=True)
  4. agent.select_action -> (direction, lot_raw, exit)
  5. intrabar_fills.compute_fill -> entry/sl/tp
  6. trade_gate.approve(order) -> if False, log BLOCKED, skip
  7. mt5_adapter.send_order(order) -> log result
  8. update rolling 20-trade accuracy SMA -> logs/accuracy_sma.json

Loads live_trading.pt from the checkpoint dir. Handles daily reset at midnight.
NOTE (HARD RULE 3): never connects to a real MT5 account unless the user has
explicitly enabled live mode AND provided a real adapter; default uses the mock.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

import torch

from core.agent.action_space import map_lot_curriculum, FLAT
from core.env.intrabar_fills import compute_fill

log = logging.getLogger("live_runner")


def resolve_lot_window(cfg: dict, phase_name: str, max_lot: float):
    """Resolve the Section-8 curriculum [lot_lo, lot_hi] window for `phase_name`
    EXACTLY as BatchedFTMOEnv._refresh_lot_window does, so a live runner sizes
    lots through the SAME window the checkpoint was trained under (S6 zero-drift).

    Mirrors the env logic: curriculum-off -> [0.01, max_lot]; beast/live phase ->
    [0.01, BEAST_MAX_LOT]; otherwise the per-phase window from CFG['LOT_CURRICULUM']
    (falling back to its '_default'). Kept here (not imported from the env) so the
    live machine never has to construct a BatchedFTMOEnv just to size a lot."""
    cur = cfg.get("LOT_CURRICULUM", {}) or {}
    default_win = cur.get("_default", [0.10, 0.50])
    enabled = bool(cfg.get("LOT_CURRICULUM_ENABLED", True))
    beast = (phase_name or "") in ("beast", "live_improve")
    if not enabled:
        lo, hi = 0.01, float(max_lot)
    elif beast:
        lo, hi = 0.01, float(cfg.get("BEAST_MAX_LOT", max_lot))
    else:
        win = cur.get(phase_name, default_win)
        lo, hi = float(win[0]), float(win[1])
    hi = min(hi, float(max_lot))
    lo = min(lo, hi)
    return float(lo), float(hi)


class LiveRunner:
    def __init__(self, agent, adapter, trade_gate, daily_guard, cfg, policy,
                 instrument="EURUSD", accuracy_path="logs/accuracy_sma.json",
                 heartbeat_path="logs/heartbeat_live_runner.txt",
                 phase_name="live_improve", live=False):
        self.agent = agent
        self.adapter = adapter
        self.gate = trade_gate
        self.guard = daily_guard
        self.cfg = cfg
        self.policy = policy
        self.instrument = instrument
        self.accuracy_path = accuracy_path
        self.heartbeat_path = heartbeat_path
        # S6 PARITY: the live runner sizes lots through the SAME curriculum window
        # + proportional scaler the trained checkpoint used. phase_name selects the
        # window (defaults to the live/beast phase, the widest, matching deployment).
        self.phase_name = phase_name
        # HARD RULE 3 / S9: dry-run is the DEFAULT. No real order leaves this
        # process unless `live` is explicitly True (the CLI sets it only on --live).
        # In dry-run every order is computed, gated and LOGGED but NOT transmitted.
        self.live = bool(live)
        # S9 EMERGENCY KILL-SWITCH: once engaged, no further order is ever sent
        # (independent of the guard; a hard local cutoff the operator controls).
        self._killed = False
        self._recent = deque(maxlen=20)   # rolling win/loss for accuracy SMA
        self._last_hb = 0.0
        # S9 DUPLICATE-ORDER PREVENTION: remember the last bar key we acted on so a
        # re-processed/duplicated bar can never fire a second identical order.
        self._last_order_key = None
        log.info("LiveRunner init: instrument=%s phase=%s mode=%s",
                 instrument, phase_name, "LIVE" if self.live else "DRY-RUN")

    # ── emergency kill-switch (S9) ───────────────────────────────────────────
    def engage_kill_switch(self, reason: str = "manual"):
        """Hard local cutoff: blocks ALL subsequent order transmission and halts
        the guard. Survives reset_day (unlike a DD halt)."""
        self._killed = True
        try:
            self.guard.force_halt()
        except Exception:
            pass
        log.critical("EMERGENCY KILL-SWITCH engaged: %s", reason)

    @staticmethod
    def _order_key(bar: dict, direction, lot: float) -> str:
        """Identity of an order for dedup: the bar's timestamp + direction + lot.
        Two calls on the SAME bar with the SAME decision share this key."""
        ts = bar.get("time", bar.get("timestamp", bar.get("close")))
        return f"{ts}|{direction}|{round(float(lot), 2)}"

    def step_bar(self, obs: torch.Tensor, bar: dict, max_lot: float = 2.0,
                 atr_14: float = None, mask=None) -> dict:
        """Process one closed M1 bar. Returns the order result dict."""
        # S9 EMERGENCY KILL-SWITCH: absolute first gate — nothing is computed/sent.
        if self._killed:
            self._heartbeat()
            log.warning("step_bar skipped — kill-switch engaged")
            return {"status": "KILLED"}

        direction, lot_raw, exit_act = self.agent.select_action(
            obs, deterministic=True, mask=mask)
        if direction == FLAT:
            self._heartbeat()
            return {"status": "FLAT", "direction": direction, "exit": exit_act}

        # S6 ZERO-DRIFT: identical lot mapping to training/eval. Resolve the active
        # phase's curriculum window and apply the item-6 proportional scaler (1.0 at
        # the trained baseline) — the SAME map_lot_curriculum the env uses on its hot
        # path. Previously this called map_lot(lot_raw, max_lot) over the FULL head
        # range, so a policy output trained to mean (say) 0.30 lots inside a narrow
        # window would have been sized to a far larger live lot.
        lot_lo, lot_hi = resolve_lot_window(self.cfg, self.phase_name, max_lot)
        lot_scale = self.agent.proportional_scale(
            self.guard.target_pct, self.guard.max_dd_pct)
        lot = map_lot_curriculum(lot_raw, lot_lo, lot_hi, lot_scale)
        # SL/TP handled deterministically by the risk module (DESIGN_DECISIONS #1);
        # use policy default pip buffers for the protective levels.
        sl_pips = int(self.cfg.get("DEFAULT_SL_PIPS", 20))
        tp_pips = int(self.cfg.get("DEFAULT_TP_PIPS", 30))
        fill = compute_fill(bar, direction, sl_pips, tp_pips,
                            self.instrument, self.policy, atr_14)
        order = {"symbol": self.instrument, "direction": direction, "lot": lot,
                 "entry": fill["entry"], "sl": fill["sl"], "tp": fill["tp"]}

        # HARD RULE 5: gate FIRST (daily-halt / consent). A halted guard (DD breach
        # or kill-switch force_halt) blocks the order here.
        if not self.gate.approve(order):
            self._heartbeat()
            log.info("order BLOCKED by gate: %s", order)
            return {"status": "BLOCKED", "order": order}

        # S9 DUPLICATE-ORDER PREVENTION: a re-delivered/duplicated bar with the same
        # decision must NOT fire a second identical order.
        key = self._order_key(bar, direction, lot)
        if key == self._last_order_key:
            self._heartbeat()
            log.warning("DUPLICATE order suppressed for key=%s", key)
            return {"status": "DUPLICATE", "order": order}

        # HARD RULE 3 / S9: dry-run DEFAULT — log the order but do NOT transmit.
        if not self.live:
            self._last_order_key = key
            self._heartbeat()
            log.info("[DRY-RUN] would send: %s", order)
            return {"status": "DRY_RUN", "order": order}

        result = self.adapter.send_order(order)
        # S9 RECONNECT during an open trade: a dropped link reports DISCONNECTED.
        # Attempt one reconnect + resend; a still-failed send books NO PnL.
        if result.get("status") == "DISCONNECTED":
            log.error("link dropped on send; attempting reconnect")
            if hasattr(self.adapter, "reconnect") and self.adapter.reconnect():
                log.info("reconnected; resending order")
                result = self.adapter.send_order(order)
            else:
                log.error("reconnect failed; order NOT sent (no PnL booked)")
        # Only a genuinely transmitted (FILLED/PARTIAL) order updates the dedup key.
        if result.get("status") in ("FILLED", "PARTIAL"):
            self._last_order_key = key
        log.info("order result: %s", result)
        self._heartbeat()
        return result

    # ── TRAIN/LIVE OBSERVATION PARITY (S9, float-for-float) ──────────────────
    @staticmethod
    def build_market_features(open_, high, low, close, volume, device=None):
        """Build the market-feature block EXACTLY as training does.

        Both training (BatchedFTMOEnv._get_state) and this live path call the SAME
        core.env.indicators.build_feature_matrix, so for identical candle inputs the
        produced (N, NUM_FEATURES) float32 matrix is bit-for-bit identical — there
        is no separate live feature pipeline to drift. Returned on CPU by default so
        a live machine without CUDA gets the same float32 values."""
        import torch as _t
        from core.env.indicators import build_feature_matrix
        dev = device or _t.device("cpu")
        return build_feature_matrix(open_, high, low, close, volume, dev)

    def record_trade_close(self, won: bool):
        """Call when a trade closes; updates the rolling 20-trade accuracy SMA."""
        self._recent.append(1 if won else 0)
        acc = (sum(self._recent) / len(self._recent) * 100.0) if self._recent else 0.0
        os.makedirs(os.path.dirname(self.accuracy_path) or ".", exist_ok=True)
        with open(self.accuracy_path, "w") as f:
            json.dump({"accuracy_sma": acc, "n": len(self._recent),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        return acc

    def reset_day(self):
        self.guard.reset_day()

    def _heartbeat(self):
        now = time.time()
        if now - self._last_hb >= 60:
            os.makedirs(os.path.dirname(self.heartbeat_path) or ".", exist_ok=True)
            with open(self.heartbeat_path, "w") as f:
                json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "status": "running",
                           "mode": "live" if self.live else "dry-run"}, f)
            self._last_hb = now


def main() -> int:
    """CLI entry point (S9). DRY-RUN IS THE DEFAULT (HARD RULE 3): without --live
    no order is ever transmitted. --live must be passed EXPLICITLY to arm real
    transmission, and even then a real MT5 module must be available (otherwise the
    mock is used and nothing reaches an account)."""
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="MT5 live runner (dry-run by default)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--instrument", default="EURUSD")
    ap.add_argument("--phase-name", default="live_improve")
    ap.add_argument("--live", action="store_true",
                    help="ARM real order transmission (default: dry-run, no orders sent)")
    args = ap.parse_args()

    if args.live:
        log.warning("LIVE MODE ARMED via --live — real orders WILL be transmitted")
    else:
        log.info("DRY-RUN (default) — orders are computed + logged but NOT sent. "
                 "Pass --live to transmit.")
    # NOTE: wiring of the agent/adapter/guard/gate is done by the operator's launch
    # script via AccountManager + build_pipeline; this CLI documents the --live
    # contract and the dry-run default. It intentionally does not auto-connect to a
    # broker so that merely importing/invoking it can never place an order.
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
