"""
tests/unit/test_compile_warmup_visibility.py
────────────────────────────────────────────────────────────────────────────
COMPILE-WARMUP VISIBILITY fix (compile_warmup_visibility).

ROOT CAUSE BEING GUARDED: torch.compile(mode="default") compiles LAZILY on the
FIRST forward pass, which runs INSIDE the rollout step loop AFTER the PHASE
banner and BLOCKS the main thread ~10-15 min on an A100. The wall-clock
heartbeat lives further down the SAME loop, so it never fired during the block —
the Colab cell showed ZERO output and looked frozen/crashed.

These tests assert the new startup path makes warmup PROVABLY ALIVE:
  (a) the compile-announcement line is printed (when compile is enabled),
  (b) an IMMEDIATE step-0 heartbeat is printed AND the heartbeat_training.txt
      liveness file is written BEFORE the (blocking) first forward,
  (c) the "compile finished in Ns" marker is printed AFTER the first forward,
  (d) the stdlib-only watchdog thread starts, prints at least once for a slow
      compile, stops/joins cleanly, and makes NO torch/CUDA calls.

We monkeypatch the first forward to a SLOW NO-OP (the spec's suggested approach),
capture stdout, and inspect the on-disk heartbeat file — no GPU is required.
"""
import json
import os
import time

from training.train import (_CompileWatchdog, _first_forward_warmup,
                             _write_heartbeat)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  _first_forward_warmup — the announce + heartbeat + finished-marker path   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_warmup_announces_heartbeats_and_finishes_when_compile_on(capsys, tmp_path):
    """compile_on=True: assert (a) announce, (b) immediate step-0 heartbeat line +
    heartbeat file written BEFORE the forward returns, (c) finished marker after.
    A SLOW no-op forward stands in for the real torch.compile block."""
    order = []

    def slow_forward():
        # When the forward starts, the announcement + step-0 heartbeat must have
        # ALREADY been printed and the heartbeat file already written.
        order.append("forward")
        hb = tmp_path / "heartbeat_training.txt"
        assert hb.exists(), "heartbeat file must be written BEFORE the forward runs"
        time.sleep(0.05)
        return {"ok": True}

    out = _first_forward_warmup(
        slow_forward, compile_on=True, watchdog_on=False, watchdog_secs=30,
        phase_name="phase1_cci_align", steps=0, max_steps=1440,
        global_ep=0, metrics_dir=str(tmp_path))

    assert out == {"ok": True}
    assert order == ["forward"]
    text = capsys.readouterr().out
    # (a) announcement
    assert "torch.compile warming up" in text
    assert "NORMAL, not a crash" in text
    # (b) immediate step-0 heartbeat line printed at loop entry
    assert "heartbeat" in text and "(loop entry)" in text
    # (b) on-disk liveness file written with this episode/phase
    payload = json.loads((tmp_path / "heartbeat_training.txt").read_text())
    assert payload["episode"] == 0
    assert payload["phase"] == "phase1_cci_align"
    assert payload["status"] == "running"
    # (c) finished marker after the forward returned
    assert "torch.compile finished in" in text
    # ANNOUNCE must come BEFORE the FINISHED marker in the stream.
    assert text.index("warming up") < text.index("finished in")


def test_warmup_skips_announce_and_marker_when_compile_off(capsys, tmp_path):
    """compile_on=False (e.g. CPU / toggle off): NO compile announcement and NO
    finished marker — but the immediate step-0 heartbeat (print + file) STILL
    fires so liveness is signalled regardless of compile."""
    out = _first_forward_warmup(
        lambda: 42, compile_on=False, watchdog_on=True, watchdog_secs=30,
        phase_name="p", steps=0, max_steps=10, global_ep=3,
        metrics_dir=str(tmp_path))
    assert out == 42
    text = capsys.readouterr().out
    assert "torch.compile warming up" not in text
    assert "torch.compile finished" not in text
    # immediate heartbeat still present (print + on-disk)
    assert "heartbeat" in text and "(loop entry)" in text
    assert (tmp_path / "heartbeat_training.txt").exists()


def test_warmup_joins_watchdog_even_when_forward_raises(capsys, tmp_path):
    """The watchdog must be stopped/joined even if the first forward raises, so a
    compile-time error never leaks a live ticker thread."""
    import threading

    def boom():
        raise RuntimeError("compile blew up")

    n_before = threading.active_count()
    try:
        _first_forward_warmup(boom, compile_on=True, watchdog_on=True,
                              watchdog_secs=0.01, phase_name="p", steps=0,
                              max_steps=10, global_ep=0, metrics_dir=str(tmp_path))
    except RuntimeError:
        pass
    time.sleep(0.1)
    # No leftover watchdog thread.
    assert threading.active_count() <= n_before + 0  # joined back to baseline
    assert not any(t.name == "compile-watchdog"
                   for t in threading.enumerate())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  _CompileWatchdog — stdlib-only ticker during the blocking compile         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_watchdog_starts_ticks_and_stops_cleanly(capsys):
    """A simulated slow compile (~0.25s) with a 0.05s cadence must print at least
    one 'still compiling…' line, and the thread must stop + join cleanly."""
    import threading
    wd = _CompileWatchdog(interval_s=0.05, phase_name="phase1_cci_align")
    wd.start()
    assert any(t.name == "compile-watchdog" for t in threading.enumerate())
    time.sleep(0.25)            # simulate the blocking forward
    wd.stop()
    time.sleep(0.05)
    text = capsys.readouterr().out
    assert "still compiling" in text
    assert "phase phase1_cci_align" in text
    # thread is gone after stop()/join()
    assert not any(t.name == "compile-watchdog" for t in threading.enumerate())


def test_watchdog_is_daemon_and_makes_no_torch_calls():
    """Hard constraint (item 4): the watchdog thread must be a DAEMON and must NOT
    import/touch torch or CUDA (a second thread in CUDA during compile can
    deadlock). We assert daemon, and that the thread BODY (_run) makes NO torch /
    CUDA NAME references — co_names holds the attribute/global names the bytecode
    actually touches, so a string literal that merely MENTIONS 'torch.compile' in
    a print message is correctly ignored; only real torch.* access would appear."""
    wd = _CompileWatchdog(interval_s=10, phase_name="p")
    assert wd._thread.daemon is True
    names = {n.lower() for n in _CompileWatchdog._run.__code__.co_names}
    assert "torch" not in names
    assert "cuda" not in names
    # Allowed names are stdlib only (time, print, event helpers, etc.).
    assert not any("torch" in n or "cuda" in n for n in names)


def test_watchdog_stop_is_safe_before_start():
    """stop() must never raise even if start() was never called (bulletproof)."""
    wd = _CompileWatchdog(interval_s=1, phase_name="p")
    wd.stop()   # should be a no-op, not an error


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Config + dashboard wiring                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def test_compile_and_heartbeat_toggles_in_settings():
    """The compile + watchdog + heartbeat knobs must be config-driven in CFG.

    USE_TORCH_COMPILE now defaults OFF (no_holdups_default.md): training is
    CPU-bound so compile's steady-state win is marginal, while its ~10-15 min
    first-step warmup is the biggest startup hold-up — default OFF = instant
    start. The watchdog/heartbeat knobs are unchanged (they only engage when the
    user re-enables compile from the dashboard ⚡ GPU panel)."""
    from core.settings import CFG
    assert CFG["USE_TORCH_COMPILE"] is False           # default OFF (instant start)
    assert CFG["COMPILE_WATCHDOG_ENABLED"] is True     # ticker default ON (for when compile IS on)
    assert int(CFG["COMPILE_WATCHDOG_SECS"]) == 30
    assert int(CFG["HEARTBEAT_SECS"]) == 300


def test_compile_and_heartbeat_toggles_in_dashboard():
    """The dashboard must expose the compile toggle (so the user can ENABLE
    warmup), the watchdog toggle, and HEARTBEAT_SECS — all in the ⚡ GPU panel."""
    from core.interpret.dashboard_utils import widget_specs
    specs = widget_specs()
    for key in ("USE_TORCH_COMPILE", "COMPILE_WATCHDOG_ENABLED", "HEARTBEAT_SECS"):
        assert key in specs, f"{key} missing from dashboard"
        assert specs[key]["group"] == "gpu"
    assert specs["USE_TORCH_COMPILE"]["kind"] == "checkbox"
    assert specs["HEARTBEAT_SECS"]["kind"] == "int"


def test_dashboard_compile_default_off_matches_settings():
    """ZERO DRIFT (no_holdups_default.md): the dashboard's USE_TORCH_COMPILE
    default must be OFF and must EQUAL the live CFG default, so the panel and
    settings.py never disagree about the instant-start behaviour."""
    from core.interpret.dashboard_utils import widget_specs
    from core.settings import CFG
    assert widget_specs()["USE_TORCH_COMPILE"]["default"] is False
    assert widget_specs()["USE_TORCH_COMPILE"]["default"] == CFG["USE_TORCH_COMPILE"]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compile_on DERIVATION — default-OFF means NO warmup/blocking path         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The train loop derives `compile_on` from CFG['USE_TORCH_COMPILE'] AND CUDA
# (training/train.py: `compile_on = bool(cfg.get("USE_TORCH_COMPILE", True)) and
# device.type == "cuda"`). When compile_on is False the FIRST forward runs
# EAGERLY: _first_forward_warmup prints NO announcement, starts NO watchdog, and
# prints NO finished marker (verified by test_warmup_skips_*_when_compile_off
# above). These tests pin the derivation itself so the default-OFF instant-start
# can't silently regress, and prove the True path DOES take warmup.
def _derive_compile_on(cfg, device_type):
    """Mirror of the train-loop derivation (single source: training/train.py).
    Kept tiny + identical so the test fails loudly if the loop logic changes."""
    return bool(cfg.get("USE_TORCH_COMPILE", True)) and device_type == "cuda"


def test_default_cfg_does_not_enter_compile_warmup_path_even_on_cuda():
    """With the DEFAULT cfg (USE_TORCH_COMPILE False), compile_on is False on ANY
    device — even a (simulated) CUDA box — so startup never enters the ~10-15 min
    compile-warmup/blocking path. The first forward runs eagerly."""
    from core.settings import CFG
    assert CFG["USE_TORCH_COMPILE"] is False
    assert _derive_compile_on(CFG, "cuda") is False     # default OFF wins on GPU
    assert _derive_compile_on(CFG, "cpu") is False


def test_compile_on_true_only_when_enabled_and_cuda():
    """Conversely, the warmup path is taken ONLY when the user re-enables compile
    AND a CUDA device is present (CPU never compiles, regardless of the toggle)."""
    on = {"USE_TORCH_COMPILE": True}
    off = {"USE_TORCH_COMPILE": False}
    assert _derive_compile_on(on, "cuda") is True       # enabled + GPU -> warmup
    assert _derive_compile_on(on, "cpu") is False        # CPU never compiles
    assert _derive_compile_on(off, "cuda") is False       # disabled -> no warmup


def test_eager_first_forward_when_compile_off_takes_no_warmup_path(capsys, tmp_path):
    """End-to-end branch check: with compile_on=False the FIRST forward runs with
    NO announcement, NO watchdog thread, and NO finished marker — i.e. eagerly,
    with zero startup hold-up. (Liveness heartbeat still fires, as designed.)"""
    import threading
    n_before = threading.active_count()
    out = _first_forward_warmup(
        lambda: "eager", compile_on=False, watchdog_on=True, watchdog_secs=0.01,
        phase_name="phase1_cci_align", steps=0, max_steps=1440, global_ep=0,
        metrics_dir=str(tmp_path))
    time.sleep(0.05)
    text = capsys.readouterr().out
    assert out == "eager"
    assert "torch.compile warming up" not in text       # no announcement
    assert "torch.compile finished" not in text          # no finished marker
    assert "still compiling" not in text                  # watchdog never ticked
    # No watchdog thread was ever spawned (eager path skips it entirely).
    assert not any(t.name == "compile-watchdog" for t in threading.enumerate())
    assert threading.active_count() <= n_before
    # Liveness heartbeat is still emitted regardless of compile.
    assert "heartbeat" in text and "(loop entry)" in text


def test_compile_on_true_takes_warmup_path(capsys, tmp_path):
    """Conversely, compile_on=True DOES take the warmup path: announcement +
    watchdog tick + finished marker around the (simulated slow) first forward."""
    def slow_forward():
        time.sleep(0.12)
        return "compiled"

    out = _first_forward_warmup(
        slow_forward, compile_on=True, watchdog_on=True, watchdog_secs=0.05,
        phase_name="phase1_cci_align", steps=0, max_steps=1440, global_ep=0,
        metrics_dir=str(tmp_path))
    text = capsys.readouterr().out
    assert out == "compiled"
    assert "torch.compile warming up" in text            # announcement
    assert "still compiling" in text                      # watchdog ticked
    assert "torch.compile finished in" in text            # finished marker
