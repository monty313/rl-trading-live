"""Unit tests for flatline detection + alert dispatch resilience."""
from monitoring.flatline_detector import FlatlineDetector
from monitoring.alert_dispatcher import AlertDispatcher
from tests.mocks.mock_telegram import MockTelegram


def test_flatline_fires_on_flat_series():
    ad = AlertDispatcher()
    fd = FlatlineDetector(window=10, alert_dispatcher=ad)
    fired = False
    for _ in range(10):
        fired = fd.record(0.50)   # identical pass rate 10x
    assert fired is True
    assert fd.last_irac is not None
    assert any(a["level"] == "FLATLINE" for a in ad.alerts)


def test_no_flatline_when_improving():
    fd = FlatlineDetector(window=10)
    fired = False
    for i in range(10):
        fired = fd.record(0.40 + i * 0.02)   # steadily improving
    assert fired is False


def test_alert_appends_to_session_and_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    tg = MockTelegram()
    ad = AlertDispatcher(telegram_sender=tg)
    ad.fire("WARNING", "test message")
    assert len(ad.alerts) == 1
    assert len(tg.messages) == 1   # telegram called when token set


def test_alert_never_raises_when_all_channels_fail(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    tg = MockTelegram(fail=True)
    class BadMT5:
        def send_notification(self, m): raise RuntimeError("boom")
    ad = AlertDispatcher(mt5_module=BadMT5(), telegram_sender=tg)
    entry = ad.fire("CRITICAL", "everything is down")   # must not raise
    assert entry["level"] == "CRITICAL"
