"""
TA-Lib vs numpy indicator parity (user-flagged). On Colab/Windows TA-Lib is the
source of truth; CI uses the numpy fallback. This test runs ONLY when talib is
installed and asserts the numpy fallback matches talib within tolerance, so
behavior does not shift between CPU (numpy) and A100 (talib).
"""
import numpy as np
import pytest

talib = pytest.importorskip("talib")  # skipped automatically when talib absent

from core.env import indicators as I
from tests.fixtures.sample_candles import make_synthetic_candles


def _series():
    df = make_synthetic_candles(n=500, seed=3)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    return h, l, c


def test_cci_parity():
    h, l, c = _series()
    np_cci = I._np_cci(h, l, c, 30)
    tl_cci = talib.CCI(h, l, c, timeperiod=30)
    ok = np.isfinite(np_cci) & np.isfinite(tl_cci)
    # CCI scaling matches closely (loose tol — both use the same definition)
    assert np.nanmax(np.abs(np_cci[ok][60:] - tl_cci[ok][60:])) < 5.0


def test_rsi_parity():
    _h, _l, c = _series()
    np_rsi = I._np_rsi(c, 14)
    tl_rsi = talib.RSI(c, timeperiod=14)
    ok = np.isfinite(np_rsi) & np.isfinite(tl_rsi)
    assert np.nanmax(np.abs(np_rsi[ok][30:] - tl_rsi[ok][30:])) < 3.0


def test_bbands_parity():
    _h, _l, c = _series()
    nu, nm, nl = I._np_bbands(c, 20, 1.0)
    tu, tm, tl_ = talib.BBANDS(c, timeperiod=20, nbdevup=1.0, nbdevdn=1.0, matype=0)
    ok = np.isfinite(nm) & np.isfinite(tm)
    assert np.nanmax(np.abs(nm[ok][30:] - tm[ok][30:])) < 1e-6   # middle = SMA, exact
