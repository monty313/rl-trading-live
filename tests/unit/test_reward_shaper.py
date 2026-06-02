"""Unit tests for EpisodeRewardShaper + weekly consistency bonus."""
from core.reward.shaper import EpisodeRewardShaper
from core.settings import CFG


def _shaper(warmup=0):
    c = dict(CFG); c["SHAPE_WARMUP"] = warmup
    return EpisodeRewardShaper(c)


def test_warmup_returns_zero():
    s = _shaper(warmup=100)
    s.global_ep = 10
    assert s.compute_bonus([{"pass": True, "ret": 0.03, "dd": 0.0}]) == 0.0


def test_weekly_bonus_fires_when_rate_improves():
    s = _shaper(warmup=0)
    # prior week all fail, this week all pass -> improvement -> bonus
    for _ in range(7):
        s._daily_pass.append(0)
    for _ in range(7):
        s._daily_pass.append(1)
    assert s.weekly_consistency_bonus() == CFG["WEEKLY_BONUS"]


def test_no_weekly_bonus_when_flat():
    s = _shaper(warmup=0)
    for _ in range(14):
        s._daily_pass.append(1)
    assert s.weekly_consistency_bonus() == 0.0


def test_compute_bonus_runs_after_warmup():
    s = _shaper(warmup=0)
    s.global_ep = 5
    for _ in range(5):
        s.compute_bonus([{"pass": True, "ret": 0.03, "dd": 0.001}])
    b = s.compute_bonus([{"pass": True, "ret": 0.05, "dd": 0.0}])
    assert isinstance(b, float)
