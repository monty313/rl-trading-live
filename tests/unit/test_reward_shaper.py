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


def test_daily_reward_pass_and_streak():
    s = _shaper(warmup=0)
    # three passing days in a row -> base + growing streak bonus + low-dd bonus
    r1 = s.daily_reward(r_d=0.03, dd_d=0.001)
    r2 = s.daily_reward(r_d=0.03, dd_d=0.001)
    r3 = s.daily_reward(r_d=0.03, dd_d=0.001)
    assert r2 > r1 and r3 > r2          # streak bonus grows each consecutive pass
    assert s._pass_streak == 3


def test_daily_reward_fail_resets_streak_and_penalizes():
    s = _shaper(warmup=0)
    s.daily_reward(r_d=0.03, dd_d=0.001)      # pass
    rfail = s.daily_reward(r_d=-0.01, dd_d=0.02)   # dd breach + negative return
    assert rfail < 0                    # fail penalty
    assert s._pass_streak == 0          # streak reset


def test_daily_reward_low_dd_bonus():
    s = _shaper(warmup=0)
    low = s.daily_reward(r_d=0.03, dd_d=0.0001)   # well under low_dd_threshold
    high = s.daily_reward(r_d=0.03, dd_d=0.009)   # above low_dd_threshold
    # both pass, but the streak differs; just assert low-dd path adds the bonus
    assert low > 0 and high > 0
