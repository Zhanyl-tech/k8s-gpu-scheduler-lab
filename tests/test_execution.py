"""Shared execution bookkeeping: harness seeds, startup delay, attempts."""

from __future__ import annotations

import random

import pytest

from k8slab.execution import Attempt, StartupDelay, harness_seed


def test_repeat_zero_uses_the_seed_itself() -> None:
    """So a one-repeat benchmark reproduces Phase 1 exactly."""
    assert harness_seed(0, 0) == 0 and harness_seed(7, 0) == 7


def test_later_repeats_are_distinct_deterministic_and_seed_dependent() -> None:
    seeds = [harness_seed(0, r) for r in range(20)]
    assert len(set(seeds)) == 20
    assert seeds == [harness_seed(0, r) for r in range(20)]
    assert harness_seed(1, 3) != harness_seed(0, 3)
    with pytest.raises(ValueError):
        harness_seed(0, -1)


def test_startup_delay_parses_and_samples_like_kwok() -> None:
    d = StartupDelay.parse("50:200")
    assert (d.min_ms, d.max_ms) == (50.0, 200.0) and d.text() == "50:200"
    rng = random.Random(3)
    draws = [d.sample_ms(rng) for _ in range(2000)]
    assert all(50.0 <= x < 200.0 for x in draws)
    assert 110 < sum(draws) / len(draws) < 140  # uniform mean 125
    const = StartupDelay.parse("80")
    assert const.sample_ms(random.Random(0)) == 80.0
    # kwok: jitter <= duration means the jitter value is the delay.
    assert StartupDelay(0.0, 0.0).zero
    with pytest.raises(ValueError):
        StartupDelay.parse("200:50")
    with pytest.raises(ValueError):
        StartupDelay.parse("1:2:3")


@pytest.mark.parametrize("text", ["nan", "inf", "0:inf", "nan:200", "-inf:0", "0:nan"])
def test_startup_delay_refuses_non_finite_milliseconds(text: str) -> None:
    """Every range check is a comparison, and NaN fails none of them: "nan"
    ran a 48 h replay with NaN delivered GPU-hours and exited 0; "0:inf"
    delivered nothing and exited 0."""
    with pytest.raises(ValueError, match="finite"):
        StartupDelay.parse(text)


def test_startup_delay_converts_real_milliseconds_to_simulated_seconds() -> None:
    rng = random.Random(0)
    assert StartupDelay().sample_sim(rng, 60.0) == 0.0
    assert StartupDelay(100.0, 100.0).sample_sim(rng, 60.0) == pytest.approx(6.0)
    # A zero delay draws nothing: switching it on never shifts other streams.
    r1, r2 = random.Random(9), random.Random(9)
    StartupDelay().sample_sim(r1, 60.0)
    assert r1.random() == r2.random()


def test_attempt_with_factor_one_is_the_phase_1_arithmetic() -> None:
    a = Attempt((1, 0), "n", 2, 0, bind_time=10.0, work=123.456)
    assert a.end is None
    a.start(10.0)
    assert a.end == 10.0 + 123.456  # bit-identical to Phase 1's t + duration
    assert a.stretch_seconds(500.0) == 0.0


def test_factor_set_before_start_stretches_all_the_work() -> None:
    a = Attempt((1, 0), "n", 4, 0, bind_time=0.0, work=100.0)
    a.set_factor(1.4, 0.0)
    a.start(5.0)
    assert a.end == pytest.approx(5.0 + 140.0)
    assert a.stretch_seconds(a.end or 0.0) == pytest.approx(40.0)
    assert a.work_done(75.0) == pytest.approx(50.0)


def test_factor_set_while_running_stretches_only_the_remaining_work() -> None:
    a = Attempt((1, 0), "n", 1, 0, bind_time=0.0, work=100.0)
    a.start(0.0)
    a.set_factor(2.0, 40.0)  # 40 done, 60 left at x2
    assert a.end == pytest.approx(40.0 + 120.0)
    assert a.work_done(100.0) == pytest.approx(70.0)
    assert a.stretch_seconds(160.0) == pytest.approx(60.0)
    a.set_factor(2.0, 50.0)  # unchanged factor: no new segment
    assert a.end == pytest.approx(160.0)


def test_a_finished_attempt_is_never_stretched() -> None:
    """The runner polls, so an attempt can be finished and still bound. Setting
    a factor on it used to move its end to 'now': Running time past its
    duration, booked as topology extension."""
    a = Attempt((1, 0), "n", 1, 0, bind_time=60.0, work=200.0)
    assert not a.finished_by(1e9)  # not started: no end yet
    a.start(60.0)
    assert not a.finished_by(259.0) and a.finished_by(260.0) and a.finished_by(300.0)
    a.set_factor(1.2, 300.0)
    assert a.end == 260.0 and a.factor == 1.0 and not a.stretched
    assert a.stretch_seconds(300.0) == 0.0
    # At exactly its end there is no remaining work either.
    b = Attempt((1, 1), "n", 1, 0, bind_time=0.0, work=100.0)
    b.start(0.0)
    b.set_factor(2.0, 100.0)
    assert b.end == 100.0 and not b.stretched
