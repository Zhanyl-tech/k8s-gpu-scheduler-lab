"""kube-scheduler queue mechanics (k8slab.queueing), against the v1.32.2 source
semantics cited in the module docstring."""

from __future__ import annotations

import pytest

from k8slab.queueing import (
    ACTIVE,
    BACKOFF,
    IN_FLIGHT,
    UNSCHEDULABLE,
    QueueParams,
    SchedulingQueue,
)
from k8slab.timescale import TimeScale

K = (1, 0)
FAST = QueueParams(
    initial_backoff=1.0,
    max_backoff=10.0,
    max_in_unschedulable=300.0,
    backoff_flush_interval=1.0,
    unschedulable_flush_interval=30.0,
)


def _fail(q: SchedulingQueue, key: tuple[int, int], now: float) -> None:
    q.pop(key)
    q.failed(key, now)


def test_backoff_doubles_from_the_initial_value_and_caps() -> None:
    """calculateBackoffDuration: 0 attempts -> 0; then 1, 2, 4, 8, 10, 10."""
    q = SchedulingQueue(FAST)
    assert [q.backoff_duration(a) for a in range(7)] == [0.0, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


def test_a_new_pod_is_active_with_no_attempts() -> None:
    q = SchedulingQueue(FAST)
    q.add(K, 5.0)
    assert q.where(K) == ACTIVE and q.attempts(K) == 0
    assert not q.is_backing_off(K, 5.0)
    q.add(K, 9.0)  # idempotent: re-adding does not reset anything
    assert q.attempts(K) == 0


def test_pop_counts_an_attempt_and_failure_parks_the_pod() -> None:
    q = SchedulingQueue(FAST)
    q.add(K, 0.0)
    q.pop(K)
    assert q.where(K) == IN_FLIGHT and q.attempts(K) == 1
    q.failed(K, 3.0)
    assert q.where(K) == UNSCHEDULABLE
    # Backoff runs from the failure timestamp: 3 + 1 s.
    assert q.backoff_expiry(K) == 4.0
    with pytest.raises(ValueError, match="not activeQ"):
        q.pop(K)


def test_capacity_freed_moves_parked_pods_to_backoff_or_active() -> None:
    q = SchedulingQueue(FAST)
    q.add(K, 0.0)
    q.add((2, 0), 0.0)
    _fail(q, K, 10.0)  # expiry 11
    _fail(q, (2, 0), 0.0)  # expiry 1
    q.capacity_freed(10.5)
    assert q.where(K) == BACKOFF  # still backing off
    assert q.where((2, 0)) == ACTIVE  # backoff long over
    # "After": a pod whose backoff expires exactly now is no longer backing off.
    assert not q.is_backing_off(K, 11.0)


def test_backoff_flush_runs_on_its_own_period() -> None:
    q = SchedulingQueue(QueueParams(1.0, 10.0, 300.0, 60.0, 1800.0))
    q.add(K, 0.0)
    q.flush(0.0)  # first tick at t=0, next due at 60
    _fail(q, K, 0.0)
    q.capacity_freed(0.5)
    assert q.where(K) == BACKOFF
    q.flush(30.0)  # backoff expired at 1, but the flush is not due
    assert q.where(K) == BACKOFF
    q.flush(60.0)
    assert q.where(K) == ACTIVE


def test_a_backoff_flush_tick_leaves_a_pod_that_is_still_backing_off() -> None:
    """The speedup-60 case: backoff (60 s) equals the flush period (60 s). A
    pod that failed at t=5 backs off until 65, so the t=60 tick must leave it
    in backoffQ and the t=120 tick move it. (Every other test's pod had
    already finished backing off at its first tick, so removing the expiry
    check in flush left the suite green while moving the reference model's
    D-largest mean wait from 126.3 to 130.6 min.)"""
    q = SchedulingQueue(QueueParams(60.0, 60.0, 300.0, 60.0, 1800.0))
    q.add(K, 0.0)
    q.flush(0.0)  # first tick at t=0, next due at 60
    _fail(q, K, 5.0)
    assert q.backoff_expiry(K) == 65.0
    q.capacity_freed(10.0)  # released by an event while still backing off
    assert q.where(K) == BACKOFF
    q.flush(60.0)
    assert q.where(K) == BACKOFF  # the tick came, the backoff has not ended
    q.flush(120.0)
    assert q.where(K) == ACTIVE


def test_parked_pods_leave_after_the_unschedulable_timeout_without_an_event() -> None:
    q = SchedulingQueue(FAST)
    q.add(K, 0.0)
    _fail(q, K, 0.0)
    for t in range(0, 300, 30):
        q.flush(float(t))
        assert q.where(K) == UNSCHEDULABLE
    q.flush(300.0)  # 300 - 0 is not > 300
    assert q.where(K) == UNSCHEDULABLE
    q.flush(330.0)
    assert q.where(K) == ACTIVE


def test_reconcile_adds_new_and_drops_vanished_pods() -> None:
    q = SchedulingQueue(FAST)
    q.reconcile([(1, 0), (2, 0)], 0.0)
    _fail(q, (1, 0), 0.0)
    q.reconcile([(1, 0), (3, 0)], 1.0)
    assert (2, 0) not in q and (3, 0) in q
    assert q.where((1, 0)) == UNSCHEDULABLE and q.attempts((1, 0)) == 1
    assert q.keys_in(ACTIVE) == [(3, 0)]


def test_done_removes_the_pod() -> None:
    q = SchedulingQueue(FAST)
    q.add(K, 0.0)
    q.pop(K)
    q.done(K)
    assert K not in q and len(q) == 0


def test_params_follow_what_kube_scheduler_experiences_at_a_speedup() -> None:
    """At speedup 60 the integer schema forces 1 s real = 60 simulated s for
    both backoff fields; the flush timers are unscalable (1 s, 30 s real)."""
    p = QueueParams.from_timescale(TimeScale(60.0))
    assert (p.initial_backoff, p.max_backoff) == (60.0, 60.0)
    assert p.max_in_unschedulable == pytest.approx(300.0)
    assert (p.backoff_flush_interval, p.unschedulable_flush_interval) == (60.0, 1800.0)
    # At speedup 1 every timing is the upstream default (1 s, 10 s, 5 min, and
    # the two flush periods), in simulated seconds.
    assert QueueParams.from_timescale(TimeScale(1.0)) == QueueParams(
        initial_backoff=1.0, max_backoff=10.0, max_in_unschedulable=300.0,
        backoff_flush_interval=1.0, unschedulable_flush_interval=30.0,
    )


def test_params_are_validated_like_the_scheduler_config() -> None:
    with pytest.raises(ValueError):
        QueueParams(0.0, 10.0, 300.0, 1.0, 30.0)
    with pytest.raises(ValueError):
        QueueParams(5.0, 1.0, 300.0, 1.0, 30.0)
