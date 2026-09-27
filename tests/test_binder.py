"""The in-process binder (queue models, cycle budget, preemption) and the
D-preempt victim rule."""

from __future__ import annotations

import random

import pytest

from k8slab.baselines import (
    DEGENERATE_IDS,
    POLICIES,
    PREEMPTIVE_IDS,
    SPECS,
    PendingPod,
    _first_fit,
)
from k8slab.binder import Binder
from k8slab.model import RunInfo
from k8slab.preemption import RunningPod, plan_preemption
from k8slab.queueing import ACTIVE, BACKOFF, UNSCHEDULABLE, QueueParams

KUBE = RunInfo(queue_model="kube", speedup=60.0)
PARAMS = QueueParams(
    initial_backoff=60.0, max_backoff=60.0, max_in_unschedulable=300.0,
    backoff_flush_interval=60.0, unschedulable_flush_interval=1800.0,
)


def _pod(job: int, gpus: int, submit: float = 0.0, priority: int = 0, index: int = 0,
         gang: int = 1) -> PendingPod:
    return PendingPod(job_id=job, pod_index=index, gpus=gpus, submit_time=submit,
                      priority=priority, gang_size=gang)


# ---- queue model none: the literal Phase 1 call ----------------------------------


@pytest.mark.parametrize("config", ["D-fifo", "D-random", "D-largest"])
def test_none_is_the_phase_1_policy_call(config: str) -> None:
    pods = [_pod(i, g, float(i)) for i, g in enumerate([4, 2, 8, 1, 1, 4], start=1)]
    free = {"a": 8, "b": 4, "c": 2}
    expected = POLICIES[config](list(pods), dict(free), random.Random(5))
    got = Binder(config, RunInfo(), random.Random(5)).schedule(0.0, pods, free)
    assert got.binds == expected and got.preemptions == []


def test_none_draws_nothing_when_nothing_is_pending() -> None:
    """Phase 1 only called the policy when something was pending; D-random
    would otherwise consume RNG draws and shift every later decision."""
    rng = random.Random(1)
    Binder("D-random", RunInfo(), rng).schedule(0.0, [], {"a": 8})
    assert rng.random() == random.Random(1).random()


# ---- queue model kube ------------------------------------------------------------


def test_kube_parks_a_pod_that_does_not_fit_and_backs_it_off() -> None:
    b = Binder("D-fifo", KUBE, random.Random(0), PARAMS)
    big, small = _pod(1, 8, 0.0), _pod(2, 1, 1.0)
    d = b.schedule(0.0, [big, small], {"a": 4})
    assert [(p.job_id, n) for p, n in d.binds] == [(2, "a")]
    q = b.queue
    assert q is not None and q.where(big.key) == UNSCHEDULABLE and q.attempts(big.key) == 1
    # Not retried on the next pass: nothing happened to make it schedulable.
    d = b.schedule(5.0, [big], {"a": 8})
    assert d.binds == [] and q.attempts(big.key) == 1
    # Capacity freed while still backing off (60 s from t=0) -> backoffQ ...
    b.capacity_freed(10.0)
    assert q.where(big.key) == BACKOFF
    assert b.schedule(15.0, [big], {"a": 8}).binds == []
    # ... and the next backoff flush after expiry (t=60) makes it active.
    d = b.schedule(60.0, [big], {"a": 8})
    assert [(p.job_id, n) for p, n in d.binds] == [(1, "a")]


def test_kube_retries_immediately_when_backoff_is_already_over() -> None:
    b = Binder("D-fifo", KUBE, random.Random(0), PARAMS)
    big = _pod(1, 8)
    b.schedule(0.0, [big], {"a": 4})
    b.capacity_freed(100.0)  # backoff expired at 60
    assert b.queue is not None and b.queue.where(big.key) == ACTIVE
    assert b.schedule(100.0, [big], {"a": 8}).binds


def test_none_retries_every_pass_where_kube_waits() -> None:
    """The Phase 1 asymmetry, stated as a test: without the queue a pod that did
    not fit is tried again on the very next pass."""
    none = Binder("D-fifo", RunInfo(), random.Random(0))
    big = _pod(1, 8)
    none.schedule(0.0, [big], {"a": 4})
    assert none.schedule(5.0, [big], {"a": 8}).binds


def test_cycle_latency_limits_attempts_per_pass() -> None:
    info = RunInfo(queue_model="kube", speedup=60.0, cycle_latency=2.0)
    b = Binder("D-fifo", info, random.Random(0), PARAMS, window=5.0)
    pods = [_pod(i, 1, float(i)) for i in range(1, 6)]
    d = b.schedule(0.0, pods, {"a": 8})
    assert [p.job_id for p, _ in d.binds] == [1, 2]  # floor(5 / 2) attempts
    q = b.queue
    assert q is not None and q.where((3, 0)) == ACTIVE and q.attempts((3, 0)) == 0
    d = b.schedule(5.0, [p for p in pods if p.job_id > 2], {"a": 6})
    assert [p.job_id for p, _ in d.binds] == [3, 4]


def test_cycle_latency_needs_the_kube_queue() -> None:
    with pytest.raises(ValueError, match="cycle_latency"):
        RunInfo(cycle_latency=1.0)
    with pytest.raises(ValueError, match="QueueParams"):
        Binder("D-fifo", KUBE, random.Random(0))


# ---- D-preempt --------------------------------------------------------------------


def test_d_preempt_is_registered_as_a_preemptive_degenerate_policy() -> None:
    assert "D-preempt" in SPECS and "D-preempt" in DEGENERATE_IDS
    assert PREEMPTIVE_IDS == ("D-preempt",)
    assert not any(SPECS[c].preemptive for c in ("D-fifo", "D-random", "D-largest"))
    # Phase 1's whole-pass functions exist for exactly the non-preemptive
    # specs; D-preempt is always driven one pod at a time by the binder.
    assert set(POLICIES) == {c for c, s in SPECS.items() if not s.preemptive}
    assert DEGENERATE_IDS == tuple(SPECS)


def test_d_preempt_binds_in_priority_then_submission_order() -> None:
    pods = [_pod(1, 1, 0.0, 0), _pod(2, 1, 5.0, 500), _pod(3, 1, 1.0, 500), _pod(4, 1, 0.5, 100)]
    for info, params in ((RunInfo(), None), (KUBE, PARAMS)):
        d = Binder("D-preempt", info, random.Random(0), params).schedule(0.0, pods, {"a": 4})
        assert [p.job_id for p, _ in d.binds] == [3, 2, 4, 1]


@pytest.mark.parametrize("config", ["D-fifo", "D-random", "D-largest"])
@pytest.mark.parametrize("seed", [0, 5, 11])
def test_whole_pass_functions_and_the_per_cycle_path_cannot_drift(config: str, seed: int) -> None:
    """Two registries describe each non-preemptive policy: its Phase 1 whole-pass
    function (POLICIES, used verbatim under queue model none) and its spec's
    pod-order/node-order halves (SPECS, used one cycle at a time under kube).
    On a first pass, where every pod is in activeQ, both must make the same
    decisions from the same RNG draws."""
    pods = [_pod(i, g, float(i % 3)) for i, g in enumerate([4, 2, 8, 1, 1, 4, 2, 8], start=1)]
    free = {"a": 8, "b": 4, "c": 2, "d": 8}
    whole = POLICIES[config](list(pods), dict(free), random.Random(seed))
    spec = SPECS[config]
    rng = random.Random(seed)
    halves = _first_fit(spec.order(list(pods), rng), dict(free), spec.node_order(dict(free), rng))
    per_cycle = Binder(config, KUBE, random.Random(seed), PARAMS).schedule(0.0, pods, free)
    assert whole == halves == per_cycle.binds


def test_plan_picks_the_node_with_the_fewest_evictions() -> None:
    running = [
        RunningPod((10, 0), "a", 2, 0), RunningPod((11, 0), "a", 2, 0),  # a: two jobs
        RunningPod((12, 0), "b", 4, 0),  # b: one job frees 4
    ]
    plan = plan_preemption(_pod(1, 4, priority=100), {"a": 0, "b": 0}, running)
    assert plan is not None and plan.node == "b" and plan.victims == ((12, 0),)


def test_plan_evicts_whole_gangs_and_counts_every_member() -> None:
    running = [
        # gang 20: one member here, one elsewhere -> 2 evictions
        RunningPod((20, 0), "a", 4, 0), RunningPod((20, 1), "b", 4, 0),
        # two singles on c -> 2 evictions, lower max priority does not matter first
        RunningPod((21, 0), "c", 2, 0), RunningPod((22, 0), "c", 2, 0),
    ]
    plan = plan_preemption(_pod(1, 4, priority=100), {"a": 0, "b": 4, "c": 0}, running)
    # b already fits: not a preemption case there. a and c both need 2
    # evictions; tie broken on the highest victim priority (equal), then name.
    assert plan is not None and plan.node == "a"
    assert plan.victims == ((20, 0), (20, 1))


def test_plan_never_evicts_equal_or_higher_priority() -> None:
    running = [RunningPod((10, 0), "a", 8, 100)]
    assert plan_preemption(_pod(1, 8, priority=100), {"a": 0}, running) is None
    assert plan_preemption(_pod(1, 8, priority=500), {"a": 0}, running) is not None
    assert plan_preemption(_pod(1, 0, priority=500), {"a": 0}, running) is None


def test_plan_prefers_lower_priority_victims_on_a_node() -> None:
    running = [RunningPod((10, 0), "a", 4, 100), RunningPod((11, 0), "a", 4, 0)]
    plan = plan_preemption(_pod(1, 4, priority=500), {"a": 0}, running)
    assert plan is not None and plan.victims == ((11, 0),)


def test_binder_nominates_the_preemptor_and_reserves_the_node() -> None:
    for info, params in ((RunInfo(), None), (KUBE, PARAMS)):
        b = Binder("D-preempt", info, random.Random(0), params)
        hi, lo = _pod(1, 8, 10.0, 500), _pod(2, 4, 0.0, 0)
        running = [RunningPod((9, 0), "a", 8, 0)]
        d = b.schedule(10.0, [hi, lo], {"a": 0, "b": 4}, running)
        assert [(p.preemptor, p.node, p.victims) for p in d.preemptions] == [
            ((1, 0), "a", ((9, 0),))
        ]
        assert [(p.job_id, n) for p, n in d.binds] == [(2, "b")]
        assert b.nominated == {(1, 0): ("a", 8)}
        # While the victim's GPUs are grace-locked nothing preempts again ...
        again = b.schedule(15.0, [hi], {"a": 0, "b": 0}, [])
        assert again.preemptions == [] and again.binds == []
        # ... and while only part of the node is free -- too little for the
        # preemptor, enough for a 1-GPU pod -- the lower-priority pod never
        # takes the reserved GPUs, pass after pass. (The previous form of this
        # check freed all 8 GPUs, so the preemptor, tried first by priority,
        # always bound and the reservation was never exercised: deleting the
        # rule left every test passing.)
        other = _pod(3, 1, 20.0, 0)
        for now in (120.0, 180.0, 240.0):
            b.capacity_freed(now)
            d = b.schedule(now, [other, hi], {"a": 4, "b": 0}, [])
            assert d.binds == [] and d.preemptions == [], now
            assert b.nominated == {(1, 0): ("a", 8)}
        # Once the node is whole the preemptor takes it, and the nomination ends.
        b.capacity_freed(300.0)
        d = b.schedule(300.0, [other, hi], {"a": 8, "b": 0}, [])
        assert [(p.job_id, n) for p, n in d.binds] == [(1, "a")]
        assert b.nominated == {}


def test_nomination_is_dropped_when_the_preemptor_is_no_longer_pending() -> None:
    b = Binder("D-preempt", RunInfo(), random.Random(0))
    b.schedule(0.0, [_pod(1, 8, 0.0, 500)], {"a": 0}, [RunningPod((9, 0), "a", 8, 0)])
    assert b.nominated
    b.schedule(5.0, [_pod(2, 1)], {"a": 0}, [])
    assert b.nominated == {}
