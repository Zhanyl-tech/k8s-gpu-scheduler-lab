"""The O(samples x pods) -> sweep rewrite of metrics.compute is bit-identical.

``legacy_compute`` below is the Phase 1 implementation copied VERBATIM from
``git show 771729d:src/k8slab/metrics.py`` (lines 23-29 and 107-257; SECONDS_PER_HOUR is imported, unchanged at 3600.0). The only
edits are mechanical renames so it can live beside the new code (``_legacy_``
prefixes, ``legacy_compute``) and returning a ``dict`` of the same keyword
arguments instead of constructing ``Metrics``. Do not "fix" it: its job is to
be the old behaviour.

Every metric that existed in Phase 1 must come out ``==`` (not approximately
equal) from the new implementation, on reference-model observations of every
degenerate policy on both shipped fleets and on hand-built edge cases, with one
deliberate exception: ``gang_wasted_gpu_hours`` was REPLACED by
``gang_stranded_gpu_hours`` (docs/metrics.md, "Gang behaviour"), and the
relationship between the two is tested separately in test_metrics.py.
"""

# ruff: noqa: E501

from __future__ import annotations

import statistics
from typing import Any

import pytest

from k8slab.fleet import load
from k8slab.metrics import SECONDS_PER_HOUR, _min_pending_at, compute
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent
from k8slab.sim import run
from k8slab.trace import WorkloadProfile, generate

# ---------------------------------------------------------------------------
# Phase 1 implementation, verbatim apart from renames. See module docstring.
# ---------------------------------------------------------------------------

def _legacy_percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency for two numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


def _legacy_stranded_at(free: dict[str, int], min_pending_request: int | None) -> int:
    """GPUs that are free but cannot satisfy the smallest pending pod.

    ``None`` means nothing is pending: idle capacity nobody wants is idle, not
    fragmented, and counting it as fragmentation would make an empty cluster
    look maximally fragmented.
    """
    if min_pending_request is None:
        return 0
    return sum(f for f in free.values() if 0 < f < min_pending_request)


def legacy_compute(obs: Observation, deadlock_threshold: float = 60.0) -> dict[str, Any]:
    jobs_by_id: dict[int, Job] = {j.job_id: j for j in obs.jobs}
    total_gpus = obs.fleet.total_gpus
    horizon = obs.horizon or max(
        (p.end_time or 0.0 for p in obs.pods),
        default=0.0,
    )

    # ---- delivered GPU-time -------------------------------------------------
    gpu_seconds_used = 0.0
    for pod in obs.pods:
        if pod.start_time is None or pod.end_time is None:
            continue
        gpu_seconds_used += jobs_by_id[pod.job_id].gpus * (pod.end_time - pod.start_time)

    capacity_seconds = total_gpus * horizon
    utilization = gpu_seconds_used / capacity_seconds if capacity_seconds > 0 else 0.0

    # ---- waits --------------------------------------------------------------
    # A job's wait is measured to the moment its LAST pod is bound: a gang that
    # is half-placed has not started, and crediting it with the first binding
    # would flatter every scheduler that admits gangs partially.
    waits: list[float] = []
    completed = 0
    for job in obs.jobs:
        pods = obs.pods_of(job.job_id)
        if pods and all(p.scheduled_time is not None for p in pods):
            last = max(p.scheduled_time or 0.0 for p in pods)
            waits.append(max(0.0, last - job.submit_time))
            completed += 1

    # ---- fragmentation ------------------------------------------------------
    # Trapezoid-integrate stranded GPUs and free GPUs over the sampled run.
    stranded_seconds = 0.0
    stranded_ref_seconds = 0.0
    free_seconds = 0.0
    # Definition B's reference request: the largest per-pod ask the trace
    # contains. Fixed for the whole run, so it cannot move with the queue.
    reference = max((j.gpus for j in obs.jobs), default=1) or 1
    samples = obs.gpu_free_samples
    for i in range(len(samples) - 1):
        t0, free0 = samples[i]
        t1, _ = samples[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        pending = _legacy_min_pending_request(obs, jobs_by_id, t0)
        stranded_seconds += _legacy_stranded_at(free0, pending) * dt
        stranded_ref_seconds += _legacy_stranded_at(free0, reference) * dt
        free_seconds += sum(free0.values()) * dt

    frag_of_free = stranded_seconds / free_seconds if free_seconds > 0 else 0.0
    frag_of_fleet = stranded_seconds / capacity_seconds if capacity_seconds > 0 else 0.0
    frag_ref = stranded_ref_seconds / free_seconds if free_seconds > 0 else 0.0

    # ---- gang behaviour -----------------------------------------------------
    gang_jobs = [j for j in obs.jobs if j.is_gang]
    stalled = 0
    deadlocked = 0
    wasted_seconds = 0.0
    for job in gang_jobs:
        pods = obs.pods_of(job.job_id)
        times = [p.scheduled_time for p in pods]
        placed = [t for t in times if t is not None]
        if not placed:
            continue
        if len(placed) < len(pods):
            # Some pods hold GPUs; at least one never arrived. Hard deadlock.
            deadlocked += 1
            wasted_seconds += job.gpus * len(placed) * max(0.0, horizon - min(placed))
            continue
        spread = max(placed) - min(placed)
        if spread > deadlock_threshold:
            stalled += 1
            # Pods placed early idle until the last one lands.
            wasted_seconds += sum(job.gpus * (max(placed) - t) for t in placed)

    deadlock_rate = (deadlocked + stalled) / len(gang_jobs) if gang_jobs else 0.0

    # ---- fairness -----------------------------------------------------------
    demanded: dict[str, float] = {}
    delivered: dict[str, float] = {}
    for job in obs.jobs:
        demanded[job.account] = demanded.get(job.account, 0.0) + job.total_gpus * job.duration
    for pod in obs.pods:
        if pod.start_time is None or pod.end_time is None:
            continue
        job = jobs_by_id[pod.job_id]
        delivered[job.account] = delivered.get(job.account, 0.0) + job.gpus * (
            pod.end_time - pod.start_time
        )
    ratios = {
        a: (delivered.get(a, 0.0) / d if d > 0 else 0.0) for a, d in demanded.items()
    }
    positive = [r for r in ratios.values() if r > 0]
    fairness = (max(positive) / min(positive)) if len(positive) > 1 else 1.0

    return dict(
        config=obs.config,
        measured_on_cluster=obs.measured_on_cluster,
        jobs=len(obs.jobs),
        pods=len(obs.pods),
        jobs_completed=completed,
        gpu_hours_used=gpu_seconds_used / SECONDS_PER_HOUR,
        makespan_hours=horizon / SECONDS_PER_HOUR,
        gpu_hours_idle=max(0.0, capacity_seconds - gpu_seconds_used) / SECONDS_PER_HOUR,
        utilization=utilization,
        mean_wait=statistics.fmean(waits) if waits else 0.0,
        p95_wait=_legacy_percentile(waits, 95),
        fragmentation_rate=frag_of_free,
        fragmentation_of_fleet=frag_of_fleet,
        stranded_gpu_hours=stranded_seconds / SECONDS_PER_HOUR,
        fragmentation_ref=frag_ref,
        reference_request=reference,
        gang_jobs=len(gang_jobs),
        gang_stalled=stalled,
        gang_deadlocked=deadlocked,
        gang_deadlock_rate=deadlock_rate,
        gang_wasted_gpu_hours=wasted_seconds / SECONDS_PER_HOUR,
        fairness_ratio=fairness,
        service_ratio=ratios,
    )


def _legacy_min_pending_request(
    obs: Observation, jobs_by_id: dict[int, Job], t: float
) -> int | None:
    """Smallest per-pod GPU request among pods pending at time ``t``."""
    smallest: int | None = None
    for pod in obs.pods:
        job = jobs_by_id[pod.job_id]
        if job.submit_time > t:
            continue
        if pod.scheduled_time is not None and pod.scheduled_time <= t:
            continue
        g = job.gpus
        if g > 0 and (smallest is None or g < smallest):
            smallest = g
    return smallest


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

#: Every Phase 1 metric except the one M2 deliberately replaced.
PRESERVED = (
    "config", "measured_on_cluster", "jobs", "pods", "jobs_completed",
    "gpu_hours_used", "makespan_hours", "gpu_hours_idle", "utilization",
    "mean_wait", "p95_wait", "fragmentation_rate", "fragmentation_of_fleet",
    "stranded_gpu_hours", "fragmentation_ref", "reference_request", "gang_jobs",
    "gang_stalled", "gang_deadlocked", "gang_deadlock_rate", "fairness_ratio",
    "service_ratio",
)


def _assert_identical(obs: Observation) -> None:
    old = legacy_compute(obs)
    new = compute(obs)
    assert set(old) == set(PRESERVED) | {"gang_wasted_gpu_hours"}
    for name in PRESERVED:
        # ==, not approx: the claim is bit-identity.
        assert getattr(new, name) == old[name], f"{obs.config}: {name} changed"


@pytest.mark.parametrize("fleet_file", ["default", "homogeneous"])
@pytest.mark.parametrize("policy", ["D-fifo", "D-random", "D-largest"])
@pytest.mark.parametrize("seed", [0, 3])
def test_new_compute_is_bit_identical_on_reference_model_runs(
    fleet_file: str, policy: str, seed: int
) -> None:
    fleet = load(f"fleets/{fleet_file}.yaml")
    jobs = generate(WorkloadProfile(job_count=200, arrival_interval=12.0), seed=seed)
    _assert_identical(run(fleet, jobs, policy, seed=seed))


def test_bit_identical_on_the_full_default_profile() -> None:
    """The contended 800-job trace the README's tables are built from."""
    fleet = load("fleets/default.yaml")
    jobs = generate(seed=0)
    for policy in ("D-fifo", "D-largest"):
        _assert_identical(run(fleet, jobs, policy))


FLEET = Fleet("t", (NodeClass("big", 1, 8), NodeClass("small", 2, 2)))


def test_bit_identical_on_edge_cases() -> None:
    """Never-bound pods, bound-before-submit, zero-GPU jobs, repeated and
    zero-length samples, a partial gang -- the cases a sweep gets wrong first."""
    jobs = [
        Job(1, "a", submit_time=0.0, duration=50.0, gpus=4),
        Job(2, "a", submit_time=5.0, duration=50.0, gpus=1, gang_size=2),
        Job(3, "b", submit_time=5.0, duration=10.0, gpus=0),
        Job(4, "b", submit_time=12.0, duration=30.0, gpus=2),
        Job(5, "c", submit_time=20.0, duration=30.0, gpus=8),
    ]
    pods = [
        PodEvent(1, 0, scheduled_time=10.0, start_time=10.0, end_time=60.0, node="big-0"),
        PodEvent(2, 0, scheduled_time=5.0, start_time=5.0, end_time=55.0, node="small-0"),
        PodEvent(2, 1),
        PodEvent(3, 0, scheduled_time=5.0, start_time=5.0, end_time=15.0, node="small-1"),
        # Bound "before" it was submitted: an empty pending interval.
        PodEvent(4, 0, scheduled_time=11.0, start_time=11.0, end_time=41.0, node="small-1"),
        PodEvent(5, 0),
    ]
    samples = [
        (0.0, {"big-0": 8, "small-0": 2, "small-1": 2}),
        (5.0, {"big-0": 8, "small-0": 1, "small-1": 2}),
        (5.0, {"big-0": 8, "small-0": 1, "small-1": 2}),
        (10.0, {"big-0": 4, "small-0": 1, "small-1": 2}),
        (12.0, {"big-0": 4, "small-0": 1, "small-1": 0}),
        (20.0, {"big-0": 4, "small-0": 1, "small-1": 0}),
        (60.0, {"big-0": 8, "small-0": 2, "small-1": 2}),
        (100.0, {"big-0": 8, "small-0": 2, "small-1": 2}),
    ]
    obs = Observation("edge", FLEET, jobs, pods, samples, horizon=100.0)
    _assert_identical(obs)


def test_pending_sweep_matches_the_scan_at_arbitrary_times() -> None:
    """Directly against the old per-sample scan, including unsorted, repeated
    and boundary query times (exactly at a submit or a bind)."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=150, arrival_interval=12.0), seed=9)
    obs = run(fleet, jobs, "D-random", seed=9)
    by_id = {j.job_id: j for j in obs.jobs}
    times = [t for t, _ in obs.gpu_free_samples][::7]
    times += [j.submit_time for j in jobs[::5]]
    times += [p.scheduled_time for p in obs.pods[::5] if p.scheduled_time is not None]
    times += [-1.0, 1e12]
    times = times[::-1]  # the sweep must not rely on the caller sorting
    swept = _min_pending_at(obs, by_id, times)
    for t in times:
        assert swept[t] == _legacy_min_pending_request(obs, by_id, t), t


def test_legacy_copy_is_still_the_old_behaviour() -> None:
    """Guard against someone editing the frozen copy: on this tiny observation
    the Phase 1 numbers were derived by hand."""
    job = Job(job_id=1, account="a", submit_time=0.0, duration=100.0, gpus=1, gang_size=2)
    obs = Observation(
        config="t", fleet=FLEET, jobs=[job],
        pods=[
            PodEvent(1, 0, scheduled_time=10.0, start_time=10.0, end_time=110.0, node="big-0"),
            PodEvent(1, 1, scheduled_time=90.0, start_time=90.0, end_time=190.0, node="big-0"),
        ],
        horizon=200.0,
    )
    old = legacy_compute(obs)
    assert old["mean_wait"] == 90.0
    assert old["gang_stalled"] == 1
    # Phase 1 wasted = g * (max T - t) summed: 1 * (90 - 10) seconds.
    assert old["gang_wasted_gpu_hours"] == 80.0 / 3600.0
    assert statistics.fmean([90.0]) == old["mean_wait"]
