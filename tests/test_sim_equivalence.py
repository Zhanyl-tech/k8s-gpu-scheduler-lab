"""With every Phase 2 setting at its default, the reference model is Phase 1.

``legacy_run`` and the ``_legacy_*`` policies below are the Phase 1 code copied
VERBATIM from ``git show 493d815:src/k8slab/sim.py`` (lines 17-109) and
``git show 493d815:src/k8slab/baselines.py`` (lines 17-102). The only edits are
mechanical renames so they can live beside the new code (``_legacy`` /
``LEGACY_`` prefixes) and dropping their relative imports. Do not "fix" them:
their job is to be the old behaviour.

The new ``sim.run`` routes the same decisions through ``k8slab.binder`` and
``k8slab.execution``. Every PodEvent and every free-GPU sample must come out
``==`` -- not approximately equal -- on every Phase 1 policy, both shipped
fleets and several seeds, including D-random's RNG draws.
"""

# ruff: noqa: E501

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from k8slab.fleet import load
from k8slab.model import Fleet, Job, Observation, PodEvent, RunInfo
from k8slab.sim import run
from k8slab.trace import WorkloadProfile, generate

# ---------------------------------------------------------------------------
# Phase 1 code, verbatim apart from renames. See module docstring.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegacyPendingPod:
    job_id: int
    pod_index: int
    gpus: int
    submit_time: float
    priority: int
    gang_size: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.job_id, self.pod_index)


#: (pending pods, free GPUs per node, rng) -> [(pod, node name)]
_LegacyPolicy = Callable[[list[_LegacyPendingPod], dict[str, int], random.Random], list[tuple[_LegacyPendingPod, str]]]


def _legacy_first_fit(
    pods: list[_LegacyPendingPod],
    free: dict[str, int],
    node_order: list[str],
) -> list[tuple[_LegacyPendingPod, str]]:
    """Bind each pod to the first node in ``node_order`` with room.

    ``free`` is mutated as bindings are made, so one pass never double-books a
    node. This is the behaviour the degenerate policies share; only the pod
    ordering and the node ordering differ between them.
    """
    out: list[tuple[_LegacyPendingPod, str]] = []
    for pod in pods:
        for node in node_order:
            if free.get(node, 0) >= pod.gpus and pod.gpus > 0:
                free[node] -= pod.gpus
                out.append((pod, node))
                break
    return out


def _legacy_fifo(
    pods: list[_LegacyPendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[_LegacyPendingPod, str]]:
    """Oldest submission first, first node with room. No packing, no gang logic."""
    ordered = sorted(pods, key=lambda p: (p.submit_time, p.job_id, p.pod_index))
    return _legacy_first_fit(ordered, dict(free), sorted(free))


def _legacy_random_policy(
    pods: list[_LegacyPendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[_LegacyPendingPod, str]]:
    """Random pod order, random node order. The true floor."""
    ordered = list(pods)
    rng.shuffle(ordered)
    nodes = sorted(free)
    rng.shuffle(nodes)
    return _legacy_first_fit(ordered, dict(free), nodes)


def _legacy_largest_first(
    pods: list[_LegacyPendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[_LegacyPendingPod, str]]:
    """Biggest request first. Often beats FIFO on utilization and starves small jobs.

    Included because it is the cheap heuristic people reach for, and because a
    scheduler that loses to it on fragmentation is worth knowing about.
    """
    ordered = sorted(
        pods, key=lambda p: (-p.gpus, p.submit_time, p.job_id, p.pod_index)
    )
    return _legacy_first_fit(ordered, dict(free), sorted(free))


#: Registry. Keys are the config ids used in results tables and `--config`.
LEGACY_POLICIES: dict[str, _LegacyPolicy] = {
    "D-fifo": _legacy_fifo,
    "D-random": _legacy_random_policy,
    "D-largest": _legacy_largest_first,
}


#: Seconds between scheduling passes. Matches the runner's poll interval so the
#: two paths have the same granularity.
LEGACY_TICK = 5.0


def legacy_run(
    fleet: Fleet,
    jobs: list[Job],
    config: str,
    seed: int = 0,
    max_horizon: float = 48 * 3600.0,
) -> Observation:
    if config not in LEGACY_POLICIES:
        raise KeyError(f"unknown policy {config!r}; have {sorted(LEGACY_POLICIES)}")
    policy = LEGACY_POLICIES[config]
    rng = random.Random(seed)

    free: dict[str, int] = {n: fleet.gpus_of(n) for n in fleet.node_names()}
    pods: dict[tuple[int, int], PodEvent] = {}
    pending: dict[tuple[int, int], _LegacyPendingPod] = {}
    running: list[tuple[float, tuple[int, int], str, int]] = []  # (end, key, node, gpus)
    by_id = {j.job_id: j for j in jobs}

    for job in jobs:
        for i in range(job.gang_size):
            key = (job.job_id, i)
            pods[key] = PodEvent(job_id=job.job_id, pod_index=i)

    samples: list[tuple[float, dict[str, int]]] = []
    t = 0.0
    last_submit = max(j.submit_time for j in jobs)

    while t < max_horizon:
        # Release finished pods.
        for end, key, node, gpus in [r for r in running if r[0] <= t]:
            free[node] += gpus
            pods[key].end_time = end
        running = [r for r in running if r[0] > t]

        # Admit newly-submitted pods.
        for job in jobs:
            if job.submit_time <= t:
                for i in range(job.gang_size):
                    key = (job.job_id, i)
                    if pods[key].scheduled_time is None and key not in pending:
                        pending[key] = _LegacyPendingPod(
                            job_id=job.job_id,
                            pod_index=i,
                            gpus=job.gpus,
                            submit_time=job.submit_time,
                            priority=job.priority,
                            gang_size=job.gang_size,
                        )

        samples.append((t, dict(free)))

        if pending:
            for pod, node in policy(list(pending.values()), free, rng):
                job = by_id[pod.job_id]
                free[node] -= pod.gpus
                ev = pods[pod.key]
                ev.scheduled_time = t
                ev.start_time = t
                ev.node = node
                running.append((t + job.duration, pod.key, node, pod.gpus))
                pending.pop(pod.key, None)

        if not pending and not running and t > last_submit:
            break
        t += LEGACY_TICK

    horizon = t
    for ev in pods.values():
        if ev.start_time is not None and ev.end_time is None:
            ev.end_time = horizon
    samples.append((horizon, dict(free)))

    return Observation(
        config=config,
        fleet=fleet,
        jobs=jobs,
        pods=list(pods.values()),
        gpu_free_samples=samples,
        horizon=horizon,
        measured_on_cluster=False,
    )


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _same(new: Observation, old: Observation) -> None:
    assert new.horizon == old.horizon
    assert new.gpu_free_samples == old.gpu_free_samples
    assert [(p.job_id, p.pod_index, p.scheduled_time, p.start_time, p.end_time, p.node,
             p.preempted) for p in new.pods] == [
        (p.job_id, p.pod_index, p.scheduled_time, p.start_time, p.end_time, p.node,
         p.preempted) for p in old.pods
    ]
    assert new.evictions == [] and new.extension_gpu_seconds == 0.0


@pytest.mark.parametrize("fleet_file", ["default", "homogeneous"])
@pytest.mark.parametrize("policy", ["D-fifo", "D-random", "D-largest"])
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_default_harness_is_bit_identical_to_phase_1(
    fleet_file: str, policy: str, seed: int
) -> None:
    fleet = load(f"fleets/{fleet_file}.yaml")
    jobs = generate(WorkloadProfile(job_count=150, arrival_interval=12.0), seed=seed)
    _same(run(fleet, jobs, policy, seed=seed), legacy_run(fleet, jobs, policy, seed=seed))


def test_explicit_phase_1_settings_are_the_same_as_the_defaults() -> None:
    """queue none, report mode, zero delay -- spelled out, with a speedup that
    only a queue model or a delay would read -- is still Phase 1."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=150), seed=2)
    info = RunInfo(queue_model="none", topology_penalty="report", speedup=60.0)
    _same(run(fleet, jobs, "D-random", seed=2, run_info=info),
          legacy_run(fleet, jobs, "D-random", seed=2))


def test_full_default_profile_is_bit_identical() -> None:
    """The contended 800-job trace the README's model table is built from."""
    fleet = load("fleets/default.yaml")
    jobs = generate(seed=0)
    _same(run(fleet, jobs, "D-fifo"), legacy_run(fleet, jobs, "D-fifo"))


def test_legacy_copy_is_still_the_old_behaviour() -> None:
    """Guard against edits to the frozen copy: hand-derived on a tiny case."""
    fleet = load("fleets/homogeneous.yaml")
    jobs = [Job(1, "a", 0.0, 12.0, 8), Job(2, "a", 1.0, 3.0, 8)]
    obs = legacy_run(fleet, jobs, "D-fifo")
    first, second = obs.pods
    assert (first.scheduled_time, first.end_time, first.node) == (0.0, 12.0, "dgx8-0")
    # Admitted at the t=5 pass; dgx8-0 is full, so first-fit moves on.
    assert (second.scheduled_time, second.end_time, second.node) == (5.0, 8.0, "dgx8-1")
    assert obs.horizon == 15.0
    assert PodEvent(1, 0).scheduled is False
