"""In-process reference model.

This is **not** a Kubernetes simulator and makes no claim to reproduce
kube-scheduler. It applies a :mod:`k8slab.baselines` policy to the same trace
and fleet on a discrete-event clock, so that:

* the metrics code has something to be tested against in CI, where no cluster
  exists; and
* a degenerate policy has a cheap sanity number before it is run for real.

Any :class:`~k8slab.model.Observation` it produces carries
``measured_on_cluster=False``, and every report prints that flag next to the
numbers. Do not put a reference-model number in a results table beside a
cluster number without saying which is which.
"""

from __future__ import annotations

import random

from .baselines import POLICIES, PendingPod
from .model import Fleet, Job, Observation, PodEvent

#: Seconds between scheduling passes. Matches the runner's poll interval so the
#: two paths have the same granularity.
TICK = 5.0


def run(
    fleet: Fleet,
    jobs: list[Job],
    config: str,
    seed: int = 0,
    max_horizon: float = 48 * 3600.0,
) -> Observation:
    if config not in POLICIES:
        raise KeyError(f"unknown policy {config!r}; have {sorted(POLICIES)}")
    policy = POLICIES[config]
    rng = random.Random(seed)

    free: dict[str, int] = {n: fleet.gpus_of(n) for n in fleet.node_names()}
    pods: dict[tuple[int, int], PodEvent] = {}
    pending: dict[tuple[int, int], PendingPod] = {}
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
                        pending[key] = PendingPod(
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
        t += TICK

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
