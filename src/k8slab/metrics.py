"""Metrics. Every definition here is also written out in docs/metrics.md.

Read that file before quoting a number from this one. Fragmentation in
particular has no canonical definition in the literature, and the published
claims that motivated this repo do not state theirs — so a number is only
comparable to another number computed the same way.

Everything is computed from an :class:`~k8slab.model.Observation`, so a run
against a real control plane and a run against the in-process reference model
are scored by identical code.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .model import Job, Observation

SECONDS_PER_HOUR = 3600.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency for two numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


@dataclass
class Metrics:
    config: str
    measured_on_cluster: bool

    jobs: int
    pods: int
    #: Jobs whose every pod was scheduled before the horizon.
    jobs_completed: int

    gpu_hours_used: float
    gpu_hours_idle: float
    utilization: float
    #: Wall-clock (simulated) to drain the whole trace. Reported explicitly
    #: because utilization is *derived* from it: when every job completes,
    #: delivered GPU-hours are fixed by the trace, so utilization is only a
    #: restatement of makespan and quoting it alone hides that.
    makespan_hours: float

    mean_wait: float
    p95_wait: float

    #: Share of *free* GPU-time that no pending pod could have used.
    fragmentation_rate: float
    #: The same quantity against total fleet GPU-time, for readers who prefer it.
    fragmentation_of_fleet: float
    stranded_gpu_hours: float
    #: Definition B: queue-independent. Free GPU-time on nodes too small to host
    #: a pod of ``reference_request`` GPUs. Reported alongside A because the two
    #: disagree, and the disagreement is diagnostic rather than noise.
    fragmentation_ref: float
    reference_request: int

    gang_jobs: int
    gang_stalled: int
    gang_deadlocked: int
    gang_deadlock_rate: float
    gang_wasted_gpu_hours: float

    fairness_ratio: float
    service_ratio: dict[str, float] = field(default_factory=dict)

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("jobs / pods", f"{self.jobs} / {self.pods}"),
            ("jobs completed", str(self.jobs_completed)),
            ("makespan", f"{self.makespan_hours:.1f} h"),
            ("GPU-hours used", f"{self.gpu_hours_used:.1f}"),
            ("GPU-hours idle", f"{self.gpu_hours_idle:.1f}"),
            ("utilization", f"{self.utilization * 100:.1f} %"),
            ("mean wait", f"{self.mean_wait / 60:.1f} min"),
            ("p95 wait", f"{self.p95_wait / 60:.1f} min"),
            ("frag A (of free)", f"{self.fragmentation_rate * 100:.1f} %"),
            ("frag A (of fleet)", f"{self.fragmentation_of_fleet * 100:.1f} %"),
            (f"frag B (ref {self.reference_request}gpu)", f"{self.fragmentation_ref * 100:.1f} %"),
            ("stranded GPU-hours", f"{self.stranded_gpu_hours:.1f}"),
            ("gang jobs", str(self.gang_jobs)),
            ("gang stalled", str(self.gang_stalled)),
            ("gang deadlocked", str(self.gang_deadlocked)),
            ("gang deadlock rate", f"{self.gang_deadlock_rate * 100:.1f} %"),
            ("gang wasted GPU-hours", f"{self.gang_wasted_gpu_hours:.1f}"),
            ("fairness (max/min)", f"{self.fairness_ratio:.2f}"),
        ]

    def format(self) -> str:
        flag = "" if self.measured_on_cluster else "   [reference model, not a cluster run]"
        lines = [f"{self.config}{flag}"]
        lines += [f"  {k:<26} {v:>12}" for k, v in self.rows()]
        if self.service_ratio:
            lines.append("  service ratio by account")
            for acct, r in sorted(self.service_ratio.items()):
                lines.append(f"    {acct:<22} {r:>10.2f}")
        return "\n".join(lines)


def _stranded_at(free: dict[str, int], min_pending_request: int | None) -> int:
    """GPUs that are free but cannot satisfy the smallest pending pod.

    ``None`` means nothing is pending: idle capacity nobody wants is idle, not
    fragmented, and counting it as fragmentation would make an empty cluster
    look maximally fragmented.
    """
    if min_pending_request is None:
        return 0
    return sum(f for f in free.values() if 0 < f < min_pending_request)


def compute(obs: Observation, deadlock_threshold: float = 60.0) -> Metrics:
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
        pending = _min_pending_request(obs, jobs_by_id, t0)
        stranded_seconds += _stranded_at(free0, pending) * dt
        stranded_ref_seconds += _stranded_at(free0, reference) * dt
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

    return Metrics(
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
        p95_wait=percentile(waits, 95),
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


def _min_pending_request(
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
