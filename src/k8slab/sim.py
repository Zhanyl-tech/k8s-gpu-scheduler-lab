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

Phase 2 execution layer. :class:`~k8slab.model.RunInfo` switches on, one at a
time and each defaulting to Phase 1 behaviour:

* ``queue_model="kube"`` -- the binder goes through kube-scheduler's queue
  mechanics (:mod:`k8slab.queueing`), with backoff in simulated seconds derived
  from ``speedup`` exactly as the real scheduler experiences it;
* ``startup_delay_ms`` -- a seeded bind-to-Running gap; trace runtime counts
  from Running and the gap is held-but-idle GPU time;
* ``topology_penalty="extend"`` -- a SCENARIO: when a job's last pod binds, its
  members' remaining runtime is stretched by the ASSUMED placement factor of
  every member's node (members that already finished included; they have no
  remaining runtime to stretch);
* D-preempt -- evictions hold the victims' GPUs for ``grace_seconds`` and
  requeue the victims with their work lost (bar ``checkpoint_fraction``);
* ``admission_gate`` -- a DIAGNOSTIC: one pending pod at a time.

With every setting at its default the model is bit-identical to Phase 1
(``tests/test_sim_equivalence.py`` keeps the Phase 1 code and checks).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .baselines import SPECS, PendingPod
from .binder import Binder
from .execution import Attempt, StartupDelay
from .model import Eviction, Fleet, Job, Observation, PodEvent, RunInfo
from .preemption import RunningPod
from .queueing import QueueParams
from .timescale import TimeScale
from .topology import DEFAULT_FACTORS, JobPlacement, PenaltyFactors, derive, placement_factor

#: Simulated seconds between scheduling passes. Finer than the runner, which
#: polls every TICK *real* seconds -- ``speedup`` x 1 s = 60 simulated seconds
#: at the default speedup. Kept at Phase 1's value so Phase 1 results
#: reproduce; the granularity difference is recorded in docs/limitations.md.
TICK = 5.0

Key = tuple[int, int]


@dataclass(frozen=True)
class EvictionRequest:
    """The reference model's eviction API: evict ``pod`` at ``time``.

    Applied at the first scheduling pass at or after ``time`` (the model is
    pass-granular), and only if the pod is bound then; otherwise it is ignored
    and a note says so. "Bound" means still holding its GPUs: requests are
    applied after the pass has released pods whose work ended by then, so an
    attempt that finished between two passes completes, and is never evicted
    after the fact (which charged Running time beyond its duration as lost and
    ran it again). Evicting one member of a gang evicts every bound member.
    Each evicted attempt keeps its GPUs locked for ``grace_seconds``, then
    releases them, and is requeued with its work lost except for
    ``checkpoint_fraction``. D-preempt's evictions go through the same code.
    """

    pod: Key
    time: float
    reason: str = "requested"


def run(
    fleet: Fleet,
    jobs: list[Job],
    config: str,
    seed: int = 0,
    max_horizon: float = 48 * 3600.0,
    *,
    run_info: RunInfo | None = None,
    factors: PenaltyFactors = DEFAULT_FACTORS,
    evictions: Sequence[EvictionRequest] = (),
) -> Observation:
    """Replay ``jobs`` on ``fleet`` under degenerate policy ``config``.

    ``seed`` seeds the policy's RNG (D-random's choices) and, separately, the
    startup-delay draws. ``run_info`` defaults to Phase 1's harness; its
    ``harness_seed`` is recorded as ``seed``. ``factors`` are the ASSUMED
    penalty factors ``topology_penalty="extend"`` stretches runtimes by.
    ``evictions`` are externally requested evictions (:class:`EvictionRequest`).
    """
    if config not in SPECS:
        raise KeyError(f"unknown policy {config!r}; have {sorted(SPECS)}")
    info = dataclasses.replace(run_info or RunInfo(), harness_seed=seed)
    rng = random.Random(seed)
    # A separate stream, so switching the delay on never perturbs the policy.
    startup_rng = random.Random(f"k8slab-startup|{seed}")
    delay = StartupDelay(*info.startup_delay_ms)
    speedup = info.speedup if info.speedup is not None else 1.0
    params = (
        QueueParams.from_timescale(TimeScale(speedup)) if info.queue_model == "kube" else None
    )
    binder = Binder(config, info, rng, params, window=TICK)
    preemptive = SPECS[config].preemptive
    extend = info.topology_penalty == "extend"
    topo = derive(fleet) if extend else None

    free: dict[str, int] = {n: fleet.gpus_of(n) for n in fleet.node_names()}
    pods: dict[Key, PodEvent] = {}
    pending: dict[Key, PendingPod] = {}
    running: list[Attempt] = []  # bound attempts, started or not, in bind order
    current: dict[Key, Attempt] = {}
    locks: list[tuple[float, str, int]] = []  # (release time, node, GPUs)
    evicted: list[Eviction] = []
    requests = sorted(evictions, key=lambda r: r.time)
    notes: list[str] = []
    work_left: dict[Key, float] = {}
    by_id = {j.job_id: j for j in jobs}
    extension = 0.0
    extension_by_account: dict[str, float] = {}

    def stretched(key: Key, gpu_seconds: float) -> None:
        """Book topology extension, in total and against the job's account
        (fairness takes each account's share back out of its delivered time)."""
        nonlocal extension
        extension += gpu_seconds
        account = by_id[key[0]].account
        extension_by_account[account] = extension_by_account.get(account, 0.0) + gpu_seconds

    for job in jobs:
        for i in range(job.gang_size):
            key = (job.job_id, i)
            pods[key] = PodEvent(job_id=job.job_id, pod_index=i)
            work_left[key] = job.duration

    samples: list[tuple[float, dict[str, int]]] = []
    t = 0.0
    last_submit = max(j.submit_time for j in jobs)

    def evict(a: Attempt, now: float, reason: str) -> None:
        running.remove(a)
        current.pop(a.key, None)
        release = now + info.grace_seconds
        locks.append((release, a.node, a.gpus))
        started = a.start_time is not None and a.start_time < now
        run_wall = now - a.start_time if started and a.start_time is not None else 0.0
        c = info.checkpoint_fraction
        retained_wall = c * run_wall
        retained_work = c * a.work_done(now)
        if a.stretched and c > 0:
            stretched(a.key, a.gpus * (retained_wall - retained_work))
        work_left[a.key] = a.work - retained_work
        evicted.append(
            Eviction(
                job_id=a.key[0], pod_index=a.key[1], node=a.node, gpus=a.gpus,
                bind_time=a.bind_time, start_time=a.start_time if started else None,
                evict_time=now, release_time=release, reason=reason,
                lost_seconds=run_wall - retained_wall, retained_seconds=retained_wall,
            )
        )
        ev = pods[a.key]
        ev.scheduled_time = ev.start_time = ev.end_time = None
        ev.node = None

    def evict_job(key: Key, now: float, reason: str) -> bool:
        """Evict every bound attempt of ``key``'s job (a gang goes whole)."""
        job = by_id[key[0]]
        members = [current.get((job.job_id, i)) for i in range(job.gang_size)]
        bound = [m for m in members if m is not None]
        for m in bound:
            evict(m, now, reason)
        return bool(bound)

    while t < max_horizon:
        # Release finished pods, then expired grace locks.
        freed = False
        for a in [a for a in running if a.end is not None and a.end <= t]:
            end = a.end
            assert end is not None
            free[a.node] += a.gpus
            pods[a.key].end_time = end
            if a.stretched:
                stretched(a.key, a.gpus * a.stretch_seconds(end))
            current.pop(a.key, None)
            freed = True
        running = [a for a in running if not (a.end is not None and a.end <= t)]
        if locks:
            for _release, node, gpus in [lk for lk in locks if lk[0] <= t]:
                free[node] += gpus
                freed = True
            locks = [lk for lk in locks if lk[0] > t]
        if freed:
            binder.capacity_freed(t)

        # Requested evictions due by now -- AFTER the release above, so an
        # attempt whose work ended by t has completed and cannot be evicted.
        # (They used to run first: a 12 s job with an eviction requested at
        # t=15 was evicted with 15 s "lost" and ran again.) D-preempt's own
        # evictions are chosen in the bind pass below, also after the release.
        while requests and requests[0].time <= t:
            req = requests.pop(0)
            # The named pod itself must be bound, as documented; a finished
            # member must not drag its still-running siblings out with it.
            if req.pod not in current or not evict_job(req.pod, t, req.reason):
                done = pods[req.pod].end_time is not None
                notes.append(
                    f"eviction of j{req.pod[0]}-p{req.pod[1]} requested at {req.time:g} "
                    + (f"ignored: its work finished at {pods[req.pod].end_time:g}" if done
                       else f"ignored: not bound at t={t:g}")
                )

        # Admit newly-submitted pods (and requeued evicted ones).
        backlog = False
        for job in jobs:
            if job.submit_time <= t:
                for i in range(job.gang_size):
                    key = (job.job_id, i)
                    if pods[key].scheduled_time is None and key not in pending:
                        if info.admission_gate and pending:
                            backlog = True
                            continue
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
            view = (
                [RunningPod(a.key, a.node, a.gpus, a.priority) for a in running]
                if preemptive else []
            )
            decision = binder.schedule(t, list(pending.values()), free, view)
            for plan in decision.preemptions:
                reason = f"preempted by j{plan.preemptor[0]}-p{plan.preemptor[1]}"
                for victim in plan.victims:
                    victim_attempt = current.get(victim)
                    if victim_attempt is not None:
                        evict(victim_attempt, t, reason)
            touched: list[int] = []
            for pod, node in decision.binds:
                job = by_id[pod.job_id]
                free[node] -= pod.gpus
                ev = pods[pod.key]
                ev.scheduled_time = t
                attempt = Attempt(
                    key=pod.key, node=node, gpus=pod.gpus, priority=job.priority,
                    bind_time=t, work=work_left[pod.key],
                )
                start = t + delay.sample_sim(startup_rng, speedup)
                attempt.start(start)
                ev.start_time = start
                ev.node = node
                running.append(attempt)
                current[pod.key] = attempt
                pending.pop(pod.key, None)
                if job.job_id not in touched:
                    touched.append(job.job_id)
            if extend and topo is not None:
                for job_id in touched:
                    job = by_id[job_id]
                    # Every member's node counts, including a member that has
                    # already FINISHED (its PodEvent keeps its node): the
                    # factor is the job's placement factor, the one
                    # placement_penalty_mean reports. Requiring every member
                    # to be still bound exempted exactly the gangs a
                    # scheduler failed to co-schedule -- D-random most of all
                    # -- so the worst gang schedulers paid the least penalty.
                    # A member evicted and not yet re-bound has no node, so
                    # the job waits for it. Only the remaining work of members
                    # still bound is stretched; a finished one is left alone.
                    nodes = [pods[(job_id, i)].node for i in range(job.gang_size)]
                    placed = tuple(n for n in nodes if n is not None)
                    if len(placed) != job.gang_size:
                        continue
                    f = placement_factor(JobPlacement(placed, job.gpus), topo, factors)
                    for i in range(job.gang_size):
                        member = current.get((job_id, i))
                        if member is not None:
                            member.set_factor(f, t)

        if not pending and not running and not backlog and t > last_submit:
            break
        t += TICK

    horizon = t
    # A request the run never reached must not stretch the horizon (it would
    # dilute utilization); it is recorded instead.
    notes += [
        f"eviction of j{r.pod[0]}-p{r.pod[1]} requested at {r.time:g} ignored: "
        f"the run ended at {horizon:g}"
        for r in requests
    ]
    for a in running:
        ev = pods[a.key]
        if a.start_time is not None and a.start_time > horizon:
            ev.start_time = None  # bound, never Running before the horizon
        elif a.stretched:
            stretched(a.key, a.gpus * a.stretch_seconds(horizon))
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
        run=info,
        evictions=evicted,
        extension_gpu_seconds=extension,
        extension_by_account=extension_by_account,
        notes=notes,
    )
