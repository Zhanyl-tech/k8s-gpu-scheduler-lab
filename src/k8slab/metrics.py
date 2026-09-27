"""Metrics. Every definition here is also written out in docs/metrics.md.

Read that file before quoting a number from this one. Fragmentation in
particular has no canonical definition in the literature, and the published
claims that motivated this repo do not state theirs — so a number is only
comparable to another number computed the same way.

Everything is computed from an :class:`~k8slab.model.Observation`, so a run
against a real control plane and a run against the in-process reference model
are scored by identical code.

Cost. Every pass here is linear in pods, jobs or samples, except the pending-
queue query behind fragmentation A, which is a single sorted sweep
(O((P + S) log P) for P pods and S samples). It used to rescan every pod at
every sample -- O(S x P), about 12 million iterations for one default-profile
reference-model run -- and repeated runs multiply that. The rewrite is
bit-identical on every metric it preserves; ``tests/test_metrics_equivalence.py``
keeps the old implementation verbatim and checks.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from . import topology as topology_mod
from .fragmentation import structural_fragmentation
from .model import Eviction, Job, Observation, PodEvent
from .topology import DEFAULT_FACTORS, JobPlacement, PenaltyFactors

SECONDS_PER_HOUR = 3600.0

#: Job footprint buckets for the admission-delay breakdown, by TOTAL GPUs
#: (per-pod GPUs x gang size): ``(label, low, high)``, ``high=None`` = open.
FOOTPRINT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17+", 17, None),
)


def percentile(values: list[float], pct: float) -> float:
    """Phase 1's percentile, kept only for ``p95_wait``. NOT nearest-rank.

    It takes the value of rank ``round(pct/100 * n)`` (clamped to ``[1, n]``),
    where ``round`` is Python's round-half-to-even. That differs from
    nearest-rank (rank ``ceil(pct/100 * n)``, :func:`nearest_rank`) whenever
    ``pct/100 * n`` has a fractional part strictly between 0 and one half, or
    exactly one half that rounds down to even -- e.g. the median of
    ``[1, 2, 3, 4, 5]`` comes out as 2, and the p95 of eleven values as the
    10th. It was called "nearest-rank" until that was measured. ``p95_wait``
    keeps it so every Phase 1 number stays bit-identical
    (``tests/test_metrics_equivalence.py``); every metric added since uses
    :func:`nearest_rank`.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


def nearest_rank(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value with at least ``pct``% of
    the values at or below it, i.e. the value of rank ``ceil(pct/100 * n)``.

    ``pct * n / 100`` is computed in that order so an exact rank (e.g. 7% of
    100) is not pushed past an integer by the float error of ``0.07 * 100``.
    ``ValueError`` on an empty sequence: a percentile of nothing is undefined,
    and every caller here reports ``None`` for that case instead.
    """
    if not values:
        raise ValueError("nearest_rank of an empty sequence is undefined")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct * len(ordered) / 100.0))
    return ordered[rank - 1]


def footprint_bucket(total_gpus: int) -> str | None:
    """The :data:`FOOTPRINT_BUCKETS` label for a job; ``None`` for zero GPUs."""
    for label, low, high in FOOTPRINT_BUCKETS:
        if total_gpus >= low and (high is None or total_gpus <= high):
            return label
    return None


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties given the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation: Pearson correlation of tie-averaged ranks.

    ``None`` when undefined -- fewer than two points, or no variation in either
    variable (every job the same size, or every wait identical). Stdlib only:
    ``statistics.correlation(method="ranked")`` would do this but needs 3.12,
    and the lab supports 3.11.
    """
    if len(xs) != len(ys):
        raise ValueError("spearman needs paired samples")
    if len(xs) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = math.fsum((a - mx) ** 2 for a in rx)
    syy = math.fsum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def trace_digest(jobs: Iterable[Job]) -> str:
    """Short, stable fingerprint of a trace. Two runs are repeats of one
    experiment only if they replayed the same trace, and this is how a results
    file proves it did."""
    h = hashlib.sha256()
    for j in jobs:
        h.update(
            f"{j.job_id},{j.account},{j.submit_time!r},{j.duration!r},"
            f"{j.gpus},{j.gang_size},{j.priority}\n".encode()
        )
    return h.hexdigest()[:12]


@dataclass
class Metrics:
    """One run, scored. ``None`` on any field means *undefined for this run*
    (for example no gang ever assembled), never zero."""

    config: str
    measured_on_cluster: bool
    #: Fleet name and trace fingerprint. Repeats are aggregated per config;
    #: these make it checkable that the repeats really replayed one experiment.
    fleet: str
    trace_digest: str

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

    #: Wait = time a job spent pending, to its last pod's bind (docs/metrics.md).
    #: For a job with evicted-and-requeued attempts, only the pending time
    #: counts -- not the time an evicted attempt was bound or Running.
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
    #: Definition C at the node level -- the headline C. Share of free
    #: GPU-time on nodes that are partly allocated (not whole-idle). Pure
    #: function of capacity, domains and samples: :mod:`k8slab.fragmentation`.
    fragmentation_structural: float
    #: Definition C at the rack and switch levels. Depend on the DECLARED
    #: topology; see docs/metrics.md for why node is the headline.
    fragmentation_structural_rack: float
    fragmentation_structural_switch: float

    gang_jobs: int
    #: DEPRECATED with ``gang_deadlocked`` and ``gang_deadlock_rate``: a
    #: threshold-gated binary kept for backwards compatibility in results.json
    #: only. Superseded by the continuous ``gang_stranded_*`` metrics.
    gang_stalled: int
    gang_deadlocked: int
    gang_deadlock_rate: float
    #: Gangs whose every member started running.
    gang_assembled: int
    #: GPU-hours held by gang members while their gang was not assembled
    #: (docs/metrics.md), over every attempt: the final one and every evicted
    #: and requeued one. Replaces Phase 1's ``gang_wasted_gpu_hours``.
    gang_stranded_gpu_hours: float
    #: ``gang_stranded_gpu_hours / gpu_hours_used``.
    gang_stranded_share: float
    #: Assembly delay (last member Running minus first member bound), seconds,
    #: over gangs whose final attempt assembled; nearest-rank. ``None`` when
    #: no gang assembled.
    gang_assembly_p50: float | None
    gang_assembly_p95: float | None

    #: Spearman rank correlation of total job GPUs against admission wait,
    #: over admitted jobs. Positive: big jobs wait longer.
    footprint_wait_spearman: float | None
    #: The fleet's largest single-node GPU capacity. A job needing at least
    #: this many GPUs in total needs at least one whole largest node -- it can
    #: only start once such a node (or several nodes) has drained. That, not
    #: "cannot fit on one node" (a job of exactly this size fits one empty
    #: largest node), is what makes it "large" for the starvation ratio.
    large_job_gpus: int
    #: mean wait(total GPUs >= large_job_gpus) / mean wait(1-GPU jobs).
    large_job_starvation_ratio: float | None
    #: Per footprint bucket: ``jobs``, ``admitted`` (every pod bound),
    #: ``mean`` and nearest-rank ``p95`` wait in seconds over admitted jobs
    #: (``None`` if none).
    wait_by_footprint: dict[str, dict[str, float | None]]

    #: False when the fleet declared no topology and the tiers use defaults.
    topology_declared: bool
    #: Fully placed multi-pod jobs; the denominator of the tier shares.
    placement_multi_pod_jobs: int
    #: Share of those jobs whose widest span is node / rack / switch /
    #: cross-switch. A measurement of placement decisions. Every tier is
    #: ``None`` when there were no such jobs (0/0 is undefined, not 0%).
    placement_tier_share: dict[str, float | None]
    #: GPU-weighted mean ASSUMED penalty factor over placed jobs with more than
    #: one GPU. Inherits the scenario's factors; not a measured slowdown.
    placement_penalty_mean: float | None
    #: The ASSUMED factors ``placement_penalty_mean`` was computed under, so a
    #: results file always carries the premise next to the number.
    penalty_factors: dict[str, float]

    fairness_ratio: float
    service_ratio: dict[str, float] = field(default_factory=dict)

    # ---- Phase 2 execution layer (defaults describe a Phase 1 run) ----------

    #: The harness settings this run was produced under
    #: (:meth:`k8slab.model.RunInfo.harness`). Runs are pooled only when equal.
    harness: dict[str, Any] = field(default_factory=dict)
    #: Seed of the binder and startup-delay RNGs. Identity, varies by repeat.
    harness_seed: int | None = None
    #: GPU-hours the trace asks for: sum of total GPUs x duration.
    gpu_hours_demanded: float = 0.0
    #: GPU-hours held between bind and Running (startup delay): allocated,
    #: doing nothing. Includes evicted attempts' bind-to-Running time.
    startup_overhead_gpu_hours: float = 0.0
    #: Pod attempts evicted before they finished (gang members count each).
    preemptions: int = 0
    #: GPU-hours of Running thrown away by evictions (all of it unless a
    #: checkpoint fraction was set).
    preempted_gpu_hours_lost: float = 0.0
    #: GPU-hours locked by evicted pods during their grace period.
    grace_locked_gpu_hours: float = 0.0
    #: GPU-hours of delivered time that exist only because runtimes were
    #: stretched by ASSUMED topology factors. 0 unless topology_penalty=extend.
    topology_extension_gpu_hours: float = 0.0
    #: The execution layer's notes about this run (e.g. a grace lock the
    #: control plane did not honour). Carried into results.json verbatim.
    notes: list[str] = field(default_factory=list)

    #: Numeric fields that identify a run rather than measure it.
    _IDENTITY = frozenset({"harness_seed"})

    @property
    def scenario(self) -> bool:
        """Produced under ``topology_penalty=extend``: a scenario, not a measurement."""
        return bool(self.harness.get("topology_penalty") == "extend")

    @property
    def gated(self) -> bool:
        """Produced under a diagnostic gate (admission gate or serialise)."""
        return bool(self.harness.get("admission_gate") or self.harness.get("serialise"))

    def scalars(self) -> dict[str, float | None]:
        """Every numeric quantity as a flat ``{name: value}`` map.

        Nested breakdowns are flattened with dotted keys
        (``wait_by_footprint.3-4.mean``, ``placement_tier_share.rack``,
        ``service_ratio.research``) so repeated runs can be aggregated
        uniformly by :mod:`k8slab.stats`. Booleans and strings are identity,
        not measurements, and are left out.
        """
        out: dict[str, float | None] = {}
        for f in fields(self):
            if f.name in self._IDENTITY:
                continue
            v = getattr(self, f.name)
            if isinstance(v, bool) or isinstance(v, str):
                continue
            if isinstance(v, (int, float)):
                out[f.name] = float(v)
            elif v is None:
                out[f.name] = None
        for bucket, stats in self.wait_by_footprint.items():
            for key, value in stats.items():
                out[f"wait_by_footprint.{bucket}.{key}"] = value
        for tier, share in self.placement_tier_share.items():
            out[f"placement_tier_share.{tier}"] = share
        for acct, ratio in self.service_ratio.items():
            out[f"service_ratio.{acct}"] = ratio
        return out


def _stranded_at(free: dict[str, int], min_pending_request: int | None) -> int:
    """GPUs that are free but cannot satisfy the smallest pending pod.

    ``None`` means nothing is pending: idle capacity nobody wants is idle, not
    fragmented, and counting it as fragmentation would make an empty cluster
    look maximally fragmented.
    """
    if min_pending_request is None:
        return 0
    return sum(f for f in free.values() if 0 < f < min_pending_request)


def _pods_by_job(obs: Observation) -> dict[int, list[PodEvent]]:
    """Group pods once, preserving observation order (as ``pods_of`` does)."""
    out: dict[int, list[PodEvent]] = {}
    for pod in obs.pods:
        out.setdefault(pod.job_id, []).append(pod)
    return out


Key = tuple[int, int]


def _requeued_attempts(obs: Observation) -> dict[Key, list[tuple[float, float]]]:
    """``(bind, evict)`` of every evicted-and-requeued attempt, per pod."""
    attempts: dict[Key, list[tuple[float, float]]] = {}
    for e in obs.evictions:
        if e.requeued:
            attempts.setdefault((e.job_id, e.pod_index), []).append((e.bind_time, e.evict_time))
    return attempts


def _pending_intervals(
    submit: float, earlier: list[tuple[float, float]] | None, scheduled: float | None
) -> list[tuple[float, float | None]]:
    """The half-open intervals during which one pod was pending.

    ``[submit, b_1), [e_1, b_2), ..., [e_k, scheduled)`` for a pod whose
    earlier attempts were bound at ``b_i`` and evicted-and-requeued at
    ``e_i``; just ``[submit, scheduled)`` without evictions. Empty intervals
    are dropped. The last interval is open (``None``) for a pod never bound.
    Shared by Definition A and by wait, so the two agree on what "pending"
    means.
    """
    out: list[tuple[float, float | None]] = []
    opened = submit
    if earlier:
        for bind, evict in sorted(earlier):
            if bind > opened:  # an empty interval adds nothing
                out.append((opened, bind))
            opened = max(opened, evict)
    if scheduled is not None and scheduled <= opened:
        return out  # empty: bound no later than it became pending
    out.append((opened, scheduled))
    return out


def _pending_seconds(
    job: Job, pods: Sequence[PodEvent], requeued: dict[Key, list[tuple[float, float]]]
) -> float:
    """Seconds during which at least one pod of ``job`` was pending: the length
    of the union of its pods' pending intervals. Every pod must be bound."""
    spans = sorted(
        (start, end)
        for p in pods
        for start, end in _pending_intervals(
            job.submit_time, requeued.get((p.job_id, p.pod_index)), p.scheduled_time
        )
        if end is not None
    )
    pieces: list[float] = []
    lo: float | None = None
    hi = 0.0
    for start, end in spans:
        if lo is None or start > hi:
            if lo is not None:
                pieces.append(hi - lo)
            lo, hi = start, end
        else:
            hi = max(hi, end)
    if lo is not None:
        pieces.append(hi - lo)
    return math.fsum(pieces)


def _pending_until(pod: PodEvent) -> float | None:
    """When the final attempt of ``pod`` stopped being pending; ``None`` if it
    was still pending at the horizon.

    Its bind, normally. A pod the control plane destroyed before any bind was
    observed (the runner's ``_destroyed`` with no bound attempt: Terminating
    before its binding was seen, or vanished never seen bound) has no bind,
    but it left the queue at its destruction, ``end_time``. Passing ``None``
    for it kept its request in m(t) until the horizon: on one 8-GPU node with
    2 GPUs free, a 4-GPU pod destroyed at 10 stranded those 2 GPUs for the
    whole run (200 GPU-s over 100 s) instead of for 10 s (20 GPU-s).
    """
    if pod.scheduled_time is not None:
        return pod.scheduled_time
    if pod.preempted:
        return pod.end_time
    return None


def _min_pending_at(
    obs: Observation, jobs_by_id: dict[int, Job], times: Iterable[float]
) -> dict[float, int | None]:
    """Smallest per-pod GPU request among pods pending at each of ``times``.

    A pod is pending at ``t`` when it has been submitted and is not bound at
    ``t``; zero-GPU pods never count. Without evictions that is the half-open
    interval ``[submit, scheduled)``. A pod evicted and requeued (a requeued
    :class:`~k8slab.model.Eviction`) was bound -- holding GPUs -- during each
    earlier attempt, so it is pending on ``[submit, bind_1)``, then
    ``[evict_1, bind_2)``, ..., ``[evict_k, scheduled)``: the final
    ``PodEvent`` describes only the last attempt, and treating the whole
    ``[submit, scheduled)`` as pending counted every earlier attempt's bound
    time as pending. A pod the control plane destroyed and nothing recreated
    (``PodEvent.preempted``) is its own final attempt and is never pending
    again: its interval ends at its bind if it was bound, and at its
    destruction (``end_time``) if it was not -- see :func:`_pending_until`.

    The query is a sweep: sort interval starts and ends once, walk the query
    times in order, and keep a count per request size with a lazy min-heap over
    the sizes present. With no requeued eviction every pod contributes exactly
    the one interval the O(S x P) scan this replaced used, applied to the same
    floats, so the answers are identical.
    """
    attempts = _requeued_attempts(obs)
    starts: list[tuple[float, int]] = []
    ends: list[tuple[float, int]] = []
    for pod in obs.pods:
        job = jobs_by_id[pod.job_id]
        g = job.gpus
        if g <= 0:
            continue
        for opened, closed in _pending_intervals(
            job.submit_time, attempts.get((pod.job_id, pod.pod_index)), _pending_until(pod)
        ):
            starts.append((opened, g))
            if closed is not None:
                ends.append((closed, g))
    starts.sort()
    ends.sort()

    counts: dict[int, int] = {}
    heap: list[int] = []
    out: dict[float, int | None] = {}
    si = ei = 0
    for t in sorted(set(times)):
        while si < len(starts) and starts[si][0] <= t:
            g = starts[si][1]
            if counts.get(g, 0) == 0:
                heapq.heappush(heap, g)
            counts[g] = counts.get(g, 0) + 1
            si += 1
        # Every end is strictly after its own start, so an end <= t always
        # belongs to a start already applied: counts never go negative.
        while ei < len(ends) and ends[ei][0] <= t:
            counts[ends[ei][1]] -= 1
            ei += 1
        while heap and counts[heap[0]] == 0:
            heapq.heappop(heap)
        out[t] = heap[0] if heap else None
    return out


def _gang_stranding(
    job: Job, pods: list[PodEvent], horizon: float
) -> tuple[float, float | None]:
    """``(stranded GPU-seconds, assembly delay)`` for one gang.

    A member holds its GPUs from bind (``scheduled_time``) until it ends (or
    the horizon). The gang is assembled at ``A`` = the latest member
    ``start_time``, once every member has started. Stranded time is held time
    before ``A``, capped at each member's own end -- a member that finished
    before the last one started was stranded for its whole life. A gang that
    never assembles is stranded for all of every member's held time. Assembly
    delay is ``A`` minus the first bind, and ``None`` when never assembled.
    """
    held: list[tuple[float, float]] = []
    for p in pods:
        bind = p.scheduled_time if p.scheduled_time is not None else p.start_time
        if bind is None:
            continue
        end = p.end_time if p.end_time is not None else horizon
        held.append((bind, min(end, horizon)))
    if not held:
        return 0.0, None
    starts = [p.start_time for p in pods]
    if len(pods) == job.gang_size and all(s is not None for s in starts):
        assembled_at = max(s for s in starts if s is not None)
        stranded = math.fsum(
            job.gpus * max(0.0, min(end, assembled_at) - bind) for bind, end in held
        )
        return stranded, assembled_at - min(bind for bind, _ in held)
    return math.fsum(job.gpus * max(0.0, end - bind) for bind, end in held), None


def _evicted_gang_stranding(job: Job, attempt: list[Eviction]) -> float:
    """Stranded GPU-seconds of one evicted attempt of a gang.

    ``attempt`` is every member evicted at one instant -- evictions take a
    gang's bound members together (``sim.evict_job``, the D-preempt plan), so
    ``(job, evict_time)`` identifies one attempt. The rule is the final
    attempt's (:func:`_gang_stranding`) with the eviction as each member's
    end: if every member was bound and had started before the eviction, the
    attempt assembled at ``A`` = its latest start and is stranded before ``A``;
    otherwise it never assembled and every bound member was stranded from its
    bind until the eviction.
    """
    starts = [e.start_time for e in attempt]
    assembled = len(attempt) == job.gang_size and all(s is not None for s in starts)
    if assembled:
        a = max(s for s in starts if s is not None)
        return math.fsum(
            job.gpus * max(0.0, min(e.evict_time, a) - e.bind_time) for e in attempt
        )
    return math.fsum(job.gpus * max(0.0, e.evict_time - e.bind_time) for e in attempt)


def compute(
    obs: Observation,
    deadlock_threshold: float = 60.0,
    factors: PenaltyFactors = DEFAULT_FACTORS,
) -> Metrics:
    """Score one run.

    ``deadlock_threshold`` only feeds the deprecated ``gang_deadlock_rate``.
    ``factors`` only feeds ``placement_penalty_mean`` -- ASSUMED scenario
    parameters, see :mod:`k8slab.topology`.
    """
    jobs_by_id: dict[int, Job] = {j.job_id: j for j in obs.jobs}
    pods_by_job = _pods_by_job(obs)
    total_gpus = obs.fleet.total_gpus
    horizon = obs.horizon or max(
        (p.end_time or 0.0 for p in obs.pods),
        default=0.0,
    )
    topo = obs.topology if obs.topology is not None else topology_mod.derive(obs.fleet)

    # ---- delivered GPU-time -------------------------------------------------
    # Pods the control plane destroyed and nothing recreated: their Running
    # time is lost work (preempted_gpu_hours_lost), not delivered, exactly as
    # for an evicted-and-requeued attempt. Empty unless something was evicted,
    # so a run without evictions is scored bit-identically to Phase 1.
    destroyed = {(e.job_id, e.pod_index) for e in obs.evictions if not e.requeued}
    gpu_seconds_used = 0.0
    for pod in obs.pods:
        if pod.start_time is None or pod.end_time is None:
            continue
        if destroyed and (pod.job_id, pod.pod_index) in destroyed:
            continue
        gpu_seconds_used += jobs_by_id[pod.job_id].gpus * (pod.end_time - pod.start_time)
    # Evicted attempts deliver only what a checkpoint kept; the rest is lost
    # work, reported separately.
    retained_by_account: dict[str, float] = {}
    if obs.evictions:
        for e in obs.evictions:
            kept = e.gpus * e.retained_seconds
            gpu_seconds_used += kept
            account = jobs_by_id[e.job_id].account
            retained_by_account[account] = retained_by_account.get(account, 0.0) + kept

    capacity_seconds = total_gpus * horizon
    utilization = gpu_seconds_used / capacity_seconds if capacity_seconds > 0 else 0.0

    # ---- waits --------------------------------------------------------------
    # A job's wait is the time it spent waiting for admission, measured to the
    # moment its LAST pod is bound: a gang that is half-placed has not started,
    # and crediting it with the first binding would flatter every scheduler
    # that admits gangs partially. Without evictions that is simply
    # last bind - submit (Phase 1's formula, kept bit-identical). A job with an
    # evicted-and-requeued attempt waited only while some pod of it was
    # PENDING -- the union of its pods' pending intervals, the ones Definition
    # A uses. Its final PodEvents describe only the last attempt, so "last
    # bind - submit" counted the time earlier attempts were bound and Running
    # as waiting (D-preempt's headline wait and its size-bias metrics).
    requeued = _requeued_attempts(obs) if obs.evictions else {}
    waits: list[float] = []
    wait_of: dict[int, float] = {}
    completed = 0
    for job in obs.jobs:
        pods = pods_by_job.get(job.job_id, [])
        if pods and all(p.scheduled_time is not None for p in pods):
            if requeued and any((p.job_id, p.pod_index) in requeued for p in pods):
                wait = _pending_seconds(job, pods, requeued)
            else:
                last = max(p.scheduled_time or 0.0 for p in pods)
                wait = max(0.0, last - job.submit_time)
            waits.append(wait)
            wait_of[job.job_id] = wait
            completed += 1

    # ---- fragmentation A and B ----------------------------------------------
    # Left-point step integral: each sample holds until the next one.
    stranded_seconds = 0.0
    stranded_ref_seconds = 0.0
    free_seconds = 0.0
    # Definition B's reference request: the largest per-pod ask the trace
    # contains. Fixed for the whole run, so it cannot move with the queue.
    reference = max((j.gpus for j in obs.jobs), default=1) or 1
    samples = obs.gpu_free_samples
    query_times = [
        samples[i][0] for i in range(len(samples) - 1) if samples[i + 1][0] - samples[i][0] > 0
    ]
    min_pending = _min_pending_at(obs, jobs_by_id, query_times)
    for i in range(len(samples) - 1):
        t0, free0 = samples[i]
        t1, _ = samples[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        pending = min_pending[t0]
        stranded_seconds += _stranded_at(free0, pending) * dt
        stranded_ref_seconds += _stranded_at(free0, reference) * dt
        free_seconds += sum(free0.values()) * dt

    frag_of_free = stranded_seconds / free_seconds if free_seconds > 0 else 0.0
    frag_of_fleet = stranded_seconds / capacity_seconds if capacity_seconds > 0 else 0.0
    frag_ref = stranded_ref_seconds / free_seconds if free_seconds > 0 else 0.0

    # ---- fragmentation C ----------------------------------------------------
    frag_c = structural_fragmentation(topo.capacity(), topo.domains(), samples)

    # ---- gang behaviour -----------------------------------------------------
    gang_jobs = [j for j in obs.jobs if j.is_gang]
    stalled = 0
    deadlocked = 0
    assembled = 0
    gang_stranded_seconds = 0.0
    assembly_delays: list[float] = []
    for job in gang_jobs:
        pods = pods_by_job.get(job.job_id, [])
        # Deprecated binary classification, unchanged from Phase 1.
        placed = [p.scheduled_time for p in pods if p.scheduled_time is not None]
        if placed:
            if len(placed) < len(pods):
                deadlocked += 1
            elif max(placed) - min(placed) > deadlock_threshold:
                stalled += 1
        stranded, delay = _gang_stranding(job, pods, horizon)
        gang_stranded_seconds += stranded
        if delay is not None:
            assembled += 1
            assembly_delays.append(delay)
    # Evicted-and-requeued attempts held GPUs too, and are invisible in the
    # final PodEvents (which describe the last attempt only). Counting only
    # the final attempt understated D-preempt's stranding. Destroyed pods
    # (requeued=False) ARE their final PodEvent and are already counted above.
    if obs.evictions:
        evicted_attempts: dict[tuple[int, float], list[Eviction]] = {}
        for e in obs.evictions:
            if e.requeued and jobs_by_id[e.job_id].is_gang:
                evicted_attempts.setdefault((e.job_id, e.evict_time), []).append(e)
        for (job_id, _), members in evicted_attempts.items():
            gang_stranded_seconds += _evicted_gang_stranding(jobs_by_id[job_id], members)

    deadlock_rate = (deadlocked + stalled) / len(gang_jobs) if gang_jobs else 0.0

    # ---- size bias in admission (head-of-line blocking / starvation) --------
    bucket_jobs = {label: 0 for label, _, _ in FOOTPRINT_BUCKETS}
    bucket_waits: dict[str, list[float]] = {label: [] for label, _, _ in FOOTPRINT_BUCKETS}
    footprints: list[float] = []
    admitted_waits: list[float] = []
    large = obs.fleet.max_node_gpus
    large_waits: list[float] = []
    single_waits: list[float] = []
    for job in obs.jobs:
        bucket = footprint_bucket(job.total_gpus)
        if bucket is None:
            continue  # a zero-GPU job does not compete for GPUs
        bucket_jobs[bucket] += 1
        w = wait_of.get(job.job_id)
        if w is None:
            continue
        bucket_waits[bucket].append(w)
        footprints.append(float(job.total_gpus))
        admitted_waits.append(w)
        if job.total_gpus >= large:
            large_waits.append(w)
        if job.total_gpus == 1:
            single_waits.append(w)
    wait_by_footprint: dict[str, dict[str, float | None]] = {
        label: {
            "jobs": float(bucket_jobs[label]),
            "admitted": float(len(bucket_waits[label])),
            "mean": statistics.fmean(bucket_waits[label]) if bucket_waits[label] else None,
            "p95": nearest_rank(bucket_waits[label], 95) if bucket_waits[label] else None,
        }
        for label, _, _ in FOOTPRINT_BUCKETS
    }
    starvation: float | None = None
    if large_waits and single_waits:
        single_mean = statistics.fmean(single_waits)
        if single_mean > 0:
            starvation = statistics.fmean(large_waits) / single_mean

    # ---- placement ----------------------------------------------------------
    placements: list[JobPlacement] = []
    for job in obs.jobs:
        pods = sorted(pods_by_job.get(job.job_id, []), key=lambda p: p.pod_index)
        nodes = tuple(p.node for p in pods if p.node is not None)
        if pods and len(nodes) == len(pods):
            placements.append(JobPlacement(nodes=nodes, gpus_per_pod=job.gpus))
    placement = topology_mod.placement_quality(placements, topo, factors)

    # ---- fairness -----------------------------------------------------------
    demanded: dict[str, float] = {}
    delivered: dict[str, float] = {}
    for job in obs.jobs:
        demanded[job.account] = demanded.get(job.account, 0.0) + job.total_gpus * job.duration
    for pod in obs.pods:
        if pod.start_time is None or pod.end_time is None:
            continue
        if destroyed and (pod.job_id, pod.pod_index) in destroyed:
            continue
        job = jobs_by_id[pod.job_id]
        delivered[job.account] = delivered.get(job.account, 0.0) + job.gpus * (
            pod.end_time - pod.start_time
        )
    for account, kept in retained_by_account.items():
        delivered[account] = delivered.get(account, 0.0) + kept
    # V_a is the trace's work delivered to account a. Under
    # topology_penalty=extend part of every stretched pod's Running time exists
    # only because of the ASSUMED factor, so each account's share of the
    # extension comes back out. Counting it read demand plus that account's
    # stretch over demand: ratios above 1.0 with every job complete, and a
    # fairness ratio of 1.08-1.13 in the default extend run that measured
    # which accounts had more multi-GPU cross-switch work -- the premise, not
    # the service. Nothing to subtract (and bit-identical) in any other mode.
    if obs.extension_gpu_seconds:
        split = math.fsum(obs.extension_by_account.values())
        if not math.isclose(split, obs.extension_gpu_seconds, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(
                f"extension_by_account sums to {split} GPU-s but extension_gpu_seconds is "
                f"{obs.extension_gpu_seconds}: fairness cannot separate the stretch from "
                f"the trace's work"
            )
    for account, stretch in obs.extension_by_account.items():
        delivered[account] = delivered.get(account, 0.0) - stretch
    ratios = {
        a: (delivered.get(a, 0.0) / d if d > 0 else 0.0) for a, d in demanded.items()
    }
    positive = [r for r in ratios.values() if r > 0]
    fairness = (max(positive) / min(positive)) if len(positive) > 1 else 1.0

    # ---- execution layer: startup, preemption, topology extension -----------
    # Held-but-idle: bind until Running (or until the pod ended or the horizon,
    # if it never ran). Zero whenever start_time == scheduled_time, as in
    # Phase 1.
    startup_seconds = 0.0
    for pod in obs.pods:
        if pod.scheduled_time is None:
            continue
        stop = pod.start_time
        if stop is None:
            stop = pod.end_time if pod.end_time is not None else horizon
        startup_seconds += jobs_by_id[pod.job_id].gpus * max(
            0.0, min(stop, horizon) - pod.scheduled_time
        )
    lost_seconds = 0.0
    locked_seconds = 0.0
    for e in obs.evictions:
        if e.requeued:  # otherwise the final PodEvent already covered it
            until_running = e.start_time if e.start_time is not None else e.evict_time
            startup_seconds += e.gpus * max(0.0, until_running - e.bind_time)
        lost_seconds += e.gpus * e.lost_seconds
        locked_seconds += e.gpus * max(0.0, min(e.release_time, horizon) - e.evict_time)
    demanded_seconds = math.fsum(j.total_gpus * j.duration for j in obs.jobs)
    penalty_mean = placement.penalty_mean
    if obs.run.topology_penalty == "off":
        penalty_mean = None  # factors neither applied nor reported

    return Metrics(
        config=obs.config,
        measured_on_cluster=obs.measured_on_cluster,
        fleet=obs.fleet.name,
        trace_digest=trace_digest(obs.jobs),
        jobs=len(obs.jobs),
        pods=len(obs.pods),
        jobs_completed=completed,
        gpu_hours_used=gpu_seconds_used / SECONDS_PER_HOUR,
        makespan_hours=horizon / SECONDS_PER_HOUR,
        gpu_hours_idle=max(0.0, capacity_seconds - gpu_seconds_used) / SECONDS_PER_HOUR,
        utilization=utilization,
        mean_wait=statistics.fmean(waits) if waits else 0.0,
        # Phase 1's rounded rank, not nearest-rank: kept bit-identical.
        p95_wait=percentile(waits, 95),
        fragmentation_rate=frag_of_free,
        fragmentation_of_fleet=frag_of_fleet,
        stranded_gpu_hours=stranded_seconds / SECONDS_PER_HOUR,
        fragmentation_ref=frag_ref,
        reference_request=reference,
        fragmentation_structural=frag_c.rate("node"),
        fragmentation_structural_rack=frag_c.rate("rack"),
        fragmentation_structural_switch=frag_c.rate("switch"),
        gang_jobs=len(gang_jobs),
        gang_stalled=stalled,
        gang_deadlocked=deadlocked,
        gang_deadlock_rate=deadlock_rate,
        gang_assembled=assembled,
        gang_stranded_gpu_hours=gang_stranded_seconds / SECONDS_PER_HOUR,
        gang_stranded_share=(
            gang_stranded_seconds / gpu_seconds_used if gpu_seconds_used > 0 else 0.0
        ),
        gang_assembly_p50=nearest_rank(assembly_delays, 50) if assembly_delays else None,
        gang_assembly_p95=nearest_rank(assembly_delays, 95) if assembly_delays else None,
        footprint_wait_spearman=spearman(footprints, admitted_waits),
        large_job_gpus=large,
        large_job_starvation_ratio=starvation,
        wait_by_footprint=wait_by_footprint,
        topology_declared=topo.declared,
        placement_multi_pod_jobs=placement.multi_pod_jobs,
        placement_tier_share=placement.tier_share,
        placement_penalty_mean=penalty_mean,
        penalty_factors=asdict(factors),
        fairness_ratio=fairness,
        service_ratio=ratios,
        harness=obs.run.harness(),
        harness_seed=obs.run.harness_seed,
        gpu_hours_demanded=demanded_seconds / SECONDS_PER_HOUR,
        startup_overhead_gpu_hours=startup_seconds / SECONDS_PER_HOUR,
        preemptions=len(obs.evictions),
        preempted_gpu_hours_lost=lost_seconds / SECONDS_PER_HOUR,
        grace_locked_gpu_hours=locked_seconds / SECONDS_PER_HOUR,
        topology_extension_gpu_hours=obs.extension_gpu_seconds / SECONDS_PER_HOUR,
        notes=list(obs.notes),
    )
