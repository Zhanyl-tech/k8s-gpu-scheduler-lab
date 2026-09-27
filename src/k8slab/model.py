"""Core types: the fleet, the workload trace, and the observations a run produces.

Three layers, deliberately separated:

* :class:`NodeClass` / :class:`Fleet` describe the *simulated* hardware.
* :class:`Job` is a trace record — what was asked for, never what happened.
* :class:`Observation` is what a run measured. Metrics read only this, so the
  same metrics code scores a real cluster run and a degenerate baseline.

The `Job` field names deliberately match `slurm-scheduler-lab`'s trace records
where the concepts coincide (``job_id``, ``account``, ``submit_time``,
``duration``), so a translated ``sacct`` trace -- planned, not part of the Phase
2 work so far -- could be loaded without a second vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from .topology import Topology


@dataclass(frozen=True)
class NodeClass:
    """A group of identical simulated nodes.

    The default fleet mixes classes so that request shapes and node shapes
    mismatch: a 4-GPU pod fits an 8-GPU node but not a 2-GPU one, so where a
    scheduler puts small jobs decides what large ones can still use. Identical
    nodes fragment too, under a mixed trace (small jobs carve 8-GPU nodes);
    exactly zero is forced only for a workload of whole-node jobs on a
    single-shape fleet -- the control in docs/metrics.md, "The control".
    (This docstring used to say fragmentation cannot be measured on a fleet of
    identical nodes; the reference model measures it there, and it differs by
    policy.)
    """

    name: str
    count: int
    gpus: int
    cpus: int = 32
    memory_gi: int = 256
    #: The GPUs inside one node of this class share an NVLink (or equivalent
    #: intra-node) fabric. DECLARED, never detected: a kwok node has no GPUs,
    #: so this is a scenario property of the simulated fleet. It only feeds the
    #: placement measurement in :mod:`k8slab.topology`.
    nvlink: bool = False
    #: Nodes of this class per rack. ``None`` puts the whole class in one rack.
    #: Racks never mix node classes; see :func:`k8slab.topology.derive`.
    nodes_per_rack: int | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"node class {self.name!r}: count must be >= 1")
        if self.gpus < 0:
            raise ValueError(f"node class {self.name!r}: gpus must be >= 0")
        if self.nodes_per_rack is not None and self.nodes_per_rack < 1:
            raise ValueError(f"node class {self.name!r}: nodesPerRack must be >= 1")


@dataclass(frozen=True)
class Fleet:
    name: str
    classes: tuple[NodeClass, ...]
    #: Consecutive racks grouped under one switch. ``None`` puts every rack
    #: under a single switch -- a flat fabric, the assumption that invents the
    #: least topology when a fleet file declares none.
    racks_per_switch: int | None = None
    #: True when the fleet file declared any topology key. False means every
    #: rack/switch/NVLink value is a default, and placement tiers computed on
    #: this fleet describe the defaults rather than a declared scenario.
    topology_declared: bool = False

    def __post_init__(self) -> None:
        if self.racks_per_switch is not None and self.racks_per_switch < 1:
            raise ValueError(f"fleet {self.name!r}: racksPerSwitch must be >= 1")

    @property
    def total_nodes(self) -> int:
        return sum(c.count for c in self.classes)

    @property
    def total_gpus(self) -> int:
        return sum(c.count * c.gpus for c in self.classes)

    @property
    def max_node_gpus(self) -> int:
        """The largest single-node GPU capacity.

        A job needing at least this many GPUs in total needs at least one whole
        largest node (or several nodes) to drain before it can start, which is
        what makes it "large" for the starvation ratio. It does NOT mean the
        job cannot fit on one node: a job of exactly this size fits one empty
        largest node."""
        return max((c.gpus for c in self.classes), default=0)

    def node_names(self) -> list[str]:
        """Stable, deterministic node names: ``<class>-<ordinal>``."""
        names: list[str] = []
        for cls in self.classes:
            names.extend(f"{cls.name}-{i}" for i in range(cls.count))
        return names

    def gpus_of(self, node_name: str) -> int:
        for cls in self.classes:
            if node_name.rsplit("-", 1)[0] == cls.name:
                return cls.gpus
        raise KeyError(f"unknown node {node_name!r}")


@dataclass(frozen=True)
class Job:
    """One trace record.

    ``duration`` is what the job will actually run for. A scheduler is never
    allowed to read it — it is the ground truth the harness uses to decide when
    a pod finishes, exactly as ``slurm-scheduler-lab`` treats it.
    """

    job_id: int
    account: str
    submit_time: float
    duration: float
    #: GPUs required *per pod*.
    gpus: int
    #: Pods that must run concurrently. 1 for an ordinary job; >1 is a gang.
    gang_size: int = 1
    priority: int = 0

    @property
    def total_gpus(self) -> int:
        return self.gpus * self.gang_size

    @property
    def is_gang(self) -> bool:
        return self.gang_size > 1


@dataclass
class PodEvent:
    """What happened to one pod of one job.

    Three timestamps, three different events, all in simulated seconds:

    * ``scheduled_time`` -- the pod was **bound** (``spec.nodeName`` set). From
      this instant it holds its GPUs in the scheduler's accounting, whether or
      not anything runs. Wait and fragmentation A read this.
    * ``start_time`` -- the pod was observed **Running**. A gang counts as
      assembled only when every member has started (see
      ``gang_stranded_gpu_hours`` in docs/metrics.md). Delivered GPU-hours are
      ``start_time`` to ``end_time``.
    * ``end_time`` -- the pod released its GPUs.

    With the default zero startup delay the two are equal. With a startup
    delay (``--startup-delay``) the reference model draws the bind-to-Running
    gap from a seeded RNG, and the runner records ``start_time`` on the first
    poll that shows the pod ``Running`` -- so on a cluster both are quantised
    to the poll interval. Trace runtime is counted from ``start_time``, and
    bind-to-Running is held-but-idle GPU time (``startup_overhead_gpu_hours``;
    for gangs it is also stranded time).
    """

    job_id: int
    pod_index: int
    #: Set when the pod was bound to a node. ``None`` means never scheduled.
    scheduled_time: float | None = None
    #: Set when the pod was observed Running. ``None`` means it never started.
    start_time: float | None = None
    end_time: float | None = None
    node: str | None = None
    #: The API object was deleted by the control plane, not by this runner. On
    #: a cluster that means kube-scheduler preemption destroyed it. A bare pod
    #: has no controller to recreate it, so the work is simply lost. (The
    #: reference model's own evictions requeue the pod instead; those are
    #: :class:`Eviction` records and leave this False on the final attempt.)
    preempted: bool = False

    @property
    def scheduled(self) -> bool:
        return self.scheduled_time is not None


#: Queue models an execution layer can apply to the in-process binder.
QUEUE_MODELS: tuple[str, ...] = ("none", "kube")
#: What the ASSUMED topology penalty factors are used for.
TOPOLOGY_MODES: tuple[str, ...] = ("off", "report", "extend")


@dataclass(frozen=True)
class RunInfo:
    """The harness settings one run was produced under.

    Recorded on every :class:`Observation` and copied into every result, because
    two runs are repeats of one experiment only if they were produced the same
    way. ``k8slab.stats.aggregate`` refuses to pool runs whose settings differ.
    The defaults reproduce Phase 1 exactly: no queue model, no startup delay,
    penalty factors reported but never applied, no preemption parameters in use.
    """

    #: ``none`` -- Phase 1: the in-process binder sees every pending pod on
    #: every pass and never backs off. ``kube`` -- the binder goes through
    #: :class:`k8slab.queueing.SchedulingQueue`, kube-scheduler's queue
    #: mechanics, with backoff in simulated seconds from ``speedup``.
    queue_model: str = "none"
    #: Simulated seconds each scheduling attempt consumes (``kube`` only).
    #: Default 0: no latency is invented to make any comparison look fairer.
    cycle_latency: float = 0.0
    #: The compression factor the run's real-time constants were scaled by.
    #: ``None`` means no real-time constant was used (Phase 1 reference model).
    speedup: float | None = None
    #: Bind-to-Running delay, uniform on ``[min, max)`` milliseconds of REAL
    #: time (kwok's Stage delay semantics); converted to simulated seconds by
    #: multiplying by ``speedup``. ``(0, 0)`` is Phase 1: Running at bind.
    startup_delay_ms: tuple[float, float] = (0.0, 0.0)
    #: ``off`` -- factors neither applied nor reported; ``report`` -- only
    #: ``placement_penalty_mean`` uses them; ``extend`` -- a SCENARIO: every
    #: job's remaining runtime is stretched by its placement factor.
    topology_penalty: str = "report"
    #: Simulated seconds an evicted pod keeps its GPUs locked
    #: (terminationGracePeriodSeconds; the Kubernetes default is 30).
    grace_seconds: float = 30.0
    #: Share of an evicted attempt's progress that survives the eviction.
    #: 0 (the default) restarts preempted work from zero.
    checkpoint_fraction: float = 0.0
    #: DIAGNOSTIC: a pod is submitted only once the previous one was bound.
    admission_gate: bool = False
    #: DIAGNOSTIC: kube-scheduler ran with ``parallelism: 1``.
    serialise: bool = False
    #: Seed of the binder's and the startup-delay sampler's RNGs. Varies across
    #: repeats by design, so it is identity, not a setting that must match.
    harness_seed: int | None = None

    def __post_init__(self) -> None:
        if self.queue_model not in QUEUE_MODELS:
            raise ValueError(f"queue_model must be one of {QUEUE_MODELS}, got {self.queue_model!r}")
        if self.topology_penalty not in TOPOLOGY_MODES:
            raise ValueError(
                f"topology_penalty must be one of {TOPOLOGY_MODES}, got {self.topology_penalty!r}"
            )
        # Every range check below is a comparison, and every comparison with
        # NaN is False: a NaN grace period passed `< 0` and gave a lock that
        # never releases. Non-finite values are refused before the ranges.
        low, high = self.startup_delay_ms
        numbers = {
            "cycle_latency": self.cycle_latency,
            "startup_delay_ms": low,
            "startup_delay_ms max": high,
            "grace_seconds": self.grace_seconds,
            "checkpoint_fraction": self.checkpoint_fraction,
        }
        if self.speedup is not None:
            numbers["speedup"] = self.speedup
        for name, value in numbers.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.cycle_latency < 0:
            raise ValueError("cycle_latency must be >= 0")
        if self.cycle_latency > 0 and self.queue_model != "kube":
            raise ValueError("cycle_latency needs queue_model 'kube': 'none' has no cycles")
        if self.speedup is not None and self.speedup <= 0:
            raise ValueError("speedup must be > 0")
        if low < 0 or high < low:
            raise ValueError(f"startup_delay_ms must satisfy 0 <= min <= max, got {low}:{high}")
        if high > 0 and self.speedup is None:
            raise ValueError("a startup delay in real milliseconds needs a speedup")
        if self.queue_model == "kube" and self.speedup is None:
            raise ValueError("queue_model 'kube' needs a speedup to scale its backoff")
        if self.grace_seconds < 0:
            raise ValueError("grace_seconds must be >= 0")
        if not 0.0 <= self.checkpoint_fraction <= 1.0:
            raise ValueError("checkpoint_fraction must be in [0, 1]")

    @property
    def gated(self) -> bool:
        """A diagnostic that changes WHAT is measured (docs/limitations.md)."""
        return self.admission_gate or self.serialise

    @property
    def scenario(self) -> bool:
        """Runtimes were stretched by ASSUMED factors: not a measurement."""
        return self.topology_penalty == "extend"

    def harness(self) -> dict[str, Any]:
        """Every setting that must be equal for runs to be pooled (not the seed)."""
        return {
            "queue_model": self.queue_model,
            "cycle_latency": self.cycle_latency,
            "speedup": self.speedup,
            "startup_delay_ms": [self.startup_delay_ms[0], self.startup_delay_ms[1]],
            "topology_penalty": self.topology_penalty,
            "grace_seconds": self.grace_seconds,
            "checkpoint_fraction": self.checkpoint_fraction,
            "admission_gate": self.admission_gate,
            "serialise": self.serialise,
        }


@dataclass(frozen=True)
class Eviction:
    """One pod attempt removed from its node before it finished.

    In the reference model this is D-preempt's eviction API; on a cluster it is
    a pod the control plane deleted. The evicted attempt's GPUs stay locked from
    ``evict_time`` to ``release_time`` (the grace period), and the pod is
    requeued -- in the model -- with its progress lost except for
    ``checkpoint_fraction`` of it.
    """

    job_id: int
    pod_index: int
    node: str
    gpus: int
    bind_time: float
    #: ``None`` when evicted before it was ever Running.
    start_time: float | None
    evict_time: float
    #: When the grace-period lock ended and the GPUs became free.
    release_time: float
    reason: str
    #: Wall (simulated) seconds of Running thrown away: ``(1 - c) * run``.
    lost_seconds: float = 0.0
    #: Wall seconds of Running kept by a checkpoint: ``c * run``.
    retained_seconds: float = 0.0
    #: True when the pod was requeued (the model, D-preempt): its final
    #: :class:`PodEvent` is a later attempt. False when nothing recreated it (a
    #: bare pod the control plane deleted): the final PodEvent IS this attempt,
    #: flagged ``preempted``, and its Running time is lost, not delivered.
    requeued: bool = True


@dataclass
class Observation:
    """Everything one configuration's run produced.

    Metrics are computed from this and nothing else, so a measured run and a
    simulated degenerate baseline are scored by identical code.

    With evictions, each :class:`PodEvent` describes the pod's FINAL attempt
    (the one that completed or was still holding GPUs at the horizon); every
    earlier attempt is an :class:`Eviction`.
    """

    config: str
    fleet: Fleet
    jobs: list[Job]
    pods: list[PodEvent]
    #: Sampled ``(timestamp, {node_name: free_gpus})`` over the run. Used for
    #: fragmentation, which is a time-integral and cannot be recovered from
    #: start/end timestamps alone.
    gpu_free_samples: list[tuple[float, dict[str, int]]] = field(default_factory=list)
    horizon: float = 0.0
    #: False whenever the numbers came from the in-process reference model
    #: rather than a real control plane. Printed next to every result.
    measured_on_cluster: bool = False
    #: Per-node rack/switch/NVLink domains. ``None`` (the default) derives them
    #: from ``fleet`` with :func:`k8slab.topology.derive`. An execution layer
    #: that reads the ``topology.k8slab.io/*`` labels back from the Node
    #: objects it actually ran against can set this, so placement is scored
    #: against the topology the scheduler saw rather than the one the fleet
    #: file intended.
    topology: Topology | None = None
    #: How the run was produced. The default is Phase 1's harness.
    run: RunInfo = field(default_factory=RunInfo)
    #: Every attempt evicted before it finished, in eviction order.
    evictions: list[Eviction] = field(default_factory=list)
    #: GPU-seconds of delivered time that exist only because runtimes were
    #: stretched by ASSUMED topology factors (``topology_penalty="extend"``).
    #: 0 in every other mode. Delivered GPU-time minus this is the trace's work.
    extension_gpu_seconds: float = 0.0
    #: ``extension_gpu_seconds`` split by the account of the job it stretched.
    #: Fairness subtracts each account's share from its delivered time, so a
    #: service ratio counts the trace's work, not the ASSUMED stretch. Empty
    #: when nothing was stretched; otherwise it must sum to the total.
    extension_by_account: dict[str, float] = field(default_factory=dict)
    #: Free-form notes the execution layer wants carried into the results
    #: (for example a grace lock the control plane did not honour).
    notes: list[str] = field(default_factory=list)

    def pods_of(self, job_id: int) -> list[PodEvent]:
        return [p for p in self.pods if p.job_id == job_id]
