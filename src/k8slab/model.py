"""Core types: the fleet, the workload trace, and the observations a run produces.

Three layers, deliberately separated:

* :class:`NodeClass` / :class:`Fleet` describe the *simulated* hardware.
* :class:`Job` is a trace record — what was asked for, never what happened.
* :class:`Observation` is what a run measured. Metrics read only this, so the
  same metrics code scores a real cluster run and a degenerate baseline.

The `Job` field names deliberately match `slurm-scheduler-lab`'s trace records
where the concepts coincide (``job_id``, ``account``, ``submit_time``,
``duration``), so a translated ``sacct`` trace can be loaded in Phase 2 without
a second vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NodeClass:
    """A group of identical simulated nodes.

    Heterogeneity is the point: fragmentation cannot be measured on a fleet
    where every node is the same shape, because there is nothing for a job to
    fail to fit into.
    """

    name: str
    count: int
    gpus: int
    cpus: int = 32
    memory_gi: int = 256

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"node class {self.name!r}: count must be >= 1")
        if self.gpus < 0:
            raise ValueError(f"node class {self.name!r}: gpus must be >= 0")


@dataclass(frozen=True)
class Fleet:
    name: str
    classes: tuple[NodeClass, ...]

    @property
    def total_nodes(self) -> int:
        return sum(c.count for c in self.classes)

    @property
    def total_gpus(self) -> int:
        return sum(c.count * c.gpus for c in self.classes)

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
    """What happened to one pod of one job."""

    job_id: int
    pod_index: int
    #: Set when the pod was bound to a node. ``None`` means never scheduled.
    scheduled_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    node: str | None = None
    #: The API object disappeared without this runner deleting it. On a cluster
    #: that means kube-scheduler preemption destroyed it. A bare pod has no
    #: controller to recreate it, so the work is simply lost.
    preempted: bool = False

    @property
    def scheduled(self) -> bool:
        return self.scheduled_time is not None


@dataclass
class Observation:
    """Everything one configuration's run produced.

    Metrics are computed from this and nothing else, so a measured run and a
    simulated degenerate baseline are scored by identical code.
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

    def pods_of(self, job_id: int) -> list[PodEvent]:
        return [p for p in self.pods if p.job_id == job_id]
