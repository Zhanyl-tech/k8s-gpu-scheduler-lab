"""D-preempt's victim rule, as a pure function.

Strict priority preemption, deliberately crude. When a pending pod fits on no
node, look for a node where evicting running pods of **strictly lower
priority** would free enough GPUs, and pick the one needing the **fewest
evictions**. Gang members are evicted as a whole gang: a gang with one member
gone cannot run, so evicting one member evicts every bound member of that job,
on whatever node, and every one of them counts as an eviction.

How this differs from kube-scheduler, on purpose. kube-scheduler's
DefaultPreemption picks among candidate nodes by, in order: fewest
PodDisruptionBudget violations, then the lowest highest-victim priority, then
the smallest sum of victim priorities, then the fewest victims, then the latest
start time of the highest-priority victims (``pickOneNodeForPreemption``,
pkg/scheduler/framework/preemption/preemption.go lines 568-632 at v1.32.2,
https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/framework/preemption/preemption.go).
Its PDB handling is best effort: "respecting PDB is best effort ... if no such
victims are found, preemption will still happen, and lower priority Pods will
be removed despite their PDBs being violated"
(https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/,
"PodDisruptionBudget is supported, but not guaranteed", read 2026-09-26). It
knows nothing of gangs. D-preempt has no PDBs, minimises evictions first and
expands gangs. It is a degenerate baseline -- the floor K1/K2 preemption must
beat -- not a model of any real scheduler.

The planner sees *visible* free GPUs: what is free minus what other pending
pods have been promised by earlier preemptions. GPUs still locked by an
evicted pod's grace period are not free and are not counted as about to be
free, so the planner can over-preempt while a grace lock is running. That is
recorded as a limitation, not hidden.

``running`` must hold attempts that are still RUNNING. An attempt whose trace
work has ended is never a victim: evicting it would charge Running time beyond
its duration as lost and run the job again. The reference model releases such
attempts before its pass; the cluster runner, which deletes finished pods only
after its bind pass, leaves them out of the view it passes in. Their GPUs are
not counted as about to be free either -- the same over-preemption caveat as a
grace lock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .baselines import PendingPod

Key = tuple[int, int]


@dataclass(frozen=True)
class RunningPod:
    """A bound pod attempt that could be evicted."""

    key: Key
    node: str
    gpus: int
    priority: int

    @property
    def job_id(self) -> int:
        return self.key[0]


@dataclass(frozen=True)
class PreemptionPlan:
    preemptor: Key
    node: str
    #: Every attempt to evict, gang-expanded, in a stable order.
    victims: tuple[Key, ...]

    @property
    def evictions(self) -> int:
        return len(self.victims)


class RunningIndex:
    """``running`` grouped by job and by node, built once per pass."""

    def __init__(self, running: Sequence[RunningPod]) -> None:
        self.by_job: dict[int, list[RunningPod]] = {}
        self.by_node: dict[str, list[RunningPod]] = {}
        for r in running:
            self.by_job.setdefault(r.job_id, []).append(r)
            self.by_node.setdefault(r.node, []).append(r)
        self.min_priority = min((r.priority for r in running), default=None)


def plan_preemption(
    pod: PendingPod,
    free: Mapping[str, int],
    running: Sequence[RunningPod] | RunningIndex,
    exclude: frozenset[Key] = frozenset(),
) -> PreemptionPlan | None:
    """The cheapest eviction set that makes ``pod`` fit on one node, or None.

    ``free`` is the visible free GPUs per node; ``exclude`` are attempts that
    are already being evicted this pass. On each node, candidate victim *jobs*
    are those with a bound pod there of lower priority than ``pod``; they are
    taken lowest priority first, then fewest evictions (whole gang), then most
    GPUs freed on this node, then job id -- until the node would have room.
    Among feasible nodes the plan with the fewest evictions wins, then the
    lowest highest-victim-priority, then node name.
    """
    index = running if isinstance(running, RunningIndex) else RunningIndex(running)
    if pod.gpus <= 0 or index.min_priority is None or index.min_priority >= pod.priority:
        return None  # nothing it is allowed to evict

    def members(job_id: int) -> list[RunningPod]:
        return [r for r in index.by_job[job_id] if r.key not in exclude]

    best: tuple[tuple[int, int, str], PreemptionPlan] | None = None
    for node in sorted(free):
        need = pod.gpus - free[node]
        if need <= 0:
            continue  # it fits here already; not a preemption case
        on_node: dict[int, int] = {}
        for r in index.by_node.get(node, ()):
            if r.key in exclude or r.priority >= pod.priority:
                continue
            on_node[r.job_id] = on_node.get(r.job_id, 0) + r.gpus
        if sum(on_node.values()) < need:
            continue
        gangs = {j: members(j) for j in on_node}
        candidates = sorted(
            on_node,
            key=lambda j: (gangs[j][0].priority, len(gangs[j]), -on_node[j], j),
        )
        freed = 0
        chosen: list[int] = []
        for job_id in candidates:
            if freed >= need:
                break
            chosen.append(job_id)
            freed += on_node[job_id]
        victims = tuple(sorted(r.key for j in chosen for r in gangs[j]))
        worst = max(gangs[j][0].priority for j in chosen)
        rank = (len(victims), worst, node)
        if best is None or rank < best[0]:
            best = (rank, PreemptionPlan(preemptor=pod.key, node=node, victims=victims))
    return None if best is None else best[1]
