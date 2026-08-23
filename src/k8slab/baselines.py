"""Degenerate scheduling policies, as pure functions.

These exist to give every results table a floor. A real scheduler that barely
beats FIFO has not been shown to be good; it has been shown that the trace is
too easy, and the fix is a harder trace rather than a retuned metric. The
convention is inherited from ``slurm-rca-bench``, where a degenerate agent that
reads no telemetry sets the score anyone else has to clear.

The same functions drive the in-cluster binder (:mod:`k8slab.runner`) and the
in-process reference model (:mod:`k8slab.sim`), so a degenerate baseline is
measured through exactly the path a real scheduler is measured through. Nothing
here is a simulation of Kubernetes — it is a scheduling *policy* that a real
binder applies to real API objects.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingPod:
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
Policy = Callable[[list[PendingPod], dict[str, int], random.Random], list[tuple[PendingPod, str]]]


def _first_fit(
    pods: list[PendingPod],
    free: dict[str, int],
    node_order: list[str],
) -> list[tuple[PendingPod, str]]:
    """Bind each pod to the first node in ``node_order`` with room.

    ``free`` is mutated as bindings are made, so one pass never double-books a
    node. This is the behaviour the degenerate policies share; only the pod
    ordering and the node ordering differ between them.
    """
    out: list[tuple[PendingPod, str]] = []
    for pod in pods:
        for node in node_order:
            if free.get(node, 0) >= pod.gpus and pod.gpus > 0:
                free[node] -= pod.gpus
                out.append((pod, node))
                break
    return out


def fifo(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Oldest submission first, first node with room. No packing, no gang logic."""
    ordered = sorted(pods, key=lambda p: (p.submit_time, p.job_id, p.pod_index))
    return _first_fit(ordered, dict(free), sorted(free))


def random_policy(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Random pod order, random node order. The true floor."""
    ordered = list(pods)
    rng.shuffle(ordered)
    nodes = sorted(free)
    rng.shuffle(nodes)
    return _first_fit(ordered, dict(free), nodes)


def largest_first(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Biggest request first. Often beats FIFO on utilization and starves small jobs.

    Included because it is the cheap heuristic people reach for, and because a
    scheduler that loses to it on fragmentation is worth knowing about.
    """
    ordered = sorted(
        pods, key=lambda p: (-p.gpus, p.submit_time, p.job_id, p.pod_index)
    )
    return _first_fit(ordered, dict(free), sorted(free))


#: Registry. Keys are the config ids used in results tables and `--config`.
POLICIES: dict[str, Policy] = {
    "D-fifo": fifo,
    "D-random": random_policy,
    "D-largest": largest_first,
}

DEGENERATE_IDS = tuple(POLICIES)
