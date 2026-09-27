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

``D-preempt`` is the fourth, and the only one that may evict: strict priority
order, first fit, and -- for a pod that fits nowhere -- the victim rule in
:mod:`k8slab.preemption`. It exists as the floor for K1/K2's preemption. It has
no whole-pass function: the binder always drives it one pod at a time
(:meth:`k8slab.binder.Binder._cycles`), so :data:`SPECS` -- not
:data:`POLICIES` -- is the registry of configurations.

Each policy is split into a pod order and a node order (:class:`PolicySpec`)
so that the kube queue model can pop pods one scheduling cycle at a time in
the policy's own order (:mod:`k8slab.binder`). The whole-pass functions call
the two halves in Phase 1's order, so their results -- including D-random's
RNG draws -- are unchanged.
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


#: Every degenerate policy is "order the pods, order the nodes, first-fit".
#: The two halves are exposed separately so that the kube queue model
#: (:mod:`k8slab.binder`) can pop pods one scheduling cycle at a time in the
#: policy's order. The whole-pass functions below call them in exactly the
#: order Phase 1 did (pods, then nodes), so D-random draws the same numbers
#: from its RNG as before.
PodOrder = Callable[[list[PendingPod], random.Random], list[PendingPod]]
NodeOrder = Callable[[dict[str, int], random.Random], list[str]]


@dataclass(frozen=True)
class PolicySpec:
    order: PodOrder
    node_order: NodeOrder
    #: A pending pod that fits nowhere may evict lower-priority pods
    #: (:mod:`k8slab.preemption`). Only D-preempt.
    preemptive: bool = False


def _by_name(free: dict[str, int], rng: random.Random) -> list[str]:
    return sorted(free)


def _fifo_order(pods: list[PendingPod], rng: random.Random) -> list[PendingPod]:
    return sorted(pods, key=lambda p: (p.submit_time, p.job_id, p.pod_index))


def _random_order(pods: list[PendingPod], rng: random.Random) -> list[PendingPod]:
    ordered = list(pods)
    rng.shuffle(ordered)
    return ordered


def _random_nodes(free: dict[str, int], rng: random.Random) -> list[str]:
    nodes = sorted(free)
    rng.shuffle(nodes)
    return nodes


def _largest_order(pods: list[PendingPod], rng: random.Random) -> list[PendingPod]:
    return sorted(pods, key=lambda p: (-p.gpus, p.submit_time, p.job_id, p.pod_index))


def _priority_order(pods: list[PendingPod], rng: random.Random) -> list[PendingPod]:
    """Highest priority first, then oldest -- kube-scheduler's PrioritySort
    order (priority, then queue timestamp), with the trace's submit time
    standing in for the timestamp."""
    return sorted(pods, key=lambda p: (-p.priority, p.submit_time, p.job_id, p.pod_index))


def fifo(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Oldest submission first, first node with room. No packing, no gang logic."""
    ordered = _fifo_order(pods, rng)
    return _first_fit(ordered, dict(free), _by_name(free, rng))


def random_policy(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Random pod order, random node order. The true floor."""
    ordered = _random_order(pods, rng)
    nodes = _random_nodes(free, rng)
    return _first_fit(ordered, dict(free), nodes)


def largest_first(
    pods: list[PendingPod], free: dict[str, int], rng: random.Random
) -> list[tuple[PendingPod, str]]:
    """Biggest request first. Often beats FIFO on utilization and starves small jobs.

    Included because it is the cheap heuristic people reach for, and because a
    scheduler that loses to it on fragmentation is worth knowing about.
    """
    ordered = _largest_order(pods, rng)
    return _first_fit(ordered, dict(free), _by_name(free, rng))


#: Registry of every degenerate configuration. Keys are the config ids used in
#: results tables and ``--config``. D-preempt is DEGENERATE: no gang awareness,
#: no topology, the crudest victim rule that is still a rule -- the floor that
#: Kueue's and Volcano's preemption (K1, K2) will be compared against.
SPECS: dict[str, PolicySpec] = {
    "D-fifo": PolicySpec(_fifo_order, _by_name),
    "D-random": PolicySpec(_random_order, _random_nodes),
    "D-largest": PolicySpec(_largest_order, _by_name),
    "D-preempt": PolicySpec(_priority_order, _by_name, preemptive=True),
}

#: Phase 1's whole-pass functions, one per NON-preemptive spec. The binder
#: calls them verbatim under ``--queue-model none`` so Phase 1 reproduces bit
#: for bit; ``tests/test_binder.py`` checks each against its spec's
#: order/node-order halves so the two cannot drift.
POLICIES: dict[str, Policy] = {
    "D-fifo": fifo,
    "D-random": random_policy,
    "D-largest": largest_first,
}

DEGENERATE_IDS = tuple(SPECS)
#: Configurations whose binder may evict running pods.
PREEMPTIVE_IDS = tuple(k for k, s in SPECS.items() if s.preemptive)
