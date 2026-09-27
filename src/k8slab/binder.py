"""The in-process scheduler for degenerate configurations.

One object decides, pass by pass, which pending pods a degenerate policy binds
where -- and, for D-preempt, which running pods it evicts. The reference model
(:mod:`k8slab.sim`) and the cluster runner (:mod:`k8slab.runner`) both call it,
so a degenerate baseline makes the same decisions from the same state whether
the bindings land in a dict or on a real API server.

Two queue models (``RunInfo.queue_model``):

* ``none`` -- Phase 1. Every pass the policy sees every pending pod and
  first-fits them all; a pod that does not fit is simply tried again next pass.
  For the three Phase 1 policies this path is the literal Phase 1 call,
  ``policy(pending, free, rng)``, so results are bit-identical
  (``tests/test_sim_equivalence.py``).
* ``kube`` -- kube-scheduler's queue mechanics (:mod:`k8slab.queueing`). Only
  pods in activeQ are tried; they are popped one per scheduling cycle in the
  policy's order; a pod that does not fit is parked and backs off. With
  ``cycle_latency > 0`` each attempt consumes that many simulated seconds of
  the pass window, so a pass can try only ``window / cycle_latency`` pods; the
  default is 0 (no cycle-time cost), because inventing a latency to make one
  side look fairer is exactly what this module replaces.

Preemption (D-preempt only): a pod that fits nowhere and has no nomination
asks :func:`k8slab.preemption.plan_preemption` for victims. The caller evicts
them (GPUs locked for the grace period, pods requeued) and the preemptor is
*nominated* to the node. Two things follow, and only the first is upstream
behaviour:

* It is tried on the nominated node first. That matches kube-scheduler: "The
  scheduler always tries the 'nominated Node' before iterating over any other
  nodes" (https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/,
  "User exposed information", read 2026-09-26).
* Its GPUs on that node are reserved against EVERY other pod, whatever its
  priority. That is a deliberate D-preempt simplification and differs from
  kube-scheduler, where a nominated pod is counted against a pod being
  scheduled only if its priority is greater than or equal to that pod's
  (``addNominatedPods``: ``if corev1.PodPriority(pi.Pod) >=
  corev1.PodPriority(pod)``, pkg/scheduler/framework/runtime/framework.go
  lines 1034-1057 at v1.32.2), so "the scheduler may give Node N to the new
  higher priority Pod" and clear the lower one's ``nominatedNodeName`` (same
  docs section). Recorded in docs/limitations.md ("Preemption").
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .baselines import POLICIES, SPECS, PendingPod, Policy
from .model import RunInfo
from .preemption import PreemptionPlan, RunningIndex, RunningPod, plan_preemption
from .queueing import ACTIVE, QueueParams, SchedulingQueue

Key = tuple[int, int]


@dataclass
class Decision:
    """What one pass decided. The caller applies it."""

    binds: list[tuple[PendingPod, str]] = field(default_factory=list)
    preemptions: list[PreemptionPlan] = field(default_factory=list)


class Binder:
    """Decides binds (and evictions) for one degenerate configuration."""

    def __init__(
        self,
        config: str,
        run: RunInfo,
        rng: random.Random,
        params: QueueParams | None = None,
        window: float = 5.0,
    ) -> None:
        if config not in SPECS:
            raise KeyError(f"unknown policy {config!r}; have {sorted(SPECS)}")
        self.config = config
        self.spec = SPECS[config]
        #: Phase 1's whole-pass function; ``None`` for a preemptive spec,
        #: which is always driven one pod at a time.
        self.policy: Policy | None = POLICIES.get(config)
        if self.policy is None and not self.spec.preemptive:
            raise KeyError(f"non-preemptive policy {config!r} has no whole-pass function")
        self.run = run
        self.rng = rng
        self.queue: SchedulingQueue | None = None
        if run.queue_model == "kube":
            if params is None:
                raise ValueError("queue_model 'kube' needs QueueParams")
            self.queue = SchedulingQueue(params)
        #: preemptor -> (node, GPUs reserved there)
        self.nominated: dict[Key, tuple[str, int]] = {}
        self._window = window
        self._last_pass: float | None = None

    # -- events -------------------------------------------------------------

    def capacity_freed(self, now: float) -> None:
        """A scheduled pod's GPUs were released (deleted, or grace lock over)."""
        if self.queue is not None:
            self.queue.capacity_freed(now)

    # -- the pass -------------------------------------------------------------

    def schedule(
        self,
        now: float,
        pending: Sequence[PendingPod],
        free: Mapping[str, int],
        running: Sequence[RunningPod] = (),
    ) -> Decision:
        """Decide this pass. ``free`` excludes GPUs under a grace lock."""
        keys = {p.key for p in pending}
        for key in [k for k in self.nominated if k not in keys]:
            del self.nominated[key]
        window = self._window if self._last_pass is None else now - self._last_pass
        self._last_pass = now

        if self.queue is None:
            if not pending:
                return Decision()
            if self.policy is not None and not self.nominated:
                # The literal Phase 1 call: same arguments, same RNG draws.
                return Decision(binds=self.policy(list(pending), dict(free), self.rng))
            return self._cycles(self.spec.order(list(pending), self.rng), free, running, now)

        q = self.queue
        q.reconcile([p.key for p in pending], now)
        q.flush(now)
        active = set(q.keys_in(ACTIVE))
        candidates = [p for p in pending if p.key in active]
        if not candidates:
            return Decision()
        ordered = self.spec.order(candidates, self.rng)
        if self.run.cycle_latency > 0:
            budget = max(1, math.floor(window / self.run.cycle_latency + 1e-9))
            ordered = ordered[:budget]
        return self._cycles(ordered, free, running, now)

    def _reservations(self) -> dict[str, int]:
        """GPUs promised to nominated preemptors, per node."""
        reserved: dict[str, int] = {}
        for node, gpus in self.nominated.values():
            reserved[node] = reserved.get(node, 0) + gpus
        return reserved

    def _cycles(
        self,
        ordered: Sequence[PendingPod],
        free: Mapping[str, int],
        running: Sequence[RunningPod],
        now: float,
    ) -> Decision:
        """Try pods one at a time, in order, against a working copy of free."""
        work = dict(free)
        nodes = self.spec.node_order(dict(free), self.rng)
        decision = Decision()
        evicting: set[Key] = set()
        index = RunningIndex(running) if self.spec.preemptive else None
        reserved = self._reservations()
        q = self.queue
        for pod in ordered:
            if q is not None:
                q.pop(pod.key)
            node = self._place(pod, work, nodes, reserved)
            if node is not None:
                work[node] -= pod.gpus
                nominated = self.nominated.pop(pod.key, None)
                if nominated is not None:
                    reserved[nominated[0]] -= nominated[1]
                decision.binds.append((pod, node))
                if q is not None:
                    q.done(pod.key)
                continue
            if index is not None and pod.key not in self.nominated:
                visible = {n: max(0, work[n] - reserved.get(n, 0)) for n in work}
                plan = plan_preemption(pod, visible, index, frozenset(evicting))
                if plan is not None:
                    decision.preemptions.append(plan)
                    evicting.update(plan.victims)
                    self.nominated[pod.key] = (plan.node, pod.gpus)
                    reserved[plan.node] = reserved.get(plan.node, 0) + pod.gpus
            if q is not None:
                q.failed(pod.key, now)
        return decision

    def _place(
        self,
        pod: PendingPod,
        work: Mapping[str, int],
        nodes: list[str],
        reserved: Mapping[str, int],
    ) -> str | None:
        """First node with room, trying the pod's nominated node first and
        never taking GPUs reserved for another nominated pod."""
        if pod.gpus <= 0:
            return None  # zero-GPU pods are never bound (Phase 1 rule)
        order = nodes
        nominated = self.nominated.get(pod.key)
        if nominated is not None:
            order = [nominated[0], *(n for n in nodes if n != nominated[0])]
        for node in order:
            own = nominated[1] if nominated is not None and nominated[0] == node else 0
            if work.get(node, 0) - (reserved.get(node, 0) - own) >= pod.gpus:
                return node
        return None
