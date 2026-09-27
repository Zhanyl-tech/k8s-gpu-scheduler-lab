"""kube-scheduler's scheduling-queue mechanics, as a pure state machine.

Why this exists
---------------

Phase 1 measured K0 (kube-scheduler) and the degenerate baselines through
unequal machinery. kube-scheduler pops one pod per scheduling cycle from an
active queue, parks pods that do not fit in an unschedulable pool, and retries
them only after a cluster event or a timeout, with exponential backoff. The
in-process binder that drives the degenerate policies looked at every pending
pod on every pass and never backed off. The README called that asymmetry out as
flattering the baselines on makespan.

This module gives the in-process binder the same mechanics, so that the
difference between K0 and a degenerate policy is the *policy* and not the
queue. It is used by :mod:`k8slab.binder`, which both :mod:`k8slab.sim` and the
cluster runner call. ``--queue-model none`` bypasses it and reproduces Phase 1.

What is modelled, verified against Kubernetes v1.32.2 (read 2026-09-26)
------------------------------------------------------------------------

Source: pkg/scheduler/backend/queue/scheduling_queue.go and active_queue.go,
https://github.com/kubernetes/kubernetes/tree/v1.32.2/pkg/scheduler/backend/queue

* Three places a pending pod can be: **activeQ** (ready to be tried),
  **backoffQ** (failed recently, waiting out its backoff) and the
  **unschedulable pool** (failed, waiting for something to change).
* A new pod enters activeQ with 0 attempts. ``Pop`` removes **one** pod per
  scheduling cycle and increments its attempt count (``pInfo.Attempts++`` in
  ``activeQueue.unlockedPop``).
* A pod that does not fit goes to the unschedulable pool with its timestamp
  refreshed to now (``AddUnschedulableIfNotPresent``; this model has no events
  arriving while a pod is in flight, so the in-flight requeue path never
  applies).
* Backoff is ``initial * 2^(attempts-1)`` capped at the ceiling, counted from
  the pod's timestamp (``calculateBackoffDuration``, ``getBackoffTime``); a pod
  with 0 attempts has none.
* A cluster event a pod's rejecting plugin cares about moves it out of the
  pool: to backoffQ if it is still backing off, otherwise straight to activeQ
  (``movePodsToActiveOrBackoffQueue`` -> ``requeuePodViaQueueingHint``). For
  NodeResourcesFit -- the only reason a GPU pod fails here -- the event is the
  deletion of a *scheduled* pod, and its queueing hint returns ``Queue`` for
  any such deletion (``isSchedulableAfterPodEvent`` in
  pkg/scheduler/framework/plugins/noderesources/fit.go). QueueingHints are
  on by default in 1.32 (``SchedulerQueueingHints``: beta, default true from
  1.32, pkg/features/versioned_kube_features.go). So "capacity freed" is the
  event, fired when a pod's GPUs are actually released -- after a grace-period
  lock, not at eviction.
* Every 1 s (real) backoffQ pods whose backoff has expired move to activeQ;
  every 30 s (real) pods that sat in the pool longer than
  ``podMaxInUnschedulablePodsDuration`` (default 5 min) are moved out as if an
  event had arrived (``flushBackoffQCompleted``,
  ``flushUnschedulablePodsLeftover``, both driven by ``wait.Until`` in
  ``PriorityQueue.Run``).

Every duration here is in **simulated** seconds.
:meth:`QueueParams.from_timescale` derives them from
:class:`k8slab.timescale.TimeScale`, i.e. from exactly what the generated
KubeSchedulerConfiguration makes the real kube-scheduler experience at that
speedup -- including the flush timers, which cannot be scaled. The degenerate
binder and K0 therefore face the same simulated backoff.

What is not modelled: activeQ's ordering (kube-scheduler sorts by priority
then timestamp; the in-process binder keeps each degenerate policy's own
order, because the order *is* the policy), scheduling gates, PreEnqueue
plugins, and events arriving while a pod is in flight.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .timescale import TimeScale

Key = tuple[int, int]

ACTIVE = "active"
BACKOFF = "backoff"
UNSCHEDULABLE = "unschedulable"
IN_FLIGHT = "in-flight"


@dataclass(frozen=True)
class QueueParams:
    """Queue timings, in simulated seconds."""

    initial_backoff: float
    max_backoff: float
    max_in_unschedulable: float
    backoff_flush_interval: float
    unschedulable_flush_interval: float

    def __post_init__(self) -> None:
        if self.initial_backoff <= 0 or self.max_backoff < self.initial_backoff:
            raise ValueError("need 0 < initial_backoff <= max_backoff (kube-scheduler validation)")
        if min(self.max_in_unschedulable, self.backoff_flush_interval,
               self.unschedulable_flush_interval) <= 0:
            raise ValueError("queue durations must be > 0")

    @classmethod
    def from_timescale(cls, scale: TimeScale) -> QueueParams:
        """What kube-scheduler, configured by :mod:`k8slab.timescale` for this
        speedup, experiences in trace time."""
        return cls(
            initial_backoff=scale.initial_backoff_sim,
            max_backoff=scale.max_backoff_sim,
            max_in_unschedulable=scale.max_in_unschedulable_sim,
            backoff_flush_interval=scale.backoff_flush_interval_sim,
            unschedulable_flush_interval=scale.unschedulable_flush_interval_sim,
        )



@dataclass
class QueuedPod:
    key: Key
    attempts: int
    #: Last time the pod was added or failed an attempt (``pInfo.Timestamp``).
    timestamp: float
    where: str


class SchedulingQueue:
    """activeQ, backoffQ and the unschedulable pool for one run.

    The caller drives it: :meth:`add` a pod when it becomes pending,
    :meth:`pop` it when a scheduling cycle tries it, then :meth:`done` (bound)
    or :meth:`failed` (did not fit). :meth:`capacity_freed` is the cluster
    event; :meth:`flush` runs the two periodic timers and must be called with
    the current time before each scheduling pass.
    """

    def __init__(self, params: QueueParams) -> None:
        self.params = params
        self._pods: dict[Key, QueuedPod] = {}
        self._next_backoff_flush = 0.0
        self._next_unschedulable_flush = 0.0

    # -- membership -------------------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in self._pods

    def __len__(self) -> int:
        return len(self._pods)

    def where(self, key: Key) -> str:
        return self._pods[key].where

    def attempts(self, key: Key) -> int:
        return self._pods[key].attempts

    def keys_in(self, where: str) -> list[Key]:
        """Keys in one queue, in the order they entered the queue structure."""
        return [k for k, p in self._pods.items() if p.where == where]

    def add(self, key: Key, now: float) -> None:
        """A new pending pod: straight to activeQ, no attempts, no backoff."""
        if key not in self._pods:
            self._pods[key] = QueuedPod(key=key, attempts=0, timestamp=now, where=ACTIVE)

    def remove(self, key: Key) -> None:
        """The pod left the queue for good (bound, deleted)."""
        self._pods.pop(key, None)

    # -- one scheduling cycle -----------------------------------------------------

    def pop(self, key: Key) -> None:
        """Start a scheduling cycle for ``key``, which must be in activeQ."""
        pod = self._pods[key]
        if pod.where != ACTIVE:
            raise ValueError(f"pod {key} is in {pod.where}, not activeQ")
        pod.attempts += 1
        pod.where = IN_FLIGHT

    def done(self, key: Key) -> None:
        """The cycle bound the pod."""
        self.remove(key)

    def failed(self, key: Key, now: float) -> None:
        """The cycle found no node: park it in the unschedulable pool."""
        pod = self._pods[key]
        if pod.where != IN_FLIGHT:
            raise ValueError(f"pod {key} was not in flight")
        pod.timestamp = now
        pod.where = UNSCHEDULABLE

    # -- backoff --------------------------------------------------------------

    def backoff_duration(self, attempts: int) -> float:
        """``initial * 2^(attempts-1)``, capped; 0 for a never-tried pod."""
        if attempts <= 0:
            return 0.0
        duration = self.params.initial_backoff
        for _ in range(1, attempts):
            if duration > self.params.max_backoff - duration:
                return self.params.max_backoff
            duration += duration
        return duration

    def backoff_expiry(self, key: Key) -> float:
        pod = self._pods[key]
        return pod.timestamp + self.backoff_duration(pod.attempts)

    def is_backing_off(self, key: Key, now: float) -> bool:
        """``boTime.After(now)``: strictly later than now."""
        return self.backoff_expiry(key) > now

    def _requeue(self, pod: QueuedPod, now: float) -> None:
        pod.where = BACKOFF if self.is_backing_off(pod.key, now) else ACTIVE

    # -- events and timers ------------------------------------------------------

    def capacity_freed(self, now: float) -> None:
        """A scheduled pod was deleted (its GPUs are free). NodeResourcesFit's
        queueing hint returns Queue for that, so every parked pod moves to
        backoffQ or activeQ."""
        for pod in self._pods.values():
            if pod.where == UNSCHEDULABLE:
                self._requeue(pod, now)

    def flush(self, now: float) -> None:
        """Run whichever periodic flushes are due at ``now``.

        The real timers tick every ``interval`` from scheduler start; a pass
        that jumps over several ticks runs each flush once, at ``now``, which
        is what the goroutine would have done by then.
        """
        p = self.params
        if now >= self._next_backoff_flush:
            for pod in self._pods.values():
                if pod.where == BACKOFF and not self.is_backing_off(pod.key, now):
                    pod.where = ACTIVE
            self._next_backoff_flush = _next_tick(now, p.backoff_flush_interval)
        if now >= self._next_unschedulable_flush:
            for pod in self._pods.values():
                if pod.where == UNSCHEDULABLE and now - pod.timestamp > p.max_in_unschedulable:
                    self._requeue(pod, now)
            self._next_unschedulable_flush = _next_tick(now, p.unschedulable_flush_interval)

    def reconcile(self, pending: Iterable[Key], now: float) -> None:
        """Make membership match the caller's pending set: new keys are added,
        keys no longer pending (bound or deleted elsewhere) are dropped."""
        wanted = list(pending)
        wanted_set = set(wanted)
        for key in [k for k in self._pods if k not in wanted_set]:
            self.remove(key)
        for key in wanted:
            self.add(key, now)


def _next_tick(now: float, interval: float) -> float:
    return (math.floor(now / interval) + 1) * interval
