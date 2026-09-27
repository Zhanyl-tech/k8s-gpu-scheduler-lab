"""Replay a trace against a real Kubernetes control plane.

What is real here: the API server, the scheduler being measured, the Pod and
Node objects, and every binding decision. What is not real: the kubelets, the
GPUs, and the execution of any workload. See docs/limitations.md.

Time is compressed. A trace spanning eleven hours is replayed in minutes by
dividing every timestamp and duration by ``speedup``. The control plane's own
real-time constants (backoff, the unschedulable-pool timeout) are scaled to
match by the KubeSchedulerConfiguration ``make up`` generates
(:mod:`k8slab.timescale`); what cannot be scaled is recorded in every results
file.

Job completion is driven by this runner deleting the pod when its scaled
duration elapses, not by kwok. kwok moves a bound pod to Running (after the
startup delay its Stage is configured with); the lifetime is ours to control.
Trace runtime counts from the first poll that shows the pod Running, not from
the bind.

Phase 2. Everything the runner talks to is injected -- the API object, the
clock and the sleep -- so the whole replay loop is tested against a fake
cluster (tests/test_runner.py). **No run of this module against a real cluster
has been made since the Phase 2 changes**; the fake is a model of the API
surface, not proof of it.
"""

from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .baselines import SPECS, PendingPod
from .binder import Binder
from .execution import Attempt
from .model import Eviction, Fleet, Job, Observation, PodEvent, RunInfo
from .preemption import RunningPod
from .queueing import QueueParams
from .timescale import TimeScale
from .topology import (
    DEFAULT_FACTORS,
    JobPlacement,
    PenaltyFactors,
    Topology,
    from_labels,
    placement_factor,
)

NAMESPACE = "k8slab"
GPU_RESOURCE = "nvidia.com/gpu"
#: Real seconds between control loop passes.
TICK = 1.0

#: Status codes worth retrying. etcd under a few thousand pod writes returns
#: 500 "request timed out" often enough to kill a long replay, and losing a
#: fifteen-minute run to a transient write is not acceptable in a harness whose
#: entire purpose is producing comparable numbers.
RETRYABLE = frozenset({429, 500, 502, 503, 504})
RETRIES = 6

#: Every pod this lab creates carries this label; used to clean up.
LAB_POD_LABEL = "k8slab.io/job"

Key = tuple[int, int]


def _status_of(exc: BaseException) -> int | None:
    """HTTP status of a Kubernetes API error.

    Duck-typed (``ApiException.status``) so that the runner, and the fakes it
    is tested with, do not need the optional ``kubernetes`` client installed.
    """
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _retry(
    fn: Any, *args: Any, sleep: Callable[[float], None] = time.sleep, **kwargs: Any
) -> Any:
    """Call an API method, backing off on transient server-side failures."""
    delay = 0.5
    for attempt in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except (OSError, TimeoutError):
            if attempt == RETRIES - 1:
                raise
        except Exception as exc:
            status = _status_of(exc)
            if status is None or status not in RETRYABLE or attempt == RETRIES - 1:
                raise
        sleep(delay)
        delay = min(delay * 2, 8.0)
    raise RuntimeError("unreachable")


class KubernetesApi:
    """The slice of the official client the runner uses, behind one object.

    Tests substitute a fake with the same methods and the same attribute
    shapes (``pod.metadata.name``, ``pod.spec.node_name``, ``pod.status.phase``,
    ``pod.metadata.deletion_timestamp``, ``node.status.capacity``).
    """

    def __init__(self) -> None:
        from kubernetes import client
        from kubernetes import config as kconfig

        kconfig.load_kube_config()
        self._core = client.CoreV1Api()
        self._sched = client.SchedulingV1Api()

    def read_namespace(self, name: str) -> Any:
        return self._core.read_namespace(name)

    def create_namespace(self, body: dict[str, Any]) -> Any:
        return self._core.create_namespace(body)

    def create_priority_class(self, body: dict[str, Any]) -> Any:
        return self._sched.create_priority_class(body)

    def list_node(self, label_selector: str) -> Any:
        return self._core.list_node(label_selector=label_selector)

    def create_namespaced_pod(self, namespace: str, body: dict[str, Any]) -> Any:
        return self._core.create_namespaced_pod(namespace, body)

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> Any:
        return self._core.list_namespaced_pod(namespace, label_selector=label_selector)

    def create_namespaced_pod_binding(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> Any:
        return self._core.create_namespaced_pod_binding(
            name, namespace, body, _preload_content=False
        )

    def delete_namespaced_pod(self, name: str, namespace: str, grace_period_seconds: int) -> Any:
        return self._core.delete_namespaced_pod(
            name, namespace, grace_period_seconds=grace_period_seconds
        )


@dataclass
class RunConfig:
    config: str
    speedup: float = 60.0
    namespace: str = NAMESPACE
    #: Give up if the queue has not drained after this much *simulated* time.
    max_sim_seconds: float = 48 * 3600.0
    #: Consecutive idle passes (nothing running, nothing newly placed, every pod
    #: submitted) before declaring the remainder unschedulable and stopping. A
    #: run that cannot place a pod on an empty fleet will never place it.
    stall_ticks: int = 15
    quiet: bool = False
    #: Harness settings (queue model, startup delay, topology mode, grace,
    #: gates). Its ``speedup`` is overwritten with this config's.
    run: RunInfo = field(default_factory=RunInfo)
    #: ASSUMED topology factors, applied only with topology_penalty=extend.
    factors: PenaltyFactors = DEFAULT_FACTORS
    #: Real seconds to wait for leftover lab pods to disappear before a run.
    cleanup_timeout: float = 120.0


def _scheduler_name(config: str) -> str:
    """K0 uses the stock scheduler; degenerate configs use their own name.

    A pod naming a scheduler that does not exist is simply never bound, which
    is exactly what we want: the binder in this process is that scheduler, and
    kube-scheduler must not race it.
    """
    return "default-scheduler" if config == "K0" else f"k8slab-{config.lower()}"


def priority_class_name(priority: int) -> str:
    return f"k8slab-p{priority}"


def pod_name(job_id: int, index: int, attempt: int = 0) -> str:
    """Attempt 0 keeps Phase 1's name; a requeued pod gets a fresh name, so it
    never collides with its predecessor while that one is still terminating."""
    base = f"j{job_id}-p{index}"
    return base if attempt == 0 else f"{base}-a{attempt}"


def pod_manifest(
    job: Job, index: int, config: str, namespace: str, attempt: int = 0
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name(job.job_id, index, attempt),
            "namespace": namespace,
            "labels": {
                "k8slab.io/job": str(job.job_id),
                "k8slab.io/pod-index": str(index),
                "k8slab.io/attempt": str(attempt),
                "k8slab.io/account": job.account,
                "k8slab.io/gang-size": str(job.gang_size),
                "k8slab.io/config": config,
            },
        },
        "spec": {
            "schedulerName": _scheduler_name(config),
            # Priority must arrive as a PriorityClass name: the priority
            # admission controller computes spec.priority itself and rejects a
            # pod that sets the integer directly. Learned from a 403.
            **({"priorityClassName": priority_class_name(job.priority)} if job.priority else {}),
            "restartPolicy": "Never",
            # Unchanged from Phase 1. The runner's own deletes are immediate;
            # a grace period, where one applies, is modelled by the runner's
            # GPU accounting (docs/limitations.md, "Preemption").
            "terminationGracePeriodSeconds": 0,
            "nodeSelector": {"type": "kwok"},
            "tolerations": [
                {"key": "kwok.x-k8s.io/node", "value": "fake", "effect": "NoSchedule"}
            ],
            "containers": [
                {
                    "name": "work",
                    "image": "registry.k8s.io/pause:3.9",
                    "resources": {
                        "requests": {GPU_RESOURCE: str(job.gpus), "cpu": "1"},
                        "limits": {GPU_RESOURCE: str(job.gpus)},
                    },
                }
            ],
        },
    }


@dataclass
class _Track:
    """The runner's view of one pod key (job, index) across its attempts."""

    key: Key
    attempt: int = 0
    submitted: bool = False
    #: The current attempt has finished for good (completed or destroyed).
    ended: bool = False
    misses: int = 0
    missing_since: float | None = None
    #: Set when the bind is observed.
    bound: Attempt | None = None
    #: The ASSUMED factor has been applied to this placement.
    extended: bool = False

    @property
    def name(self) -> str:
        return pod_name(self.key[0], self.key[1], self.attempt)

    @property
    def holding(self) -> bool:
        return self.bound is not None and not self.ended


@dataclass
class _Lock:
    release_at: float
    node: str
    gpus: int
    #: A Terminating pod object to force-delete when the grace period ends.
    pod: str | None = None


@dataclass
class Runner:
    fleet: Fleet
    jobs: list[Job]
    cfg: RunConfig
    #: The Kubernetes API (:class:`KubernetesApi` when ``None``).
    api: Any = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _events: dict[Key, PodEvent] = field(default_factory=dict)
    _samples: list[tuple[float, dict[str, int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.info = dataclasses.replace(self.cfg.run, speedup=self.cfg.speedup)
        self._tracks: dict[Key, _Track] = {}
        self._work: dict[Key, float] = {}
        self._by_id = {j.job_id: j for j in self.jobs}
        self._fleet_nodes = frozenset(self.fleet.node_names())
        #: Keys with a live (submitted) attempt, as a set -- iterated in set
        #: order to build the pending list exactly as Phase 1 did.
        self._submitted: set[Key] = set()
        self._locks: list[_Lock] = []
        self._evictions: list[Eviction] = []
        self._extension = 0.0
        self._extension_by_account: dict[str, float] = {}
        self._notes: list[str] = []
        self._conflicts = 0
        for job in self.jobs:
            for i in range(job.gang_size):
                key = (job.job_id, i)
                self._events[key] = PodEvent(job_id=job.job_id, pod_index=i)
                self._tracks[key] = _Track(key=key)
                self._work[key] = job.duration

    def _stretched(self, key: Key, gpu_seconds: float) -> None:
        """Book topology extension, in total and against the job's account
        (fairness takes each account's share back out of its delivered time)."""
        self._extension += gpu_seconds
        account = self._by_id[key[0]].account
        self._extension_by_account[account] = (
            self._extension_by_account.get(account, 0.0) + gpu_seconds
        )

    # -- kubernetes plumbing -------------------------------------------------
    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return _retry(fn, *args, sleep=self.sleep, **kwargs)

    def _ensure_namespace(self, api: Any) -> None:
        # A namespace still Terminating accepts the create (409) but rejects
        # every pod inside it with a 403. Wait it out rather than spinning.
        for _ in range(120):
            try:
                ns = api.read_namespace(self.cfg.namespace)
            except Exception as exc:
                if _status_of(exc) != 404:
                    raise
                break
            if ns.status.phase != "Terminating":
                return
            self.sleep(1.0)
        try:
            api.create_namespace(
                {"apiVersion": "v1", "kind": "Namespace",
                 "metadata": {"name": self.cfg.namespace}}
            )
        except Exception as exc:
            if _status_of(exc) != 409:
                raise

    def _ensure_priority_classes(self, api: Any) -> None:
        """One PriorityClass per distinct non-zero priority in the trace."""
        for value in sorted({j.priority for j in self.jobs if j.priority}):
            try:
                api.create_priority_class(
                    {
                        "apiVersion": "scheduling.k8s.io/v1",
                        "kind": "PriorityClass",
                        "metadata": {"name": priority_class_name(value)},
                        "value": value,
                        # Phase 1 measures placement and ordering, not
                        # preemption. With the default policy kube-scheduler
                        # DELETES lower-priority pods to make room -- 66 of 476
                        # in the first run -- and a bare pod has no controller
                        # to recreate it, so the job silently vanishes.
                        # Preemption is its own configuration (D-preempt, whose
                        # evictions this runner performs itself), not an
                        # uncontrolled variable in every other one.
                        "preemptionPolicy": "Never",
                        "globalDefault": False,
                        "description": "k8s-gpu-scheduler-lab trace priority",
                    }
                )
            except Exception as exc:
                if _status_of(exc) != 409:
                    raise

    def _read_topology(self, api: Any) -> Topology:
        """Score placement against the Node labels the cluster actually has.

        Refuses to run against a cluster that does not carry exactly this
        fleet: a pod bound to a node outside the fleet would otherwise fail
        scoring after a fifteen-minute replay. That includes EXTRA kwok nodes,
        not only missing ones: ``make up`` on an existing cluster applies the
        new fleet's nodes and deletes none, so ``make up FLEET=a`` then
        ``make up FLEET=b`` leaves a's extra nodes behind, and K0's pods
        (``nodeSelector: type=kwok``) can bind to them.
        """
        nodes = self._call(api.list_node, label_selector="type=kwok").items
        labels = {n.metadata.name: dict(n.metadata.labels or {}) for n in nodes}
        cluster_gpus = {
            n.metadata.name: int((n.status.capacity or {}).get(GPU_RESOURCE, 0)) for n in nodes
        }
        missing = [n for n in self.fleet.node_names() if n not in labels]
        if missing:
            raise RuntimeError(
                f"the cluster has no kwok node(s) {missing[:5]} from fleet "
                f"{self.fleet.name!r}; run `make up FLEET=...` with the same fleet"
            )
        extra = sorted(set(labels) - set(self.fleet.node_names()))
        if extra:
            raise RuntimeError(
                f"the cluster has {len(extra)} kwok node(s) outside fleet "
                f"{self.fleet.name!r} (e.g. {extra[:5]}), left by a `make up` with another "
                f"FLEET; K0 could bind pods to them. Run `make down && make up FLEET=...`"
            )
        capacity = {n: self.fleet.gpus_of(n) for n in self.fleet.node_names()}
        wrong = {n: cluster_gpus[n] for n in capacity if cluster_gpus[n] != capacity[n]}
        if wrong:
            raise RuntimeError(f"node GPU capacity differs from the fleet file: {wrong}")
        try:
            topo = from_labels({n: labels[n] for n in capacity}, capacity)
        except ValueError as exc:
            raise RuntimeError(
                f"{exc}. The nodes predate topology labels; re-run `make up` to "
                f"re-apply the node manifests"
            ) from None
        # The labels always exist -- `k8slab nodes` writes derived default
        # rack/switch labels even for a fleet that declares no topology -- so
        # their presence says nothing about whether the topology was DECLARED.
        # That is the fleet file's fact; without this, every cluster run
        # claimed a declared topology and the report's "flat defaults" note was
        # suppressed for cluster rows.
        return dataclasses.replace(topo, declared=self.fleet.topology_declared)

    def _cleanup(self, api: Any) -> None:
        """Delete every lab pod left in the namespace and wait until gone.

        A previous run that stopped on a stall leaves never-scheduled pods
        behind. A K0 leftover would be bound by kube-scheduler into the next
        configuration's run, and a same-named leftover makes the next repeat's
        create fail with 409. Neither may happen silently.
        """
        deadline = self.clock() + self.cfg.cleanup_timeout
        while True:
            pods = self._call(
                api.list_namespaced_pod, self.cfg.namespace, label_selector=LAB_POD_LABEL
            ).items
            if not pods:
                return
            for pod in pods:
                _delete(api, self.cfg.namespace, pod.metadata.name, sleep=self.sleep)
            if self.clock() > deadline:
                raise RuntimeError(
                    f"{len(pods)} lab pod(s) still present in namespace "
                    f"{self.cfg.namespace!r} after {self.cfg.cleanup_timeout:.0f} s"
                )
            self.sleep(TICK)

    def _log(self, msg: str) -> None:
        if not self.cfg.quiet:
            print(msg, flush=True)

    # -- the replay ----------------------------------------------------------
    def run(self) -> Observation:
        api = self.api if self.api is not None else KubernetesApi()
        self._ensure_namespace(api)
        self._ensure_priority_classes(api)
        topo = self._read_topology(api)
        self._cleanup(api)

        speedup = self.cfg.speedup
        capacity = {n: self.fleet.gpus_of(n) for n in self.fleet.node_names()}
        binder: Binder | None = None
        seed = self.info.harness_seed or 0
        if self.cfg.config in SPECS:
            params = (
                QueueParams.from_timescale(TimeScale(speedup))
                if self.info.queue_model == "kube" else None
            )
            binder = Binder(self.cfg.config, self.info, random.Random(seed), params,
                            window=speedup * TICK)

        started = self.clock()
        last_submit = max(j.submit_time for j in self.jobs)
        stalled = 0
        placed_before = 0

        while True:
            sim_now = (self.clock() - started) * speedup
            if sim_now > self.cfg.max_sim_seconds:
                self._log(f"  ! horizon {self.cfg.max_sim_seconds}s reached; stopping")
                break

            # 1. submit arrivals (and requeued pods)
            self._submit(api, sim_now)

            # 2. observe bindings, Running transitions and deletions
            pods = self._call(
                api.list_namespaced_pod,
                self.cfg.namespace,
                label_selector=f"k8slab.io/config={self.cfg.config}",
            ).items
            self._observe(pods, sim_now)
            if self.info.topology_penalty == "extend":
                self._extend(topo, sim_now)

            # 3. release grace locks that are due BEFORE sampling and binding.
            # A lock exists only in this runner's accounting: D-preempt deleted
            # its victim at eviction time, and a control-plane deletion is
            # already gone (or Terminating, force-deleted here). So nothing in
            # the API holds those GPUs past release_at, unlike a finished pod,
            # whose object really lingers until _retire's delete lands. Doing
            # this after the bind pass kept every lock a full poll too long --
            # in the samples and for the binder -- and understated the
            # GPU-time a grace period withholds. The reference model releases
            # locks before its sample and bind too.
            # _observe never frees GPUs: a destroyed pod's stay locked for
            # the grace period, and a finished pod is retired in step 5.
            freed = self._release_locks(api, sim_now)

            free = self._free(capacity, sim_now)
            self._samples.append((sim_now, dict(free)))

            # 4. bind, if this config's scheduler is us
            if binder is not None:
                if freed:
                    binder.capacity_freed(sim_now)
                self._bind_pass(api, binder, free, sim_now, seed)

            # 5. retire finished pods
            freed = self._retire(api, sim_now)
            if freed and binder is not None:
                binder.capacity_freed(sim_now)

            if int(sim_now) // 600 != int(sim_now - speedup * TICK) // 600:
                placed = sum(1 for e in self._events.values() if e.scheduled_time is not None)
                finished = sum(1 for t in self._tracks.values() if t.ended)
                holding = sum(1 for t in self._tracks.values() if t.holding)
                self._log(
                    f"    t+{sim_now / 3600:5.2f}h  submitted={len(self._submitted):4d} "
                    f"placed={placed:4d} running={holding:3d} done={finished:4d}"
                )

            placed_now = sum(1 for e in self._events.values() if e.scheduled_time is not None)
            all_submitted = all(t.submitted or t.ended for t in self._tracks.values())
            # With the admission gate, an unplaceable pod holds the gate shut
            # and the rest are never submitted; that is a stall too.
            gate_blocked = self.info.admission_gate and any(
                t.submitted and t.bound is None and not t.ended for t in self._tracks.values()
            )
            # A pending grace lock is capacity about to return: not idle yet.
            holding_any = any(t.holding for t in self._tracks.values()) or bool(self._locks)
            if (all_submitted or gate_blocked) and not holding_any \
                    and placed_now == placed_before:
                stalled += 1
            else:
                stalled = 0
            placed_before = placed_now

            done = all(t.ended for t in self._tracks.values())
            if done and sim_now > last_submit:
                break
            if stalled >= self.cfg.stall_ticks:
                unplaced = [k for k, e in self._events.items() if e.scheduled_time is None]
                self._log(
                    f"  ! stalled: {len(unplaced)} pod(s) unschedulable on an idle "
                    f"fleet; recorded as never-scheduled"
                )
                break
            self.sleep(TICK)

        lost = sum(1 for e in self._events.values() if e.preempted)
        if lost:
            self._log(f"  ! {lost} pod(s) destroyed by the control plane (preemption)")
        horizon = (self.clock() - started) * speedup
        for track in self._tracks.values():
            attempt = track.bound
            if attempt is not None and not track.ended and attempt.stretched:
                self._stretched(track.key, attempt.gpus * attempt.stretch_seconds(horizon))
        for ev in self._events.values():
            if ev.start_time is not None and ev.end_time is None:
                ev.end_time = horizon
        self._samples.append((horizon, {n: capacity[n] for n in capacity}))
        if self._conflicts:
            self._notes.append(
                f"{self._conflicts} sample(s) had a node where a grace lock overlapped a "
                f"new binding: the control plane freed the GPUs before the modelled "
                f"grace period ended; free GPUs were clamped at 0 there"
            )
        self._cleanup_quietly(api)

        return Observation(
            config=self.cfg.config,
            fleet=self.fleet,
            jobs=self.jobs,
            pods=list(self._events.values()),
            gpu_free_samples=self._samples,
            horizon=horizon,
            measured_on_cluster=True,
            topology=topo,
            run=self.info,
            evictions=self._evictions,
            extension_gpu_seconds=self._extension,
            extension_by_account=self._extension_by_account,
            notes=self._notes,
        )

    def _cleanup_quietly(self, api: Any) -> None:
        try:
            self._cleanup(api)
        except RuntimeError as exc:
            self._log(f"  ! cleanup: {exc}")

    # -- steps ----------------------------------------------------------------
    def _submit(self, api: Any, sim_now: float) -> None:
        """Create every due pod; with the admission gate, at most one pod is
        outstanding (submitted, bind not yet observed) at any time."""
        gate = self.info.admission_gate
        busy = gate and any(
            t.submitted and t.bound is None and not t.ended for t in self._tracks.values()
        )
        for job in self.jobs:
            if job.submit_time > sim_now:
                continue
            for i in range(job.gang_size):
                track = self._tracks[(job.job_id, i)]
                if track.submitted or track.ended:
                    continue
                if busy:
                    return
                self._call(
                    api.create_namespaced_pod,
                    self.cfg.namespace,
                    pod_manifest(job, i, self.cfg.config, self.cfg.namespace, track.attempt),
                )
                track.submitted = True
                self._submitted.add(track.key)
                busy = gate

    def _observe(self, pods: list[Any], sim_now: float) -> None:
        """Update every track from one pod listing.

        It never frees GPUs, so it reports nothing: a pod the control plane
        destroyed keeps its GPUs locked for the grace period
        (:meth:`_release_locks` frees them), and a finished pod is freed by
        :meth:`_retire`. Only this runner's own deletes (``_retire``,
        ``_evict``) remove the current attempt's pod; ``_retire`` also ends the
        track, so an ended track is skipped and an absent pod of a live track
        was removed by the control plane.
        """
        present: dict[Key, Any] = {}
        for pod in pods:
            key = _key_of(pod)
            if key is None or key not in self._tracks:
                continue
            if _attempt_of(pod) == self._tracks[key].attempt:
                present[key] = pod
        for key in list(self._submitted):
            track = self._tracks[key]
            if track.ended:
                continue
            ev = self._events[key]
            pod = present.get(key)
            if pod is None:
                # A submitted pod that is absent from the API and that this
                # runner did not delete has been destroyed by the control
                # plane. Counting it as "still pending" would silently
                # understate the loss, which is exactly how the first run
                # reported 22 phantom unschedulable pods.
                track.misses += 1
                if track.missing_since is None:
                    track.missing_since = sim_now
                if track.misses >= 2:
                    self._destroyed(track, track.missing_since, None)
                continue
            track.misses = 0
            track.missing_since = None
            if getattr(pod.metadata, "deletion_timestamp", None):
                if track.bound is not None and track.bound.finished_by(sim_now):
                    # Its work ended before this poll saw it Terminating: it
                    # completed. _retire below records it at its end and
                    # force-deletes the object -- as it already does for a
                    # finished pod that vanished (retired on its first miss).
                    # Recording it destroyed charged Running time past its
                    # duration as lost.
                    continue
                # Terminating, and not by our hand: the control plane is
                # evicting it. Its GPUs stay held for the grace period.
                self._destroyed(track, sim_now, pod.metadata.name)
                continue
            node = pod.spec.node_name
            if node and track.bound is None:
                if node not in self._fleet_nodes:
                    # Preflight refuses such clusters; a node added mid-run is
                    # stopped here rather than after the replay, in scoring.
                    raise RuntimeError(
                        f"pod {pod.metadata.name} was bound to {node!r}, which is not in "
                        f"fleet {self.fleet.name!r}"
                    )
                job = self._by_id[key[0]]
                ev.scheduled_time = sim_now
                ev.node = node
                track.bound = Attempt(
                    key=key, node=node, gpus=job.gpus, priority=job.priority,
                    bind_time=sim_now, work=self._work[key],
                )
            attempt = track.bound
            if attempt is not None and not attempt.started and _phase(pod) == "Running":
                attempt.start(sim_now)
                ev.start_time = sim_now

    def _destroyed(self, track: _Track, when: float, terminating: str | None) -> None:
        """The control plane removed this pod. Phase 1 recorded the loss; Phase
        2 also keeps its GPUs held for the grace period in the samples, so
        nothing is free until the lock releases."""
        ev = self._events[track.key]
        ev.preempted = True
        ev.end_time = when
        track.ended = True
        attempt = track.bound
        if attempt is None:
            return
        release = when + self.info.grace_seconds
        self._locks.append(_Lock(release, attempt.node, attempt.gpus, terminating))
        run_wall = (
            max(0.0, when - attempt.start_time) if attempt.start_time is not None else 0.0
        )
        self._evictions.append(
            Eviction(
                job_id=track.key[0], pod_index=track.key[1], node=attempt.node,
                gpus=attempt.gpus, bind_time=attempt.bind_time,
                start_time=attempt.start_time, evict_time=when, release_time=release,
                reason="control-plane" + (" (terminating)" if terminating else " (vanished)"),
                lost_seconds=run_wall, retained_seconds=0.0, requeued=False,
            )
        )
        # No topology extension for this attempt, stretched or not: none of
        # its Running time is delivered (retained 0; metrics count it all as
        # lost), and extension is by definition *delivered* time that exists
        # only because of a stretch. Adding it counted the same seconds in both
        # topology_extension_gpu_hours and preempted_gpu_hours_lost.

    def _extend(self, topo: Topology, sim_now: float) -> None:
        """topology_penalty=extend: when a job's last pod is bound, stretch
        every member's remaining runtime by the ASSUMED placement factor.

        The factor is computed from every member's node, including members
        that already ended (``bound`` survives _retire and _destroyed): it is
        the job's placement factor, as placement_penalty_mean reports it.
        Skipping any job with an ended member exempted exactly the gangs a
        scheduler failed to co-schedule. Only members still holding GPUs are
        stretched, and a member whose work ended since the last poll is left
        alone by :meth:`~k8slab.execution.Attempt.set_factor` (applying the
        factor used to move its end to this poll). An evicted member has
        ``bound`` reset, so its job waits until it is bound again.
        """
        for job in self.jobs:
            tracks = [self._tracks[(job.job_id, i)] for i in range(job.gang_size)]
            nodes = tuple(t.bound.node for t in tracks if t.bound is not None)
            if len(nodes) != job.gang_size:
                continue
            todo = [t for t in tracks if t.holding and not t.extended]
            if not todo:
                continue
            f = placement_factor(JobPlacement(nodes, job.gpus), topo, self.cfg.factors)
            for t in todo:
                t.extended = True
                if t.bound is not None:
                    t.bound.set_factor(f, sim_now)

    def _free(self, capacity: dict[str, int], sim_now: float) -> dict[str, int]:
        free = dict(capacity)
        for track in self._tracks.values():
            if track.holding and track.bound is not None:
                free[track.bound.node] = free.get(track.bound.node, 0) - track.bound.gpus
        for lock in self._locks:
            free[lock.node] = free.get(lock.node, 0) - lock.gpus
        for node, value in free.items():
            if value < 0:
                self._conflicts += 1
                free[node] = 0
        return free

    def _bind_pass(
        self, api: Any, binder: Binder, free: dict[str, int], sim_now: float, seed: int
    ) -> None:
        pending = [
            PendingPod(
                job_id=k[0],
                pod_index=k[1],
                gpus=self._by_id[k[0]].gpus,
                submit_time=self._by_id[k[0]].submit_time,
                priority=self._by_id[k[0]].priority,
                gang_size=self._by_id[k[0]].gang_size,
            )
            for k in self._submitted
            if self._tracks[k].bound is None and not self._tracks[k].ended
        ]
        if not pending:
            return
        if self.info.queue_model == "none":
            # Phase 1 re-created Random(0) on every pass. Kept (seeded with the
            # harness seed) so that a one-repeat cluster run is Phase 1's binder.
            binder.rng = random.Random(seed)
        # Victims are running attempts only. This pass runs before _retire
        # (the Phase 1 order: a finished pod's object lingers until its delete
        # lands, so its GPUs are still held in `free`), so an attempt whose
        # work ended since the last poll is still "holding". Offering it as a
        # victim evicted finished work: in a one-node replay a 90 s job was
        # evicted 120 s into its run, charged 120 s lost and run again. It
        # completes in _retire below and frees its GPUs for the next pass.
        running = [
            RunningPod(t.key, t.bound.node, t.bound.gpus, t.bound.priority)
            for t in self._tracks.values()
            if t.holding and t.bound is not None and not t.bound.finished_by(sim_now)
        ] if SPECS[self.cfg.config].preemptive else []
        decision = binder.schedule(sim_now, pending, free, running)
        for plan in decision.preemptions:
            reason = f"preempted by j{plan.preemptor[0]}-p{plan.preemptor[1]}"
            for victim in plan.victims:
                self._evict(api, self._tracks[victim], sim_now, reason)
        for pod, node in decision.binds:
            _bind(api, self.cfg.namespace, self._tracks[pod.key].name, node, sleep=self.sleep)

    def _evict(self, api: Any, track: _Track, sim_now: float, reason: str) -> None:
        """D-preempt's eviction: delete the victim, hold its GPUs for the grace
        period in this runner's accounting (the in-process binder is the only
        scheduler for these pods, so it honours the lock), requeue it with its
        work lost except for the checkpoint fraction. An attempt whose work
        has already ended is not evicted: it completes in :meth:`_retire`."""
        attempt = track.bound
        if attempt is None or track.ended or attempt.finished_by(sim_now):
            return
        _delete(api, self.cfg.namespace, track.name, sleep=self.sleep)
        release = sim_now + self.info.grace_seconds
        self._locks.append(_Lock(release, attempt.node, attempt.gpus))
        started = attempt.start_time is not None and attempt.start_time < sim_now
        run_wall = (
            sim_now - attempt.start_time
            if started and attempt.start_time is not None else 0.0
        )
        c = self.info.checkpoint_fraction
        retained_work = c * attempt.work_done(sim_now)
        if attempt.stretched and c > 0:
            self._stretched(track.key, attempt.gpus * (c * run_wall - retained_work))
        self._work[track.key] = attempt.work - retained_work
        self._evictions.append(
            Eviction(
                job_id=track.key[0], pod_index=track.key[1], node=attempt.node,
                gpus=attempt.gpus, bind_time=attempt.bind_time,
                start_time=attempt.start_time if started else None,
                evict_time=sim_now, release_time=release, reason=reason,
                lost_seconds=(1 - c) * run_wall, retained_seconds=c * run_wall,
            )
        )
        ev = self._events[track.key]
        ev.scheduled_time = ev.start_time = ev.end_time = None
        ev.node = None
        track.attempt += 1
        track.submitted = False
        track.bound = None
        track.extended = False
        track.misses = 0
        track.missing_since = None
        self._submitted.discard(track.key)

    def _retire(self, api: Any, sim_now: float) -> bool:
        freed = False
        for track in self._tracks.values():
            attempt = track.bound
            if attempt is None or track.ended or attempt.end is None:
                continue
            end = attempt.end
            if end > sim_now:
                continue
            _delete(api, self.cfg.namespace, track.name, sleep=self.sleep)
            track.ended = True
            # The job ran for exactly its trace duration from the poll that saw
            # it Running. Recording the tick on which this loop noticed instead
            # would add up to one poll interval of phantom GPU time per pod --
            # ~12% of total GPU-hours at speedup 400, which is larger than any
            # effect this lab is trying to measure. The pod object does linger
            # until this delete lands, so the scheduler sees the capacity
            # return up to one tick late; that lag is real and is recorded in
            # docs/limitations.md.
            self._events[track.key].end_time = end
            if attempt.stretched:
                self._stretched(track.key, attempt.gpus * attempt.stretch_seconds(end))
            freed = True
        return freed

    def _release_locks(self, api: Any, sim_now: float) -> bool:
        due = [lk for lk in self._locks if lk.release_at <= sim_now]
        if not due:
            return False
        self._locks = [lk for lk in self._locks if lk.release_at > sim_now]
        for lock in due:
            if lock.pod is not None:
                # kwok has no kubelet and `make up` removes its pod-delete
                # stage, so nothing else will finish a Terminating pod.
                _delete(api, self.cfg.namespace, lock.pod, sleep=self.sleep)
        return True


def _key_of(pod: Any) -> Key | None:
    labels = pod.metadata.labels or {}
    try:
        return int(labels["k8slab.io/job"]), int(labels["k8slab.io/pod-index"])
    except (KeyError, ValueError):
        return None


def _attempt_of(pod: Any) -> int:
    labels = pod.metadata.labels or {}
    try:
        return int(labels.get("k8slab.io/attempt", "0"))
    except ValueError:
        return -1


def _phase(pod: Any) -> str | None:
    status = getattr(pod, "status", None)
    return getattr(status, "phase", None) if status is not None else None


def _bind(
    api: Any, namespace: str, pod_name: str, node: str,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Bind a pod to a node — the one privileged act a scheduler performs."""
    body = {
        "apiVersion": "v1",
        "kind": "Binding",
        "metadata": {"name": pod_name, "namespace": namespace},
        "target": {"apiVersion": "v1", "kind": "Node", "name": node},
    }
    try:
        _retry(api.create_namespaced_pod_binding, pod_name, namespace, body, sleep=sleep)
    except Exception as exc:
        # 409 means someone bound it first; harmless and expected under races.
        if _status_of(exc) not in (404, 409):
            raise


def _delete(
    api: Any, namespace: str, name: str, sleep: Callable[[float], None] = time.sleep
) -> None:
    try:
        _retry(api.delete_namespaced_pod, name, namespace, grace_period_seconds=0, sleep=sleep)
    except Exception as exc:
        if _status_of(exc) != 404:
            raise
