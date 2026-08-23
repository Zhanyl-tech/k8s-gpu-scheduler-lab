"""Replay a trace against a real Kubernetes control plane.

What is real here: the API server, the scheduler being measured, the Pod and
Node objects, and every binding decision. What is not real: the kubelets, the
GPUs, and the execution of any workload. See docs/limitations.md.

Time is compressed. A trace spanning eleven hours is replayed in minutes by
dividing every timestamp and duration by ``speedup``. Nothing in the scheduling
path is time-dependent at the resolution this changes — but it does mean the
lab cannot observe anything that depends on wall-clock (leases, backoff
ceilings, controller resync periods), which is recorded as a limitation rather
than worked around.

Job completion is driven by this runner deleting the pod when its scaled
duration elapses, not by kwok. kwok moves a pod to Running; the lifetime is
ours to control, which is what makes the replay deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .baselines import POLICIES, PendingPod
from .model import Fleet, Job, Observation, PodEvent

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


def _retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call an API method, backing off on transient server-side failures."""
    from kubernetes.client.exceptions import ApiException

    delay = 0.5
    for attempt in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except ApiException as exc:
            if exc.status not in RETRYABLE or attempt == RETRIES - 1:
                raise
        except (OSError, TimeoutError):
            if attempt == RETRIES - 1:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 8.0)
    raise RuntimeError("unreachable")


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


def _scheduler_name(config: str) -> str:
    """K0 uses the stock scheduler; degenerate configs use their own name.

    A pod naming a scheduler that does not exist is simply never bound, which
    is exactly what we want: the binder in this process is that scheduler, and
    kube-scheduler must not race it.
    """
    return "default-scheduler" if config == "K0" else f"k8slab-{config.lower()}"


def priority_class_name(priority: int) -> str:
    return f"k8slab-p{priority}"


def pod_manifest(job: Job, index: int, config: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"j{job.job_id}-p{index}",
            "namespace": namespace,
            "labels": {
                "k8slab.io/job": str(job.job_id),
                "k8slab.io/pod-index": str(index),
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
class _Live:
    key: tuple[int, int]
    node: str
    gpus: int
    #: Simulated time at which the pod should be deleted.
    ends_at: float
    name: str


@dataclass
class Runner:
    fleet: Fleet
    jobs: list[Job]
    cfg: RunConfig
    _events: dict[tuple[int, int], PodEvent] = field(default_factory=dict)
    _samples: list[tuple[float, dict[str, int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for job in self.jobs:
            for i in range(job.gang_size):
                self._events[(job.job_id, i)] = PodEvent(job_id=job.job_id, pod_index=i)

    # -- kubernetes plumbing -------------------------------------------------
    def _client(self) -> Any:
        from kubernetes import client
        from kubernetes import config as kconfig

        kconfig.load_kube_config()
        return client.CoreV1Api()

    def _ensure_namespace(self, api: Any) -> None:
        from kubernetes.client.exceptions import ApiException

        # A namespace still Terminating accepts the create (409) but rejects
        # every pod inside it with a 403. Wait it out rather than spinning.
        for _ in range(120):
            try:
                ns = api.read_namespace(self.cfg.namespace)
            except ApiException as exc:
                if exc.status != 404:
                    raise
                break
            if ns.status.phase != "Terminating":
                return
            time.sleep(1.0)
        try:
            api.create_namespace(
                {"apiVersion": "v1", "kind": "Namespace",
                 "metadata": {"name": self.cfg.namespace}}
            )
        except ApiException as exc:
            if exc.status != 409:
                raise

    def _ensure_priority_classes(self) -> None:
        """One PriorityClass per distinct non-zero priority in the trace."""
        from kubernetes import client
        from kubernetes.client.exceptions import ApiException

        api = client.SchedulingV1Api()
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
                        # Preemption is its own configuration, not an
                        # uncontrolled variable in every other one.
                        "preemptionPolicy": "Never",
                        "globalDefault": False,
                        "description": "k8s-gpu-scheduler-lab trace priority",
                    }
                )
            except ApiException as exc:
                if exc.status != 409:
                    raise

    def _log(self, msg: str) -> None:
        if not self.cfg.quiet:
            print(msg, flush=True)

    # -- the replay ----------------------------------------------------------
    def run(self) -> Observation:
        api = self._client()
        self._ensure_namespace(api)
        self._ensure_priority_classes()

        capacity = {n: self.fleet.gpus_of(n) for n in self.fleet.node_names()}
        policy = POLICIES.get(self.cfg.config)
        by_id = {j.job_id: j for j in self.jobs}

        submitted: set[tuple[int, int]] = set()
        live: dict[tuple[int, int], _Live] = {}
        started = time.monotonic()
        last_submit = max(j.submit_time for j in self.jobs)
        stalled = 0
        placed_before = 0
        misses: dict[tuple[int, int], int] = {}

        while True:
            sim_now = (time.monotonic() - started) * self.cfg.speedup
            if sim_now > self.cfg.max_sim_seconds:
                self._log(f"  ! horizon {self.cfg.max_sim_seconds}s reached; stopping")
                break

            # 1. submit arrivals
            for job in self.jobs:
                if job.submit_time > sim_now:
                    continue
                for i in range(job.gang_size):
                    key = (job.job_id, i)
                    if key in submitted:
                        continue
                    _retry(
                        api.create_namespaced_pod,
                        self.cfg.namespace,
                        pod_manifest(job, i, self.cfg.config, self.cfg.namespace),
                    )
                    submitted.add(key)

            # 2. observe bindings
            pods = _retry(
                api.list_namespaced_pod,
                self.cfg.namespace,
                label_selector=f"k8slab.io/config={self.cfg.config}",
            ).items
            present = {p.metadata.name for p in pods}
            # A submitted pod that is absent from the API and that this runner
            # did not delete has been destroyed by the control plane. Counting
            # it as "still pending" would silently understate the loss, which is
            # exactly how the first run reported 22 phantom unschedulable pods.
            for key in submitted:
                ev = self._events[key]
                if ev.end_time is not None or ev.preempted:
                    continue
                if f"j{key[0]}-p{key[1]}" in present:
                    misses.pop(key, None)
                    continue
                misses[key] = misses.get(key, 0) + 1
                if misses[key] >= 2:
                    ev.preempted = True
                    ev.end_time = sim_now
                    live.pop(key, None)
            free = dict(capacity)
            for pod in pods:
                pod_key = _key_of(pod)
                if pod_key is None:
                    continue
                node = pod.spec.node_name
                if not node:
                    continue
                ev = self._events[pod_key]
                if ev.scheduled_time is None:
                    ev.scheduled_time = sim_now
                    ev.start_time = sim_now
                    ev.node = node
                    job = by_id[pod_key[0]]
                    live[pod_key] = _Live(
                        key=pod_key,
                        node=node,
                        gpus=job.gpus,
                        ends_at=sim_now + job.duration,
                        name=pod.metadata.name,
                    )
                if pod_key in live:
                    free[node] = free.get(node, 0) - live[pod_key].gpus

            self._samples.append((sim_now, dict(free)))

            # 3. bind, if this config's scheduler is us
            if policy is not None:
                pending = [
                    PendingPod(
                        job_id=k[0],
                        pod_index=k[1],
                        gpus=by_id[k[0]].gpus,
                        submit_time=by_id[k[0]].submit_time,
                        priority=by_id[k[0]].priority,
                        gang_size=by_id[k[0]].gang_size,
                    )
                    for k in submitted
                    if self._events[k].scheduled_time is None
                ]
                if pending:
                    import random as _random

                    for pod_req, node in policy(pending, free, _random.Random(0)):
                        _bind(api, self.cfg.namespace,
                              f"j{pod_req.job_id}-p{pod_req.pod_index}", node)

            # 4. retire finished pods
            for key, entry in list(live.items()):
                if entry.ends_at <= sim_now:
                    _delete(api, self.cfg.namespace, entry.name)
                    # The job ran for exactly its trace duration. Recording the
                    # tick on which this loop noticed instead would add up to
                    # one poll interval of phantom GPU time per pod -- ~12% of
                    # total GPU-hours at speedup 400, which is larger than any
                    # effect this lab is trying to measure. The pod object does
                    # linger until this delete lands, so the scheduler sees the
                    # capacity return up to one tick late; that lag is real and
                    # is recorded in docs/limitations.md.
                    self._events[key].end_time = entry.ends_at
                    live.pop(key, None)

            if int(sim_now) // 600 != int(sim_now - self.cfg.speedup * TICK) // 600:
                placed = sum(1 for e in self._events.values() if e.scheduled_time is not None)
                finished = sum(1 for e in self._events.values() if e.end_time is not None)
                self._log(
                    f"    t+{sim_now / 3600:5.2f}h  submitted={len(submitted):4d} "
                    f"placed={placed:4d} running={len(live):3d} done={finished:4d}"
                )

            placed_now = sum(1 for e in self._events.values() if e.scheduled_time is not None)
            all_submitted = len(submitted) == sum(j.gang_size for j in self.jobs)
            if all_submitted and not live and placed_now == placed_before:
                stalled += 1
            else:
                stalled = 0
            placed_before = placed_now

            done = all(e.end_time is not None for e in self._events.values())
            if done and sim_now > last_submit:
                break
            if stalled >= self.cfg.stall_ticks:
                unplaced = [k for k, e in self._events.items() if e.scheduled_time is None]
                self._log(
                    f"  ! stalled: {len(unplaced)} pod(s) unschedulable on an idle "
                    f"fleet; recorded as never-scheduled"
                )
                break
            time.sleep(TICK)

        lost = sum(1 for e in self._events.values() if e.preempted)
        if lost:
            self._log(f"  ! {lost} pod(s) destroyed by the control plane (preemption)")
        horizon = (time.monotonic() - started) * self.cfg.speedup
        for ev in self._events.values():
            if ev.start_time is not None and ev.end_time is None:
                ev.end_time = horizon
        self._samples.append((horizon, {n: capacity[n] for n in capacity}))

        return Observation(
            config=self.cfg.config,
            fleet=self.fleet,
            jobs=self.jobs,
            pods=list(self._events.values()),
            gpu_free_samples=self._samples,
            horizon=horizon,
            measured_on_cluster=True,
        )


def _key_of(pod: Any) -> tuple[int, int] | None:
    labels = pod.metadata.labels or {}
    try:
        return int(labels["k8slab.io/job"]), int(labels["k8slab.io/pod-index"])
    except (KeyError, ValueError):
        return None


def _bind(api: Any, namespace: str, pod_name: str, node: str) -> None:
    """Bind a pod to a node — the one privileged act a scheduler performs."""
    from kubernetes.client.exceptions import ApiException

    body = {
        "apiVersion": "v1",
        "kind": "Binding",
        "metadata": {"name": pod_name, "namespace": namespace},
        "target": {"apiVersion": "v1", "kind": "Node", "name": node},
    }
    try:
        _retry(
            api.create_namespaced_pod_binding,
            pod_name, namespace, body, _preload_content=False,
        )
    except ApiException as exc:
        # 409 means someone bound it first; harmless and expected under races.
        if exc.status not in (404, 409):
            raise


def _delete(api: Any, namespace: str, name: str) -> None:
    from kubernetes.client.exceptions import ApiException

    try:
        _retry(api.delete_namespaced_pod, name, namespace, grace_period_seconds=0)
    except ApiException as exc:
        if exc.status != 404:
            raise
