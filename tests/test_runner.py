"""The cluster runner, against a fake API server. No cluster exists here.

``FakeCluster`` models only the API surface the runner uses, with kwok's
behaviour reduced to one rule (a bound pod turns Running ``startup`` real
seconds after its bind) and, optionally, a stand-in for kube-scheduler that
binds default-scheduler pods first-fit. It is a model of the API, not evidence
about a real control plane: these tests show the runner's accounting is right
*given* those behaviours.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any

import pytest

from k8slab import baselines
from k8slab.binder import Binder
from k8slab.execution import Attempt
from k8slab.fleet import load, render_nodes
from k8slab.metrics import compute
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent, RunInfo
from k8slab.runner import TICK, RunConfig, Runner, _retry, pod_manifest, pod_name
from k8slab.topology import JobPlacement, derive, placement_factor

FLEET = load("fleets/default.yaml")


def _v(x: float | None) -> float:
    assert x is not None
    return x


def _ran(p: PodEvent) -> float:
    """Running time of a pod's final attempt."""
    return _v(p.end_time) - _v(p.start_time)


class FakeApiError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


class FakeCluster:
    def __init__(
        self,
        clock: FakeClock,
        *,
        startup: float = 0.0,
        kube_scheduler: bool = False,
        nodes: list[dict[str, Any]] | None = None,
    ) -> None:
        self.clock = clock
        self.startup = startup
        self.kube_scheduler = kube_scheduler
        self.nodes = nodes if nodes is not None else render_nodes(FLEET)
        self.pods: dict[str, SimpleNamespace] = {}
        self.bound_at: dict[str, float] = {}
        self.namespaces: set[str] = set()
        self.priority_classes: dict[str, dict[str, Any]] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.bindings: list[tuple[str, str]] = []
        #: Real time of every bind, never popped (bound_at is, on delete).
        self.bind_times: dict[str, float] = {}
        self.max_unbound = 0

    # -- the API surface -----------------------------------------------------
    def read_namespace(self, name: str) -> Any:
        if name not in self.namespaces:
            raise FakeApiError(404)
        return SimpleNamespace(status=SimpleNamespace(phase="Active"))

    def create_namespace(self, body: dict[str, Any]) -> None:
        self.namespaces.add(body["metadata"]["name"])

    def create_priority_class(self, body: dict[str, Any]) -> None:
        name = body["metadata"]["name"]
        if name in self.priority_classes:
            raise FakeApiError(409)
        self.priority_classes[name] = body

    def list_node(self, label_selector: str) -> Any:
        items = []
        for n in self.nodes:
            if n["metadata"]["labels"].get("type") != "kwok":
                continue
            items.append(SimpleNamespace(
                metadata=SimpleNamespace(name=n["metadata"]["name"],
                                         labels=dict(n["metadata"]["labels"])),
                status=SimpleNamespace(capacity=dict(n["status"]["capacity"])),
            ))
        return SimpleNamespace(items=items)

    def create_namespaced_pod(self, namespace: str, body: dict[str, Any]) -> None:
        name = body["metadata"]["name"]
        if name in self.pods:
            raise FakeApiError(409)
        self.pods[name] = SimpleNamespace(
            metadata=SimpleNamespace(name=name, labels=dict(body["metadata"]["labels"]),
                                     deletion_timestamp=None),
            spec=SimpleNamespace(node_name=None, scheduler_name=body["spec"]["schedulerName"],
                                 gpus=int(body["spec"]["containers"][0]["resources"]
                                          ["requests"]["nvidia.com/gpu"])),
            status=SimpleNamespace(phase="Pending"),
        )
        self.created.append(name)
        self._track_unbound()

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> Any:
        if self.kube_scheduler:
            self._schedule()
        for name, pod in self.pods.items():
            bound = self.bound_at.get(name)
            if bound is not None and self.clock.now - bound >= self.startup:
                pod.status.phase = "Running"
        key, _, value = label_selector.partition("=")
        items = [
            p for p in self.pods.values()
            if key in p.metadata.labels and (not value or p.metadata.labels[key] == value)
        ]
        return SimpleNamespace(items=items)

    def create_namespaced_pod_binding(self, name: str, namespace: str,
                                      body: dict[str, Any]) -> None:
        pod = self.pods.get(name)
        if pod is None:
            raise FakeApiError(404)
        if pod.spec.node_name:
            raise FakeApiError(409)
        node = body["target"]["name"]
        assert self._free()[node] >= pod.spec.gpus, f"overbooked {node}"
        pod.spec.node_name = node
        self.bound_at[name] = self.clock.now
        self.bind_times[name] = self.clock.now
        self.bindings.append((name, node))

    def delete_namespaced_pod(self, name: str, namespace: str,
                              grace_period_seconds: int) -> None:
        if name not in self.pods:
            raise FakeApiError(404)
        del self.pods[name]
        self.bound_at.pop(name, None)
        self.deleted.append(name)

    # -- behaviour -------------------------------------------------------------
    def _free(self) -> dict[str, int]:
        free = {n["metadata"]["name"]: int(n["status"]["capacity"]["nvidia.com/gpu"])
                for n in self.nodes}
        for pod in self.pods.values():
            if pod.spec.node_name:
                free[pod.spec.node_name] -= pod.spec.gpus
        return free

    def _schedule(self) -> None:
        for name in sorted(self.pods):
            pod = self.pods[name]
            if pod.spec.node_name or pod.spec.scheduler_name != "default-scheduler":
                continue
            for node, room in sorted(self._free().items()):
                if room >= pod.spec.gpus:
                    self.create_namespaced_pod_binding(
                        name, "k8slab", {"target": {"name": node}})
                    break

    def _track_unbound(self) -> None:
        unbound = sum(1 for p in self.pods.values() if not p.spec.node_name)
        self.max_unbound = max(self.max_unbound, unbound)

    # -- control-plane actions for tests ------------------------------------------
    def vanish(self, name: str) -> None:
        del self.pods[name]
        self.bound_at.pop(name, None)

    def terminate(self, name: str) -> None:
        self.pods[name].metadata.deletion_timestamp = "now"


SMALL = [
    Job(1, "a", 0.0, 1800.0, gpus=8),
    Job(2, "a", 60.0, 900.0, gpus=4, gang_size=2),
    Job(3, "b", 120.0, 600.0, gpus=1),
    Job(4, "b", 300.0, 1200.0, gpus=2, gang_size=4),
    Job(5, "c", 600.0, 300.0, gpus=4),
    Job(6, "c", 900.0, 2400.0, gpus=8, priority=100),
]


def _run(
    config: str = "D-fifo",
    jobs: list[Job] | None = None,
    *,
    speedup: float = 300.0,
    info: RunInfo | None = None,
    cluster: FakeCluster | None = None,
    clock: FakeClock | None = None,
) -> tuple[Observation, FakeCluster]:
    clock = clock or FakeClock()
    cluster = cluster or FakeCluster(clock)
    cfg = RunConfig(config=config, speedup=speedup, quiet=True, run=info or RunInfo())
    obs = Runner(FLEET, jobs or SMALL, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    return obs, cluster


def _demand(jobs: list[Job]) -> float:
    return math.fsum(j.total_gpus * j.duration for j in jobs) / 3600.0


def _valid_samples(obs: Observation) -> None:
    times = [t for t, _ in obs.gpu_free_samples]
    assert times == sorted(times)
    caps = {n: FLEET.gpus_of(n) for n in FLEET.node_names()}
    for _, free in obs.gpu_free_samples:
        assert set(free) == set(caps) and all(0 <= free[n] <= caps[n] for n in caps)


# ---- the happy path -------------------------------------------------------------------


@pytest.mark.parametrize("queue_model", ["none", "kube"])
@pytest.mark.parametrize("config", ["D-fifo", "D-random", "D-largest", "D-preempt"])
def test_degenerate_config_replays_to_completion(config: str, queue_model: str) -> None:
    info = RunInfo(queue_model=queue_model, speedup=300.0)
    obs, cluster = _run(config, info=info)
    m = compute(obs)
    assert obs.measured_on_cluster and m.jobs_completed == len(SMALL)
    # Runtime counts from the Running poll and is exactly the trace duration.
    assert m.gpu_hours_used == pytest.approx(_demand(SMALL), rel=1e-9)
    for p in obs.pods:
        assert p.scheduled_time is not None and p.start_time is not None
        assert p.scheduled_time <= p.start_time
    _valid_samples(obs)
    assert not cluster.pods  # every pod deleted: nothing leaks into the next run
    assert m.harness["queue_model"] == queue_model


def test_topology_is_read_back_from_node_labels() -> None:
    obs, _ = _run()
    assert obs.topology == derive(FLEET)


def test_setup_creates_namespace_and_non_preemptive_priority_classes() -> None:
    _, cluster = _run(jobs=SMALL)
    assert cluster.namespaces == {"k8slab"}
    (pc,) = cluster.priority_classes.values()
    assert pc["value"] == 100 and pc["preemptionPolicy"] == "Never"


def test_k0_is_bound_by_the_cluster_scheduler_not_by_the_runner() -> None:
    clock = FakeClock()
    cluster = FakeCluster(clock, kube_scheduler=True)
    obs, _ = _run("K0", cluster=cluster, clock=clock)
    assert compute(obs).jobs_completed == len(SMALL)
    manifest = pod_manifest(SMALL[0], 0, "K0", "k8slab")
    assert manifest["spec"]["schedulerName"] == "default-scheduler"


def test_degenerate_pods_name_a_scheduler_nobody_runs() -> None:
    m = pod_manifest(SMALL[0], 0, "D-fifo", "k8slab")
    assert m["spec"]["schedulerName"] == "k8slab-d-fifo"
    assert m["spec"]["terminationGracePeriodSeconds"] == 0  # unchanged from Phase 1
    assert m["metadata"]["name"] == "j1-p0"
    assert m["metadata"]["labels"]["k8slab.io/attempt"] == "0"
    again = pod_manifest(SMALL[0], 0, "D-fifo", "k8slab", attempt=2)
    assert again["metadata"]["name"] == "j1-p0-a2"


# ---- bind versus Running (E5) --------------------------------------------------------------


def test_startup_delay_is_held_idle_time_and_runtime_starts_at_running() -> None:
    clock = FakeClock()
    # 1.5 real seconds: longer than one poll, so Running is seen a poll later.
    cluster = FakeCluster(clock, startup=1.5)
    obs, _ = _run(cluster=cluster, clock=clock)
    by_id = {j.job_id: j for j in SMALL}
    for p in obs.pods:
        assert _v(p.start_time) - _v(p.scheduled_time) >= 300.0  # >= one poll (1 s x 300)
        assert _ran(p) == pytest.approx(by_id[p.job_id].duration)
    m = compute(obs)
    assert m.startup_overhead_gpu_hours > 0
    assert m.gpu_hours_used == pytest.approx(_demand(SMALL), rel=1e-9)
    _valid_samples(obs)


# ---- admission gate (E4) ------------------------------------------------------------------


def test_admission_gate_submits_one_pod_at_a_time() -> None:
    info = RunInfo(admission_gate=True)
    obs, cluster = _run(info=info)
    assert cluster.max_unbound == 1
    m = compute(obs)
    assert m.jobs_completed == len(SMALL) and m.gated


def test_without_the_gate_several_pods_wait_at_once() -> None:
    _, cluster = _run()
    assert cluster.max_unbound > 1


# ---- control-plane preemption (E6) --------------------------------------------------------


class _Hook(FakeCluster):
    """Destroys one running pod the first time it is listed as Running."""

    def __init__(self, clock: FakeClock, victim: str, mode: str) -> None:
        super().__init__(clock)
        self.victim, self.mode, self.fired_at = victim, mode, -1.0

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> Any:
        out = super().list_namespaced_pod(namespace, label_selector)
        pod = self.pods.get(self.victim)
        if self.fired_at < 0 and pod is not None and pod.status.phase == "Running" \
                and self.clock.now >= 2:
            self.fired_at = self.clock.now
            (self.vanish if self.mode == "vanish" else self.terminate)(self.victim)
        return out


@pytest.mark.parametrize("mode", ["vanish", "terminate"])
def test_control_plane_deletions_keep_gpus_held_for_the_grace_period(mode: str) -> None:
    clock = FakeClock()
    cluster = _Hook(clock, "j1-p0", mode)
    info = RunInfo(grace_seconds=900.0)
    obs, _ = _run(cluster=cluster, clock=clock, info=info)
    (e,) = obs.evictions
    assert not e.requeued and e.reason.startswith("control-plane")
    assert e.release_time - e.evict_time == 900.0
    ev = next(p for p in obs.pods if p.job_id == 1)
    assert ev.preempted  # a bare pod: nothing recreates it
    node = e.node
    held = [free[node] for t, free in obs.gpu_free_samples if e.evict_time <= t < e.release_time]
    assert held and all(f == 0 for f in held)  # j1 filled the 8-GPU node
    m = compute(obs)
    assert m.preemptions == 1 and m.grace_locked_gpu_hours == pytest.approx(8 * 900 / 3600)
    assert m.preempted_gpu_hours_lost > 0
    # The destroyed pod's Running time is lost, not delivered.
    rest = [j for j in SMALL if j.job_id != 1]
    assert m.gpu_hours_used == pytest.approx(_demand(rest), rel=1e-9)
    # ... and not delivered to its account either: "a" asked for 8 x 1800 (job
    # 1) + 4 x 2 x 900 (job 2) GPU-s and got only job 2's.
    assert m.service_ratio["a"] == pytest.approx(7200 / 21600)
    assert m.service_ratio["b"] == pytest.approx(1.0) == m.service_ratio["c"]
    assert m.fairness_ratio == pytest.approx(3.0)
    if mode == "terminate":
        assert "j1-p0" in cluster.deleted  # force-deleted when the grace ended
    _valid_samples(obs)


# ---- D-preempt through the API ----------------------------------------------------------


def test_d_preempt_evicts_through_the_api_and_requeues_under_a_new_name() -> None:
    jobs = [Job(i, "a", 0.0, 7200.0, gpus=8) for i in range(1, 13)]  # fill every dgx8
    jobs += [Job(20 + i, "a", 0.0, 7200.0, gpus=4) for i in range(8)]  # and every mid4
    jobs += [Job(40 + i, "a", 0.0, 7200.0, gpus=2) for i in range(6)]  # and every edge2
    jobs.append(Job(99, "b", 900.0, 600.0, gpus=8, priority=500))
    info = RunInfo(queue_model="kube", speedup=300.0, grace_seconds=600.0)
    obs, cluster = _run("D-preempt", jobs, info=info)
    (e,) = obs.evictions
    assert e.requeued and e.reason == "preempted by j99-p0"
    victim = f"j{e.job_id}-p0"
    assert victim in cluster.deleted and f"{victim}-a1" in cluster.created
    hi = next(p for p in obs.pods if p.job_id == 99)
    assert hi.node == e.node and _v(hi.scheduled_time) >= e.release_time
    m = compute(obs)
    assert m.jobs_completed == len(jobs) and m.preemptions == 1
    assert m.gpu_hours_used == pytest.approx(_demand(jobs), rel=1e-9)
    _valid_samples(obs)


# ---- topology extension (E7) -------------------------------------------------------------


def test_extend_mode_stretches_runtime_by_the_placement_factor() -> None:
    jobs = [Job(1, "a", 0.0, 3600.0, gpus=4, gang_size=2), Job(2, "a", 0.0, 1800.0, gpus=2)]
    info = RunInfo(topology_penalty="extend")
    obs, _ = _run(jobs=jobs, info=info)
    topo = derive(FLEET)
    for job in jobs:
        pods = sorted((p for p in obs.pods if p.job_id == job.job_id), key=lambda p: p.pod_index)
        f = placement_factor(JobPlacement(tuple(str(p.node) for p in pods), job.gpus), topo)
        for p in pods:
            assert _ran(p) == pytest.approx(job.duration * f)
    m = compute(obs)
    assert m.scenario
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-9)


# ---- refusals and plumbing -------------------------------------------------------------------


def test_refuses_a_cluster_without_the_fleet() -> None:
    clock = FakeClock()
    nodes = [n for n in render_nodes(FLEET) if n["metadata"]["name"] != "dgx8-3"]
    with pytest.raises(RuntimeError, match="dgx8-3"):
        _run(cluster=FakeCluster(clock, nodes=nodes), clock=clock)


def _extra_node(name: str) -> dict[str, Any]:
    """A kwok node outside FLEET: what `make up FLEET=a` then `make up
    FLEET=b` leaves behind (make up applies nodes and never deletes any)."""
    import copy

    node = copy.deepcopy(render_nodes(FLEET)[0])
    node["metadata"]["name"] = name
    node["metadata"]["labels"]["kubernetes.io/hostname"] = name
    return node


def test_refuses_a_cluster_with_kwok_nodes_outside_the_fleet() -> None:
    """Only MISSING nodes used to be refused. With a leftover node the K0
    replay ran to the end and scoring then raised on the unknown node's free
    count -- after the whole replay."""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=[*render_nodes(FLEET), _extra_node("aaa-0")],
                          kube_scheduler=True)
    with pytest.raises(RuntimeError, match=r"outside fleet.*aaa-0.*make down && make up"):
        _run("K0", cluster=cluster, clock=clock)
    assert cluster.created == []  # refused before anything was submitted


def test_a_bind_outside_the_fleet_stops_the_run_at_once() -> None:
    """A node that appears after the preflight: the first observation of a pod
    bound there stops the run, instead of scoring failing after it."""

    class LateNode(FakeCluster):
        def list_node(self, label_selector: str) -> Any:
            listed = super().list_node(label_selector)
            listed.items = [n for n in listed.items if n.metadata.name != "aaa-0"]
            return listed

    clock = FakeClock()
    cluster = LateNode(clock, nodes=[*render_nodes(FLEET), _extra_node("aaa-0")],
                       kube_scheduler=True)
    with pytest.raises(RuntimeError, match="'aaa-0', which is not in fleet"):
        _run("K0", cluster=cluster, clock=clock)


def test_refuses_nodes_without_topology_labels() -> None:
    clock = FakeClock()
    nodes = render_nodes(FLEET)
    for n in nodes:
        n["metadata"]["labels"] = {k: v for k, v in n["metadata"]["labels"].items()
                                   if not k.startswith("topology.")}
    with pytest.raises(RuntimeError, match="re-run `make up`"):
        _run(cluster=FakeCluster(clock, nodes=nodes), clock=clock)


def test_leftover_lab_pods_are_removed_before_the_run() -> None:
    clock = FakeClock()
    cluster = FakeCluster(clock)
    cluster.create_namespaced_pod("k8slab", pod_manifest(SMALL[0], 0, "K0", "k8slab"))
    obs, _ = _run(cluster=cluster, clock=clock)
    assert "j1-p0" in cluster.deleted[:1]
    assert compute(obs).jobs_completed == len(SMALL)


def test_retry_backs_off_on_transient_errors_only() -> None:
    calls: list[int] = []
    slept: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise FakeApiError(500)
        return "ok"

    assert _retry(flaky, sleep=slept.append) == "ok" and slept == [0.5, 1.0]

    def forbidden() -> None:
        raise FakeApiError(403)

    with pytest.raises(FakeApiError):
        _retry(forbidden, sleep=slept.append)


def test_the_poll_interval_is_one_real_second() -> None:
    clock = FakeClock()
    obs, _ = _run(clock=clock, speedup=120.0)
    times = [t for t, _ in obs.gpu_free_samples[:-1]]
    assert all(b - a == pytest.approx(TICK * 120.0) for a, b in zip(times, times[1:], strict=False))


class _HookK0(_Hook):
    """As _Hook, with the fake kube-scheduler binding default-scheduler pods."""

    def __init__(self, clock: FakeClock, victim: str) -> None:
        super().__init__(clock, victim, "vanish")
        self.kube_scheduler = True


def test_a_scheduler_reusing_grace_locked_gpus_is_clamped_and_recorded() -> None:
    """If the control plane deleted a pod outright, the real scheduler may bind
    into GPUs the runner still counts as grace-locked. Samples must stay valid
    (Definition C rejects negative free GPUs) and the conflict must be noted."""
    clock = FakeClock()
    jobs = [Job(1, "a", 0.0, 3600.0, gpus=8)] + [
        Job(10 + i, "a", 1000.0, 600.0, gpus=8) for i in range(14)  # more than fits
    ]
    cluster = _HookK0(clock, "j1-p0")
    obs, _ = _run("K0", jobs, cluster=cluster, clock=clock,
                  info=RunInfo(grace_seconds=3000.0))
    assert obs.evictions and not obs.evictions[0].requeued
    _valid_samples(obs)
    m = compute(obs)
    assert any("grace lock overlapped" in n for n in m.notes)


def test_an_unplaceable_pod_behind_the_admission_gate_is_a_stall() -> None:
    jobs = [Job(1, "a", 0.0, 60.0, gpus=16), Job(2, "a", 0.0, 60.0, gpus=1)]
    clock = FakeClock()
    obs, _ = _run(jobs=jobs, info=RunInfo(admission_gate=True), clock=clock)
    assert obs.horizon < 48 * 3600.0 / 10  # stopped on the stall, not the cap
    assert all(p.scheduled_time is None for p in obs.pods if p.job_id == 1)


def test_a_long_grace_lock_is_not_mistaken_for_a_stall() -> None:
    """One node. Its only running pod is deleted by the control plane with a
    grace period (3000 s) far longer than the stall window (15 polls x 60 s);
    the pending pod must wait for the lock, not be declared unschedulable."""
    from k8slab.model import Fleet, NodeClass

    tiny = Fleet("tiny", (NodeClass("n", 1, 8, nodes_per_rack=1),), racks_per_switch=1)
    clock = FakeClock()

    class OneNode(_Hook):
        def __init__(self) -> None:
            super().__init__(clock, "j1-p0", "vanish")
            self.nodes = render_nodes(tiny)

    cluster = OneNode()
    jobs = [Job(1, "a", 0.0, 7200.0, gpus=8), Job(2, "a", 0.0, 600.0, gpus=8)]
    cfg = RunConfig(config="D-fifo", speedup=60.0, quiet=True,
                    run=RunInfo(grace_seconds=3000.0))
    obs = Runner(tiny, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    (e,) = obs.evictions
    second = next(p for p in obs.pods if p.job_id == 2)
    assert second.scheduled_time is not None
    assert second.scheduled_time >= e.release_time


# ---- grace locks release on time --------------------------------------------------------

TINY = Fleet("tiny", (NodeClass("n", 1, 8, nodes_per_rack=1),), racks_per_switch=1)


@pytest.mark.parametrize("queue_model", ["none", "kube"])
def test_an_expired_grace_lock_is_free_on_the_first_pass_after_it(queue_model: str) -> None:
    """One 8-GPU node, D-preempt, grace 30 s, poll 60 s. The lock exists only
    in the runner's accounting, so the first sample and bind pass at or after
    release_time must see the GPUs free. Releasing after the bind pass held
    them a whole extra poll: bind at 420 instead of 360."""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 7200.0, gpus=8), Job(2, "b", 300.0, 600.0, gpus=8, priority=100)]
    speedup = 60.0
    info = RunInfo(queue_model=queue_model, speedup=speedup, grace_seconds=30.0)
    cfg = RunConfig(config="D-preempt", speedup=speedup, quiet=True, run=info)
    obs = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    first = next(e for e in obs.evictions if e.job_id == 1)
    assert (first.evict_time, first.release_time) == (300.0, 330.0)
    first_pass_after = math.ceil(first.release_time / speedup) * speedup  # 360
    assert cluster.bind_times["j2-p0"] * speedup == first_pass_after
    sample = dict(obs.gpu_free_samples)[first_pass_after]
    assert sample == {"n-0": 8}  # neither the lock nor anything else holds it
    m = compute(obs)
    assert m.grace_locked_gpu_hours == pytest.approx(8 * 30 / 3600)


# ---- a destroyed attempt delivers nothing, so it is never "extension" ---------------------


def test_a_destroyed_stretched_pod_adds_no_topology_extension() -> None:
    """extend mode, a 2 x 8 gang, one member destroyed by the control plane
    while Running. Its Running time is all lost; only the surviving member's
    stretch is delivered time that exists because of the factor."""
    clock = FakeClock()
    cluster = _Hook(clock, "j1-p0", "terminate")
    job = Job(1, "a", 0.0, 3600.0, gpus=8, gang_size=2)
    obs, _ = _run(jobs=[job], cluster=cluster, clock=clock,
                  info=RunInfo(topology_penalty="extend"))
    (e,) = obs.evictions
    assert not e.requeued and e.lost_seconds > 0
    nodes = tuple(str(p.node) for p in sorted(obs.pods, key=lambda p: p.pod_index))
    f = placement_factor(JobPlacement(nodes, 8), derive(FLEET))
    assert f > 1.0  # the premise: the gang was stretched
    m = compute(obs)
    survivor = 8 * 3600.0 / 3600.0
    assert m.topology_extension_gpu_hours == pytest.approx(survivor * (f - 1.0))
    # delivered = the delivered pods' work + extension, nothing double-counted
    assert m.gpu_hours_used == pytest.approx(survivor + m.topology_extension_gpu_hours)


# ---- declared versus default topology ------------------------------------------------------


def test_a_fleet_without_declared_topology_is_reported_as_undeclared() -> None:
    """The node labels exist either way (render_nodes writes the derived
    defaults), so their presence must not turn into topology_declared=True."""
    flat = Fleet("flat", (NodeClass("big", 2, 8), NodeClass("small", 2, 2)))
    assert not flat.topology_declared
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(flat))
    jobs = [Job(1, "a", 0.0, 600.0, gpus=4, gang_size=2), Job(2, "a", 60.0, 300.0, gpus=2)]
    cfg = RunConfig(config="D-fifo", speedup=300.0, quiet=True)
    obs = Runner(flat, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    assert obs.topology == derive(flat)
    assert compute(obs).topology_declared is False
    declared, _ = _run()
    assert compute(declared).topology_declared is True


# ---- queue model none is Phase 1's binder --------------------------------------------------


def test_none_mode_reseeds_the_binder_every_pass_as_phase_1_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 1 called the policy with a fresh Random(0) on every pass; the
    runner keeps that, seeded with the harness seed, so a one-repeat none-mode
    run is Phase 1's binder. Every call must see the RNG in its initial state,
    and the binds sent to the API must be exactly what those calls returned."""
    real = baselines.POLICIES["D-random"]
    calls: list[tuple[object, list[tuple[baselines.PendingPod, str]]]] = []

    def spy(
        pods: list[baselines.PendingPod], free: dict[str, int], rng: random.Random
    ) -> list[tuple[baselines.PendingPod, str]]:
        state = rng.getstate()
        out = real(pods, free, rng)
        calls.append((state, out))
        return out

    monkeypatch.setitem(baselines.POLICIES, "D-random", spy)
    seed = 7
    obs, cluster = _run("D-random", info=RunInfo(queue_model="none", harness_seed=seed))
    assert compute(obs).jobs_completed == len(SMALL)
    assert len(calls) >= 3  # several passes, or the reset would not matter
    fresh = random.Random(seed).getstate()
    assert all(state == fresh for state, _ in calls)
    sent = [(pod_name(p.job_id, p.pod_index), node) for _, out in calls for p, node in out]
    assert cluster.bindings == sent


# ---- work that has already ended is never evicted or stretched ------------------------------


@pytest.mark.parametrize("queue_model", ["none", "kube"])
def test_d_preempt_never_evicts_an_attempt_whose_work_ended_since_the_last_poll(
    queue_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One 8-GPU node, poll 60 s. Job 1 runs 60-150; job 2 (priority 100)
    arrives at 170 and is first tried at the 180 poll, where job 1 is finished
    but not yet retired (the bind pass runs before _retire). It used to be
    offered as a victim: evicted at 180 with 120 s 'lost' on a 90 s job, then
    run a second time. It must complete, and job 2 bind once it is retired.

    Two guards stop it: the victim view leaves finished attempts out, and
    _evict refuses one. The view is pinned here by recording what the binder
    is offered: with only _evict's guard, the plan would still pick job 1,
    nominate job 2 onto its node and report a preemption that never happened.
    (Removing the view filter alone left every test passing.)"""
    offered: list[tuple[float, list[tuple[int, int]]]] = []
    real = Binder.schedule

    def spy(self: Binder, now: float, pending: Any, free: Any, running: Any) -> Any:
        offered.append((now, [r.key for r in running]))
        return real(self, now, pending, free, running)

    monkeypatch.setattr(Binder, "schedule", spy)
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 90.0, gpus=8), Job(2, "b", 170.0, 600.0, gpus=8, priority=100)]
    info = RunInfo(queue_model=queue_model, speedup=60.0)
    cfg = RunConfig(config="D-preempt", speedup=60.0, quiet=True, run=info)
    obs = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    assert (180.0, []) in offered  # job 2's first pass: job 1 finished, not a victim
    assert all((1, 0) not in keys for now, keys in offered if now >= 150.0)
    assert obs.evictions == []
    first = next(p for p in obs.pods if p.job_id == 1)
    assert (first.start_time, first.end_time) == (60.0, 150.0)
    assert "j1-p0-a1" not in cluster.created  # never requeued
    second = next(p for p in obs.pods if p.job_id == 2)
    assert second.scheduled_time == 300.0  # bound at the 240 poll, observed at 300
    m = compute(obs)
    assert m.preemptions == 0 and m.gpu_hours_used == pytest.approx(_demand(jobs), rel=1e-12)


def test_evict_refuses_an_attempt_whose_work_has_ended() -> None:
    """_evict's own guard, independent of the victim view: an attempt whose
    work ended by now is left for _retire -- no delete, no lock, no Eviction
    record, no requeue. (The view never offers one today, so no replay reaches
    this guard; it is exercised directly.)"""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 90.0, gpus=8)]
    cfg = RunConfig(config="D-preempt", speedup=60.0, quiet=True, run=RunInfo(speedup=60.0))
    runner = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep)
    track = runner._tracks[(1, 0)]
    track.submitted = True
    attempt = Attempt(key=(1, 0), node="n-0", gpus=8, priority=0, bind_time=0.0, work=90.0)
    attempt.start(60.0)  # work ends at 150
    track.bound = attempt
    runner._evict(cluster, track, 180.0, "test")
    assert track.bound is attempt and track.attempt == 0 and track.submitted
    assert runner._evictions == [] and runner._locks == []
    assert "j1-p0" not in cluster.deleted
    # Still working at 120: evicted as before.
    runner._evict(cluster, track, 120.0, "test")
    (e,) = runner._evictions
    assert e.lost_seconds == 60.0 and track.bound is None and track.attempt == 1


class _TerminateAt(FakeCluster):
    """Marks one pod Terminating (control plane) at the first listing at or
    after real time ``at``."""

    def __init__(self, clock: FakeClock, victim: str, at: float, nodes: list[dict[str, Any]]):
        super().__init__(clock, nodes=nodes)
        self.victim, self.at = victim, at

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> Any:
        pod = self.pods.get(self.victim)
        if pod is not None and self.clock.now >= self.at and not pod.metadata.deletion_timestamp:
            self.terminate(self.victim)
        return super().list_namespaced_pod(namespace, label_selector)


@pytest.mark.parametrize(("at", "destroyed"), [(2.0, True), (3.0, False)])
def test_a_pod_seen_terminating_after_its_work_ended_completed(at: float, destroyed: bool) -> None:
    """Job 1 runs 60-150 (poll 60 s). Terminated by the control plane while
    still working (seen at 120) it is destroyed, as before. Seen Terminating
    only at 180, after its work ended, it completed: it used to be recorded
    destroyed with 120 s of a 90 s job 'lost'. A finished pod that VANISHED
    was already retired as completed on its first miss; now both agree."""
    clock = FakeClock()
    cluster = _TerminateAt(clock, "j1-p0", at, render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 90.0, gpus=8)]
    cfg = RunConfig(config="D-fifo", speedup=60.0, quiet=True, run=RunInfo(speedup=60.0))
    obs = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    (p,) = obs.pods
    m = compute(obs)
    if destroyed:
        (e,) = obs.evictions
        assert p.preempted and e.lost_seconds == 60.0 and m.gpu_hours_used == 0.0
    else:
        assert obs.evictions == [] and not p.preempted
        assert (p.start_time, p.end_time) == (60.0, 150.0)
        assert "j1-p0" in cluster.deleted  # force-deleted by _retire
        assert m.gpu_hours_used == pytest.approx(_demand(jobs), rel=1e-12)


TWO_GPU = Fleet("two", (NodeClass("n", 1, 2),))


def test_extend_never_moves_the_end_of_a_member_whose_work_already_ended() -> None:
    """One 2-GPU node (no NVLink). Gang member 0 runs 60-260; member 1 binds
    once job 1 is retired and is observed at 300 -- when member 0 has finished
    but is not yet retired. Applying the factor to member 0 used to move its
    end to 300: 40 s of phantom Running booked as topology extension."""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TWO_GPU))
    jobs = [Job(1, "a", 0.0, 100.0, gpus=1), Job(2, "a", 0.0, 200.0, gpus=1, gang_size=2)]
    info = RunInfo(speedup=60.0, topology_penalty="extend")
    cfg = RunConfig(config="D-fifo", speedup=60.0, quiet=True, run=info)
    obs = Runner(TWO_GPU, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    first, last = sorted((p for p in obs.pods if p.job_id == 2), key=lambda p: p.pod_index)
    assert (first.start_time, first.end_time) == (60.0, 260.0)
    assert last.start_time == 300.0
    f = placement_factor(JobPlacement(("n-0", "n-0"), 1), derive(TWO_GPU))
    assert f > 1.0
    assert _ran(last) == pytest.approx(200.0 * f)
    m = compute(obs)
    assert m.topology_extension_gpu_hours == pytest.approx(200.0 * (f - 1.0) / 3600.0)
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-12)


def test_extend_stretches_a_gang_whose_member_was_retired_before_the_last_bound() -> None:
    """Two 1-GPU nodes in one rack, D-preempt (priority order), poll 60 s.
    Gang member 0 runs 60-120 on n-1 and is retired; job 3 (priority 100)
    takes n-1; member 1 binds n-0 only after job 1 ends at 660. The job spans
    two nodes, so member 1 is stretched. Skipping any job with an ended member
    left such never-co-running gangs -- the worst co-scheduling -- unpenalised."""
    fleet = Fleet("pair", (NodeClass("n", 2, 1, nodes_per_rack=2),), racks_per_switch=1)
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(fleet))
    jobs = [
        Job(1, "a", 0.0, 600.0, gpus=1),
        Job(2, "a", 0.0, 60.0, gpus=1, gang_size=2),
        Job(3, "b", 150.0, 600.0, gpus=1, priority=100),
    ]
    info = RunInfo(speedup=60.0, topology_penalty="extend")
    cfg = RunConfig(config="D-preempt", speedup=60.0, quiet=True, run=info)
    obs = Runner(fleet, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    assert obs.evictions == []
    first, last = sorted((p for p in obs.pods if p.job_id == 2), key=lambda p: p.pod_index)
    assert (first.node, first.end_time, last.node) == ("n-1", 120.0, "n-0")
    assert _v(last.start_time) > _v(first.end_time)  # never co-ran
    f = placement_factor(JobPlacement(("n-1", "n-0"), 1), derive(fleet))
    assert f > 1.0
    assert _ran(first) == pytest.approx(60.0) and _ran(last) == pytest.approx(60.0 * f)
    m = compute(obs)
    assert m.topology_extension_gpu_hours == pytest.approx(60.0 * (f - 1.0) / 3600.0)


def test_a_retired_pod_frees_capacity_for_a_parked_pod_within_a_poll() -> None:
    """The runner's only signal that a completed pod freed GPUs is the event
    after _retire (_observe and _destroyed never report freed capacity). Job 2
    is parked at the 0 poll; job 1 is retired at 360, so job 2 binds at the
    next poll, 420. Without the event job 2 stays parked with nothing running,
    the stall detector (15 idle polls) ends the replay before the
    unschedulable-pool flush (1800 simulated s) could release it, and job 2 is
    recorded as never scheduled."""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 300.0, gpus=8), Job(2, "a", 0.0, 300.0, gpus=8)]
    info = RunInfo(queue_model="kube", speedup=60.0)
    cfg = RunConfig(config="D-fifo", speedup=60.0, quiet=True, run=info)
    obs = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    first = next(p for p in obs.pods if p.job_id == 1)
    assert first.end_time == 360.0
    assert cluster.bind_times["j2-p0"] * 60.0 == 420.0


# ---- a checkpoint on a cluster D-preempt run --------------------------------------------


def _checkpoint_case(extend: bool) -> tuple[Observation, list[Job]]:
    """TINY (one 8-GPU node, no NVLink), D-preempt, poll 60 s, grace 30 s,
    checkpoint 0.5. Job 1 is bound at the 0 poll, observed Running at 60 and
    evicted at 300 by job 2 (priority 100): 240 s of Running, half kept."""
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(TINY))
    jobs = [Job(1, "a", 0.0, 7200.0, gpus=8), Job(2, "b", 300.0, 600.0, gpus=8, priority=100)]
    info = RunInfo(queue_model="kube", speedup=60.0, grace_seconds=30.0,
                   checkpoint_fraction=0.5,
                   topology_penalty="extend" if extend else "report")
    cfg = RunConfig(config="D-preempt", speedup=60.0, quiet=True, run=info)
    obs = Runner(TINY, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    return obs, jobs


def test_a_checkpoint_keeps_half_the_evicted_progress_on_a_cluster_run() -> None:
    """The runner's _evict with --checkpoint-fraction, which no test set: the
    requeued attempt must run only the work the checkpoint did not keep, and
    the evicted attempt's Running time must split into lost and retained.
    Rerunning the whole job, or booking all of it as lost, both over- or
    under-deliver against the trace -- and passed the whole suite."""
    obs, jobs = _checkpoint_case(extend=False)
    e = next(e for e in obs.evictions if e.job_id == 1)
    assert (e.start_time, e.evict_time) == (60.0, 300.0)
    assert e.retained_seconds == pytest.approx(120.0) == e.lost_seconds
    final = next(p for p in obs.pods if p.job_id == 1)
    assert _ran(final) == pytest.approx(7200.0 - 0.5 * 240.0)
    m = compute(obs)
    assert m.jobs_completed == len(jobs)
    assert m.preempted_gpu_hours_lost == pytest.approx(8 * 120.0 / 3600.0)
    # Delivered = the retained part of the evicted attempt + the final attempt.
    assert m.gpu_hours_used == pytest.approx(_demand(jobs), rel=1e-12)
    assert m.service_ratio == pytest.approx({"a": 1.0, "b": 1.0}, rel=1e-12)


def test_a_checkpointed_stretched_attempt_keeps_only_trace_work_as_work() -> None:
    """The same with --topology-penalty extend: job 1 (8 GPUs on a node
    without NVLink) runs at factor 1.2, so its 240 s of Running did 200 s of
    work. The checkpoint keeps 120 s of wall time but only 100 s of work; the
    other 20 s x 8 GPUs are extension. Without that correction delivered
    GPU-time fell short of demand + extension by exactly those 160 GPU-s."""
    obs, jobs = _checkpoint_case(extend=True)
    e = next(e for e in obs.evictions if e.job_id == 1)
    assert e.retained_seconds == pytest.approx(120.0) == e.lost_seconds
    f = placement_factor(JobPlacement(("n-0",), 8), derive(TINY))
    assert f == pytest.approx(1.2)
    final = next(p for p in obs.pods if p.job_id == 1)
    assert _ran(final) == pytest.approx((7200.0 - 0.5 * 240.0 / f) * f)
    m = compute(obs)
    assert m.jobs_completed == len(jobs)
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-12)
    # Fairness counts the trace's work, not the stretch (docs/metrics.md).
    assert set(obs.extension_by_account) == {"a", "b"}
    assert m.service_ratio == pytest.approx({"a": 1.0, "b": 1.0}, rel=1e-12)


def test_extend_stretches_nothing_until_the_last_gang_member_is_bound() -> None:
    """Two 4-GPU nodes without NVLink, one per rack; poll 60 s. Job 1 (1 GPU)
    holds n-0 until 780; gang member 2-p0 is observed on n-1 at 60 and 2-p1
    on n-0 only at 900. Member 0 must run at factor 1 until 900 and then have
    its REMAINING work stretched by the whole placement's factor. Stretching
    at its own bind used the factor of a one-node partial placement (1.2),
    and no test noticed."""
    fleet = Fleet("pair4", (NodeClass("n", 2, 4, nodes_per_rack=1),))
    clock = FakeClock()
    cluster = FakeCluster(clock, nodes=render_nodes(fleet))
    jobs = [Job(1, "a", 0.0, 720.0, gpus=1), Job(2, "b", 0.0, 3600.0, gpus=4, gang_size=2)]
    info = RunInfo(speedup=60.0, topology_penalty="extend")
    cfg = RunConfig(config="D-fifo", speedup=60.0, quiet=True, run=info)
    obs = Runner(fleet, jobs, cfg, api=cluster, clock=clock, sleep=clock.sleep).run()
    first, last = sorted((p for p in obs.pods if p.job_id == 2), key=lambda p: p.pod_index)
    assert (first.node, first.start_time) == ("n-1", 60.0)
    assert (last.node, last.start_time) == ("n-0", 900.0)
    topo = derive(fleet)
    f = placement_factor(JobPlacement(("n-1", "n-0"), 4), topo)
    partial = placement_factor(JobPlacement(("n-1",), 4), topo)
    assert f > partial > 1.0
    # 840 s of work done at factor 1 by 900; the other 2760 at the job's factor.
    assert _v(first.end_time) == pytest.approx(900.0 + (3600.0 - 840.0) * f)
    assert _v(last.end_time) == pytest.approx(900.0 + 3600.0 * f)
    m = compute(obs)
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-12)
    assert set(obs.extension_by_account) == {"b"}
    assert m.fairness_ratio == pytest.approx(1.0, rel=1e-12)
