"""The reference model's Phase 2 execution layer (queue model, startup delay,
topology extension, preemption with grace locks, admission gate)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from k8slab.fleet import load
from k8slab.fragmentation import structural_fragmentation
from k8slab.metrics import compute
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent, RunInfo
from k8slab.sim import TICK, run
from k8slab.topology import JobPlacement, derive, placement_factor
from k8slab.trace import WorkloadProfile, generate

DEFAULT = load("fleets/default.yaml")
JOBS = generate(WorkloadProfile(job_count=150, arrival_interval=12.0), seed=4)
KUBE = RunInfo(queue_model="kube", speedup=60.0)


def _v(x: float | None) -> float:
    assert x is not None
    return x


def _ran(p: PodEvent) -> float:
    """Running time of a pod's final attempt."""
    return _v(p.end_time) - _v(p.start_time)


def _demand(jobs: list[Job]) -> float:
    return math.fsum(j.total_gpus * j.duration for j in jobs) / 3600.0


def _samples_are_valid(obs: Observation) -> None:
    """What Definition C requires of every execution path."""
    times = [t for t, _ in obs.gpu_free_samples]
    assert times == sorted(times)
    caps = {n: obs.fleet.gpus_of(n) for n in obs.fleet.node_names()}
    for _, free in obs.gpu_free_samples:
        assert set(free) == set(caps)
        assert all(0 <= free[n] <= caps[n] for n in caps)
    topo = derive(obs.fleet)
    structural_fragmentation(topo.capacity(), topo.domains(), obs.gpu_free_samples)


# ---- the kube queue model ----------------------------------------------------------


@pytest.mark.parametrize("policy", ["D-fifo", "D-random", "D-largest", "D-preempt"])
def test_kube_queue_completes_every_job_and_delivers_the_demand(policy: str) -> None:
    obs = run(DEFAULT, JOBS, policy, run_info=KUBE)
    m = compute(obs)
    assert m.jobs_completed == len(JOBS)
    assert m.gpu_hours_used == pytest.approx(_demand(JOBS), rel=1e-9)
    assert m.harness["queue_model"] == "kube" and m.harness["speedup"] == 60.0
    _samples_are_valid(obs)


def test_kube_queue_changes_decisions_relative_to_phase_1() -> None:
    """Backoff is a real mechanism: at least one pod binds at a different time."""
    a = run(DEFAULT, JOBS, "D-fifo")
    b = run(DEFAULT, JOBS, "D-fifo", run_info=KUBE)
    assert [p.scheduled_time for p in a.pods] != [p.scheduled_time for p in b.pods]


def test_the_speedup_sets_the_simulated_backoff() -> None:
    """Uncompressed backoff (speedup 1: 1-10 s) versus the quantised 60 s
    floor at speedup 60 must produce different schedules."""
    slow = run(DEFAULT, JOBS, "D-fifo", run_info=RunInfo(queue_model="kube", speedup=1.0))
    fast = run(DEFAULT, JOBS, "D-fifo", run_info=KUBE)
    assert [p.scheduled_time for p in slow.pods] != [p.scheduled_time for p in fast.pods]


def test_deterministic_policies_ignore_the_harness_seed_and_random_does_not() -> None:
    for policy in ("D-fifo", "D-largest"):
        a = compute(run(DEFAULT, JOBS, policy, seed=0, run_info=KUBE))
        b = compute(run(DEFAULT, JOBS, policy, seed=123, run_info=KUBE))
        assert a.scalars() == b.scalars()
        assert (a.harness_seed, b.harness_seed) == (0, 123)
    ra = compute(run(DEFAULT, JOBS, "D-random", seed=0, run_info=KUBE))
    rb = compute(run(DEFAULT, JOBS, "D-random", seed=123, run_info=KUBE))
    assert ra.scalars() != rb.scalars()


# ---- startup delay ------------------------------------------------------------------


def test_startup_delay_separates_bind_from_running_and_counts_the_gap() -> None:
    info = RunInfo(queue_model="kube", speedup=60.0, startup_delay_ms=(50.0, 200.0))
    obs = run(DEFAULT, JOBS, "D-fifo", run_info=info)
    by_id = {j.job_id: j for j in JOBS}
    gaps = []
    for p in obs.pods:
        assert p.scheduled_time is not None and p.start_time is not None
        gap = p.start_time - p.scheduled_time
        assert 3.0 <= gap < 12.0  # 50-200 ms real x 60
        gaps.append(by_id[p.job_id].gpus * gap)
        # Trace runtime counts from Running, not from bind.
        assert p.end_time == pytest.approx(p.start_time + by_id[p.job_id].duration)
    m = compute(obs)
    assert m.gpu_hours_used == pytest.approx(_demand(JOBS), rel=1e-9)
    assert m.startup_overhead_gpu_hours == pytest.approx(math.fsum(gaps) / 3600.0)
    _samples_are_valid(obs)


def test_startup_delay_is_seeded() -> None:
    info = RunInfo(speedup=60.0, startup_delay_ms=(50.0, 200.0))
    a = run(DEFAULT, JOBS, "D-random", seed=7, run_info=info)
    b = run(DEFAULT, JOBS, "D-random", seed=7, run_info=info)
    assert [p.start_time for p in a.pods] == [p.start_time for p in b.pods]
    c = run(DEFAULT, JOBS, "D-random", seed=8, run_info=info)
    assert [p.start_time for p in a.pods] != [p.start_time for p in c.pods]


@pytest.mark.parametrize("queue_model", ["none", "kube"])
def test_switching_the_startup_delay_on_never_perturbs_d_randoms_placements(
    queue_model: str,
) -> None:
    """The delay draws from its own stream (sim.run's ``startup_rng``), so a
    STARTUP_DELAY experiment changes when pods start, never where D-random
    puts them. Sixty 1-GPU jobs, one per pass, none finishing inside the
    window: no bind there depends on the delay, so the binds must be identical
    with it off and on. Drawing the delay from the policy's RNG changes them.
    (This file's previous test compared seeds 7 and 8 only, and passed with
    the two streams merged.)"""
    jobs = [Job(i, "a", 5.0 * i, 1e6, gpus=1) for i in range(60)]

    def binds(delay: tuple[float, float]) -> tuple[list[tuple[float, int, str]], list[float]]:
        info = RunInfo(queue_model=queue_model, speedup=60.0, startup_delay_ms=delay)
        obs = run(DEFAULT, jobs, "D-random", seed=7, run_info=info, max_horizon=400.0)
        placed = sorted((_v(p.scheduled_time), p.job_id, str(p.node))
                        for p in obs.pods if p.scheduled_time is not None)
        return placed, [_v(p.start_time) - _v(p.scheduled_time)
                        for p in obs.pods if p.scheduled_time is not None]

    off, off_gaps = binds((0.0, 0.0))
    on, on_gaps = binds((50.0, 200.0))
    assert len(off) == len(jobs)  # every job bound inside the window, none finished
    assert set(off_gaps) == {0.0} and all(g > 0 for g in on_gaps)  # the delay is on
    assert len({node for _, _, node in off}) > 10  # D-random really chose among nodes
    assert on == off


def test_a_gangs_startup_gap_is_stranded_time() -> None:
    fleet = Fleet("t", (NodeClass("n", 2, 8),))
    job = Job(1, "a", 0.0, 100.0, gpus=4, gang_size=2)
    info = RunInfo(speedup=60.0, startup_delay_ms=(100.0, 100.0))  # 6 s simulated
    m = compute(run(fleet, [job], "D-fifo", run_info=info))
    assert m.gang_assembled == 1
    assert m.gang_stranded_gpu_hours == pytest.approx(2 * 4 * 6.0 / 3600.0)
    assert m.startup_overhead_gpu_hours == pytest.approx(2 * 4 * 6.0 / 3600.0)


# ---- topology extension --------------------------------------------------------------


def test_report_mode_never_stretches() -> None:
    obs = run(DEFAULT, JOBS, "D-fifo", run_info=KUBE)
    assert obs.extension_gpu_seconds == 0.0
    assert compute(obs).topology_extension_gpu_hours == 0.0


@pytest.mark.parametrize("policy", ["D-fifo", "D-random"])
def test_extend_delivers_demand_times_realised_factors(policy: str) -> None:
    """The Phase 1 invariant becomes: delivered = demanded + the GPU-time the
    stretch added, when every job completes and nothing is checkpointed."""
    info = RunInfo(queue_model="kube", speedup=60.0, topology_penalty="extend")
    obs = run(DEFAULT, JOBS, policy, run_info=info)
    m = compute(obs)
    assert m.jobs_completed == len(JOBS)
    assert m.topology_extension_gpu_hours > 0
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-9
    )
    assert m.scenario and m.harness["topology_penalty"] == "extend"
    # The extension is booked against each stretched job's account ...
    assert len(obs.extension_by_account) > 1
    assert math.fsum(obs.extension_by_account.values()) == pytest.approx(
        obs.extension_gpu_seconds, rel=1e-12)
    # ... so fairness counts the trace's work: every job completed, so every
    # account got exactly what it asked for. Counting the stretch read 1.08-1.13
    # on the default trace (ratios up to 1.43), measuring the premise.
    assert m.service_ratio == pytest.approx({a: 1.0 for a in m.service_ratio}, rel=1e-9)
    assert m.fairness_ratio == pytest.approx(1.0, rel=1e-9)
    _samples_are_valid(obs)


def test_extend_stretches_a_whole_job_by_its_placement_factor() -> None:
    """Jobs whose pods all bind in one pass run exactly duration x factor."""
    info = RunInfo(topology_penalty="extend")
    obs = run(DEFAULT, JOBS, "D-fifo", run_info=info)
    topo = derive(DEFAULT)
    by_job: dict[int, list[PodEvent]] = {}
    for p in obs.pods:
        by_job.setdefault(p.job_id, []).append(p)
    checked = 0
    for job in JOBS:
        pods = sorted(by_job[job.job_id], key=lambda p: p.pod_index)
        if len({p.scheduled_time for p in pods}) != 1:
            continue
        f = placement_factor(JobPlacement(tuple(str(p.node) for p in pods), job.gpus), topo)
        for p in pods:
            assert _ran(p) == pytest.approx(job.duration * f)
        checked += f != 1.0
    assert checked > 10


def test_off_mode_reports_no_penalty() -> None:
    obs = run(DEFAULT, JOBS, "D-fifo", run_info=RunInfo(topology_penalty="off"))
    m = compute(obs)
    assert m.placement_penalty_mean is None
    assert m.placement_multi_pod_jobs > 0  # tiers are measurements: still reported


# ---- D-preempt: evictions, grace locks, requeue --------------------------------------


def _preemption_case(
    grace: float = 30.0, checkpoint: float = 0.0
) -> tuple[Observation, list[Job]]:
    fleet = Fleet("t", (NodeClass("n", 1, 8),))
    jobs = [
        Job(1, "a", 0.0, 1000.0, gpus=8, priority=0),  # fills the node
        Job(2, "b", 100.0, 50.0, gpus=8, priority=500),  # must preempt it
    ]
    info = RunInfo(speedup=60.0, grace_seconds=grace, checkpoint_fraction=checkpoint)
    return run(fleet, jobs, "D-preempt", run_info=info), jobs


def test_preemption_locks_the_victims_gpus_for_the_grace_period() -> None:
    obs, jobs = _preemption_case()
    (e,) = obs.evictions
    assert (e.job_id, e.node, e.evict_time, e.release_time) == (1, "n-0", 100.0, 130.0)
    assert e.requeued and e.reason == "preempted by j2-p0"
    # Held, not free, throughout the grace period; free again after it.
    for t, free in obs.gpu_free_samples:
        if 100.0 <= t < 130.0:
            assert free["n-0"] == 0, t
    hi = next(p for p in obs.pods if p.job_id == 2)
    assert hi.scheduled_time == 130.0  # the nominated preemptor gets the node
    lo = next(p for p in obs.pods if p.job_id == 1)
    assert _v(lo.scheduled_time) >= _v(hi.end_time)
    # Restarted from zero: the final attempt runs the whole duration.
    assert _ran(lo) == pytest.approx(1000.0)
    m = compute(obs)
    assert m.preemptions == 1
    assert m.preempted_gpu_hours_lost == pytest.approx(8 * 100.0 / 3600.0)
    assert m.grace_locked_gpu_hours == pytest.approx(8 * 30.0 / 3600.0)
    assert m.gpu_hours_used == pytest.approx(_demand(jobs))  # useful work only
    # Wait is pending time: the victim was bound 0-100, so it waited only
    # from its eviction to its re-bind -- not from its submit at 0.
    assert m.mean_wait == pytest.approx(((_v(lo.scheduled_time) - 100.0) + 30.0) / 2)
    _samples_are_valid(obs)


def test_a_checkpoint_keeps_part_of_the_evicted_progress() -> None:
    obs, jobs = _preemption_case(checkpoint=0.25)
    (e,) = obs.evictions
    assert e.retained_seconds == pytest.approx(25.0) and e.lost_seconds == pytest.approx(75.0)
    lo = next(p for p in obs.pods if p.job_id == 1)
    assert _ran(lo) == pytest.approx(1000.0 - 25.0)
    m = compute(obs)
    assert m.gpu_hours_used == pytest.approx(_demand(jobs))


def test_zero_grace_releases_at_the_next_pass() -> None:
    obs, _ = _preemption_case(grace=0.0)
    hi = next(p for p in obs.pods if p.job_id == 2)
    assert hi.scheduled_time == 100.0 + TICK


def test_gang_members_are_evicted_together() -> None:
    fleet = Fleet("t", (NodeClass("n", 2, 4),))
    jobs = [
        Job(1, "a", 0.0, 1000.0, gpus=4, gang_size=2, priority=0),
        Job(2, "b", 50.0, 10.0, gpus=4, priority=100),
    ]
    obs = run(fleet, jobs, "D-preempt", run_info=RunInfo(speedup=60.0))
    assert sorted((e.job_id, e.pod_index) for e in obs.evictions[:2]) == [(1, 0), (1, 1)]
    m = compute(obs)
    assert m.jobs_completed == 2 and m.preemptions >= 2


def test_d_preempt_on_the_trace_preempts_and_still_completes() -> None:
    obs = run(DEFAULT, JOBS, "D-preempt", run_info=KUBE)
    m = compute(obs)
    assert m.preemptions > 0 and m.grace_locked_gpu_hours > 0
    assert m.jobs_completed == len(JOBS)
    assert m.gpu_hours_used == pytest.approx(_demand(JOBS), rel=1e-9)
    # Only strictly lower priority is ever evicted.
    prio = {j.job_id: j.priority for j in JOBS}
    for e in obs.evictions:
        preemptor = int(e.reason.split("j")[1].split("-")[0])
        assert prio[e.job_id] < prio[preemptor]


def test_non_preemptive_policies_never_evict() -> None:
    for policy in ("D-fifo", "D-random", "D-largest"):
        assert run(DEFAULT, JOBS, policy, run_info=KUBE).evictions == []


# ---- admission gate -------------------------------------------------------------------


def test_admission_gate_keeps_at_most_one_pod_pending() -> None:
    jobs = JOBS[:40]
    obs = run(DEFAULT, jobs, "D-fifo", run_info=RunInfo(admission_gate=True))
    m = compute(obs)
    assert m.jobs_completed == len(jobs) and m.gated
    binds = sorted(p.scheduled_time for p in obs.pods if p.scheduled_time is not None)
    # One admission per pass, so no two pods bind in the same pass.
    assert len(binds) == len(set(binds))
    ungated = compute(run(DEFAULT, jobs, "D-fifo"))
    assert m.mean_wait > ungated.mean_wait


# ---- the eviction API ------------------------------------------------------------------


def test_requested_eviction_locks_requeues_and_restarts_from_zero() -> None:
    from k8slab.sim import EvictionRequest

    fleet = Fleet("t", (NodeClass("n", 2, 8),))
    jobs = [Job(1, "a", 0.0, 1000.0, gpus=4), Job(2, "a", 0.0, 50.0, gpus=1)]
    obs = run(fleet, jobs, "D-fifo", run_info=RunInfo(grace_seconds=40.0),
              evictions=[EvictionRequest((1, 0), 102.0, "drain")])
    (e,) = obs.evictions
    # Applied at the first pass at or after the requested time.
    assert (e.evict_time, e.release_time, e.reason) == (105.0, 145.0, "drain")
    assert e.lost_seconds == pytest.approx(105.0) and e.requeued
    lo = next(p for p in obs.pods if p.job_id == 1)
    assert _v(lo.scheduled_time) >= 105.0 and _ran(lo) == pytest.approx(1000.0)
    locked = [free["n-0"] for t, free in obs.gpu_free_samples if 105.0 <= t < 145.0]
    assert locked and all(f <= 4 for f in locked)
    m = compute(obs)
    assert m.preemptions == 1 and m.gpu_hours_used == pytest.approx(_demand(jobs))


def test_requested_eviction_of_a_gang_member_evicts_the_whole_gang() -> None:
    from k8slab.sim import EvictionRequest

    fleet = Fleet("t", (NodeClass("n", 2, 8),))
    jobs = [Job(1, "a", 0.0, 500.0, gpus=4, gang_size=3)]
    obs = run(fleet, jobs, "D-fifo", evictions=[EvictionRequest((1, 2), 50.0)])
    assert sorted(e.pod_index for e in obs.evictions) == [0, 1, 2]
    assert compute(obs).jobs_completed == 1


def test_requests_that_cannot_apply_are_noted_not_silently_dropped() -> None:
    from k8slab.sim import EvictionRequest

    fleet = Fleet("t", (NodeClass("n", 1, 8),))
    jobs = [Job(1, "a", 100.0, 50.0, gpus=1)]
    obs = run(fleet, jobs, "D-fifo", evictions=[
        EvictionRequest((1, 0), 10.0),  # not submitted yet
        EvictionRequest((1, 0), 1e6),  # after the run ends: must not stretch it
    ])
    assert obs.evictions == [] and obs.horizon < 1000.0
    assert len(obs.notes) == 2 and "not bound" in obs.notes[0] and "ended" in obs.notes[1]


@pytest.mark.parametrize("checkpoint", [0.0, 0.5])
def test_a_requested_eviction_never_evicts_work_that_already_finished(checkpoint: float) -> None:
    """The job's work ends at 12, between the passes at 10 and 15; the request
    is due at 15. It used to be applied before that pass released the pod:
    15 s 'lost' on a 12 s job, which then ran again (15-27). With a checkpoint
    the retained share even included post-completion time, so delivered
    GPU-hours exceeded the demand."""
    from k8slab.sim import EvictionRequest

    fleet = Fleet("t", (NodeClass("n", 1, 8),))
    jobs = [Job(1, "a", 0.0, 12.0, gpus=8)]
    info = RunInfo(grace_seconds=30.0, checkpoint_fraction=checkpoint)
    obs = run(fleet, jobs, "D-fifo", run_info=info,
              evictions=[EvictionRequest((1, 0), 15.0)])
    assert obs.evictions == []
    (p,) = obs.pods
    assert (p.scheduled_time, p.start_time, p.end_time) == (0.0, 0.0, 12.0)
    assert len(obs.notes) == 1 and "its work finished at 12" in obs.notes[0]
    m = compute(obs)
    assert m.preemptions == 0 and m.preempted_gpu_hours_lost == 0.0
    assert m.gpu_hours_used == pytest.approx(_demand(jobs))


def _split_gang_case(**info: Any) -> tuple[Fleet, list[Job], RunInfo]:
    """A gang that never co-runs. Two 1-GPU nodes in one rack, kube queue at
    speedup 60 (60 s backoff). Job 1 holds n-0 until 300. Gang member 2-p0
    binds n-1 at 5 and finishes at 55; 2-p1 failed at 5 and is still backing
    off (until 65) when n-1 frees, so job 3, submitted at 55, takes n-1. 2-p1
    binds n-0 only when job 1 ends, at 300."""
    fleet = Fleet("t", (NodeClass("n", 2, 1, nodes_per_rack=2),), racks_per_switch=1)
    jobs = [
        Job(1, "a", 0.0, 300.0, gpus=1),
        Job(2, "a", 1.0, 50.0, gpus=1, gang_size=2),
        Job(3, "a", 55.0, 300.0, gpus=1),
    ]
    return fleet, jobs, RunInfo(queue_model="kube", speedup=60.0, **info)


def test_a_request_naming_a_finished_gang_member_leaves_its_siblings_running() -> None:
    """A request for member 0 after it finished names a pod that is no longer
    bound, so it is ignored -- it used to evict the still-running member 1."""
    from k8slab.sim import EvictionRequest

    fleet, jobs, info = _split_gang_case()
    obs = run(fleet, jobs, "D-fifo", run_info=info,
              evictions=[EvictionRequest((2, 0), 320.0)])
    assert obs.evictions == []
    assert len(obs.notes) == 1 and "its work finished at 55" in obs.notes[0]
    second = next(p for p in obs.pods if (p.job_id, p.pod_index) == (2, 1))
    assert (second.scheduled_time, second.end_time) == (300.0, 350.0)


def test_extend_stretches_a_gang_whose_first_member_finished_before_the_last_bound() -> None:
    """A D-random-style gang that never co-ran: member 0 (n-1) finished at 55,
    member 1 bound to n-0 at 300. The job's placement spans two nodes, so
    member 1's work is stretched by that factor. Requiring every member to be
    still bound exempted such gangs entirely -- on the default trace 89 of
    D-random's penalised gangs, the most of any policy -- so the worst
    co-scheduler paid the least scenario penalty. The finished member had no
    remaining work and is not stretched."""
    fleet, jobs, info = _split_gang_case(topology_penalty="extend")
    obs = run(fleet, jobs, "D-fifo", run_info=info)
    first, last = sorted((p for p in obs.pods if p.job_id == 2), key=lambda p: p.pod_index)
    assert (first.node, first.start_time, first.end_time) == ("n-1", 5.0, 55.0)
    assert (last.node, last.scheduled_time) == ("n-0", 300.0)
    f = placement_factor(JobPlacement(("n-1", "n-0"), 1), derive(fleet))
    assert f > 1.0  # the premise: two nodes, so the job is penalised
    assert _ran(first) == pytest.approx(50.0)
    assert _ran(last) == pytest.approx(50.0 * f)
    m = compute(obs)
    assert m.topology_extension_gpu_hours == pytest.approx(50.0 * (f - 1.0) / 3600.0)
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-12)


def test_extend_stretches_nothing_until_the_last_gang_member_is_bound() -> None:
    """Two 4-GPU nodes without NVLink, one per rack. Job 1 (1 GPU) holds n-0
    until 720, so gang member 2-p0 binds n-1 at 5 and 2-p1 fits only at 720.
    Nothing may be stretched while the placement is partial: member 0 runs at
    factor 1 until 720, then its REMAINING work is stretched by the whole
    placement's factor. Stretching from the first bind would use the factor
    of a one-node partial placement (1.2 here), a placement the job never had.
    The other extend tests use 1-GPU gangs (a partial placement's factor is
    1.0) or check only delivered = demanded + extension, which holds either
    way -- so relaxing the every-member-bound check passed them all."""
    fleet = Fleet("t", (NodeClass("n", 2, 4, nodes_per_rack=1),))
    jobs = [Job(1, "a", 0.0, 720.0, gpus=1), Job(2, "a", 1.0, 3600.0, gpus=4, gang_size=2)]
    obs = run(fleet, jobs, "D-fifo", run_info=RunInfo(topology_penalty="extend"))
    first, last = sorted((p for p in obs.pods if p.job_id == 2), key=lambda p: p.pod_index)
    assert (first.node, first.start_time) == ("n-1", 5.0)
    assert (last.node, last.start_time) == ("n-0", 720.0)
    topo = derive(fleet)
    f = placement_factor(JobPlacement(("n-1", "n-0"), 4), topo)
    partial = placement_factor(JobPlacement(("n-1",), 4), topo)
    assert f > partial > 1.0  # the premise: a partial factor exists and differs
    # 715 s of work done at factor 1 by 720; the other 2885 at the job's factor.
    assert _v(first.end_time) == pytest.approx(720.0 + (3600.0 - 715.0) * f)
    assert _v(last.end_time) == pytest.approx(720.0 + 3600.0 * f)
    m = compute(obs)
    assert m.topology_extension_gpu_hours == pytest.approx(
        4 * ((3600.0 - 715.0) + 3600.0) * (f - 1.0) / 3600.0)
    assert m.gpu_hours_used == pytest.approx(
        m.gpu_hours_demanded + m.topology_extension_gpu_hours, rel=1e-12)


def test_extend_leaves_no_multi_node_gang_unstretched_on_a_real_trace() -> None:
    """Every fully placed job with a factor above 1 has at least one member
    stretched -- the member whose bind completed the placement is always still
    bound -- whatever the policy's co-scheduling."""
    info = RunInfo(queue_model="kube", speedup=60.0, topology_penalty="extend")
    topo = derive(DEFAULT)
    for policy in ("D-random", "D-fifo"):
        obs = run(DEFAULT, JOBS, policy, run_info=info)
        by_job: dict[int, list[PodEvent]] = {}
        for p in obs.pods:
            by_job.setdefault(p.job_id, []).append(p)
        for job in JOBS:
            pods = sorted(by_job[job.job_id], key=lambda p: p.pod_index)
            f = placement_factor(JobPlacement(tuple(str(p.node) for p in pods), job.gpus), topo)
            if f > 1.0:
                assert any(_ran(p) > job.duration + 1e-6 for p in pods), (policy, job.job_id)


# ---- the capacity-freed event (kube queue) ----------------------------------------


def test_a_parked_pod_is_retried_on_the_first_pass_after_capacity_is_freed() -> None:
    """The requeue-on-freed-capacity mechanism E2 is built on. Job 2 does not
    fit at t=5 and is parked (backoff until 65). Job 1 ends at 100: that frees
    capacity, the parked pod's backoff is over, so it binds at 100. Without
    the event it waits for the unschedulable-pool flush -- t=1800 at speedup
    60 -- and nothing else in the suite noticed its removal."""
    fleet = Fleet("t", (NodeClass("n", 1, 8),))
    jobs = [Job(1, "a", 0.0, 100.0, gpus=8), Job(2, "a", 1.0, 100.0, gpus=8)]
    obs = run(fleet, jobs, "D-fifo", run_info=KUBE)
    second = next(p for p in obs.pods if p.job_id == 2)
    assert second.scheduled_time == 100.0


def test_a_grace_lock_release_is_a_capacity_freed_event() -> None:
    """Kube queue, D-preempt, grace 30 s. The preemptor fails at 100 (backoff
    until 160) and its victim's GPUs are locked until 130. The lock's release
    moves it to backoffQ; the next backoffQ flush after its backoff (180)
    binds it. Without the event it would sit in the pool until 1800."""
    fleet = Fleet("t", (NodeClass("n", 1, 8),))
    jobs = [Job(1, "a", 0.0, 1000.0, gpus=8), Job(2, "b", 100.0, 50.0, gpus=8, priority=500)]
    obs = run(fleet, jobs, "D-preempt",
              run_info=RunInfo(queue_model="kube", speedup=60.0, grace_seconds=30.0))
    (e,) = obs.evictions
    assert (e.evict_time, e.release_time) == (100.0, 130.0)
    hi = next(p for p in obs.pods if p.job_id == 2)
    assert hi.scheduled_time == 180.0
