from __future__ import annotations

import dataclasses
import math
import statistics
import sys

import pytest

from k8slab.fleet import load
from k8slab.metrics import (
    FOOTPRINT_BUCKETS,
    _min_pending_at,
    _stranded_at,
    average_ranks,
    compute,
    footprint_bucket,
    nearest_rank,
    percentile,
    spearman,
    trace_digest,
)
from k8slab.model import Eviction, Fleet, Job, NodeClass, Observation, PodEvent, RunInfo
from k8slab.sim import run
from k8slab.topology import NodeTopology, PenaltyFactors, Topology, derive
from k8slab.trace import WorkloadProfile, generate

FLEET = Fleet("t", (NodeClass("big", 1, 8), NodeClass("small", 2, 2)))


def test_legacy_percentile_is_phase_1s_rounded_rank_not_nearest_rank() -> None:
    """Kept for p95_wait only (Phase 1 bit-identity). Pinned so nobody "fixes"
    it silently, and so the docs cannot call it nearest-rank again."""
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0
    assert percentile([], 95) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 2.0  # nearest-rank: 3
    assert percentile([float(i) for i in range(1, 12)], 95) == 10.0  # nearest-rank: 11


def test_nearest_rank_is_the_smallest_value_covering_pct() -> None:
    assert nearest_rank([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert nearest_rank([float(i) for i in range(1, 12)], 95) == 11.0
    # 0.07 * 100 == 7.000000000000001 in floating point; the rank must be 7.
    assert nearest_rank([float(i) for i in range(1, 101)], 7) == 7.0
    for n in range(1, 60):
        values = [float(v) for v in range(n, 0, -1)]  # unsorted on purpose
        for pct in (0, 1, 5, 7, 25, 50, 90, 95, 99, 100):
            covered = [v for v in values if sum(w <= v for w in values) * 100 >= pct * n]
            assert nearest_rank(values, pct) == min(covered), (n, pct)
    with pytest.raises(ValueError):
        nearest_rank([], 50)
    with pytest.raises(ValueError):
        nearest_rank([1.0], 101)


def test_new_percentile_metrics_are_nearest_rank() -> None:
    """gang_assembly_p50/p95 and wait_by_footprint p95 are documented as
    nearest-rank; with five gangs and eleven 1-GPU jobs the legacy rounded
    rank would give a different answer for each."""
    jobs = [Job(i, "a", 0.0, 10.0, 1, gang_size=2) for i in range(1, 6)]
    pods = []
    for i, delay in zip(range(1, 6), (0.0, 10.0, 20.0, 30.0, 40.0), strict=True):
        pods += [PodEvent(i, 0, 0.0, 0.0, 10.0, "big-0"),
                 PodEvent(i, 1, delay, delay, delay + 10.0, "big-0")]
    m = compute(Observation("t", FLEET, jobs, pods, horizon=500.0))
    assert m.gang_assembly_p50 == 20.0  # legacy rounded rank: 10.0
    singles = compute(_waits_obs([(1, 1, float(w)) for w in range(1, 12)]))
    assert singles.wait_by_footprint["1"]["p95"] == 11.0  # legacy rounded rank: 10.0


def test_idle_cluster_is_not_fragmented() -> None:
    """Capacity nobody wants is idle, not stranded."""
    assert _stranded_at({"a": 4, "b": 2}, None) == 0


def test_stranded_counts_only_unusable_remainders() -> None:
    # smallest pending pod needs 4: the node with 2 free is stranded, 4 is not.
    assert _stranded_at({"a": 2, "b": 4, "c": 0}, 4) == 2


def test_gpu_hours_equal_trace_demand_when_all_jobs_complete() -> None:
    """The check that caught a +11.9% GPU-hour inflation from polling lag.

    In ``topology_penalty="extend"`` (a scenario, not the default) delivered
    time is demand plus the stretch; that form of the invariant is
    ``test_extend_delivers_demand_times_realised_factors`` in
    tests/test_sim_phase2.py.
    """
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=120, arrival_interval=30.0), seed=5)
    obs = run(fleet, jobs, "D-fifo")
    m = compute(obs)
    demanded = sum(j.total_gpus * j.duration for j in jobs) / 3600.0
    assert m.jobs_completed == len(jobs)
    assert abs(m.gpu_hours_used - demanded) < 0.01 * demanded


def test_homogeneous_fleet_barely_fragments_under_definition_b() -> None:
    """The control. Fragmentation is structurally near-impossible here."""
    fleet = load("fleets/homogeneous.yaml")
    jobs = generate(WorkloadProfile(job_count=150, arrival_interval=20.0), seed=1)
    het = compute(run(load("fleets/default.yaml"), jobs, "D-fifo"))
    hom = compute(run(fleet, jobs, "D-fifo"))
    assert hom.fragmentation_ref < het.fragmentation_ref


def test_definition_a_collapses_for_largest_first() -> None:
    """Guards the artefact the README is built around.

    Largest-first starves small jobs, so a 1-GPU pod is always pending, so
    definition A can never mark a GPU unusable. Definition B still can. If this
    test ever fails, one of the two definitions has silently changed meaning.
    """
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=250, arrival_interval=12.0), seed=0)
    m = compute(run(fleet, jobs, "D-largest"))
    # Not identically zero: at the very start and the very tail there are brief
    # windows with no 1-GPU pod queued. It rounds to 0.0% and is ~4 orders of
    # magnitude below definition B on the same run, which is the point.
    assert m.fragmentation_rate < 0.001
    assert m.fragmentation_ref > 0.2
    assert m.fragmentation_ref > 100 * m.fragmentation_rate


def test_gang_wait_is_measured_to_the_last_pod() -> None:
    job = Job(job_id=1, account="a", submit_time=0.0, duration=100.0, gpus=1, gang_size=2)
    obs = Observation(
        config="t", fleet=FLEET, jobs=[job],
        pods=[
            PodEvent(1, 0, scheduled_time=10.0, start_time=10.0, end_time=110.0, node="big-0"),
            PodEvent(1, 1, scheduled_time=90.0, start_time=90.0, end_time=190.0, node="big-0"),
        ],
        horizon=200.0,
    )
    assert compute(obs).mean_wait == 90.0


def test_partially_placed_gang_counts_as_deadlocked() -> None:
    job = Job(job_id=1, account="a", submit_time=0.0, duration=100.0, gpus=2, gang_size=2)
    obs = Observation(
        config="t", fleet=FLEET, jobs=[job],
        pods=[
            PodEvent(1, 0, scheduled_time=0.0, start_time=0.0, end_time=100.0, node="big-0"),
            PodEvent(1, 1),
        ],
        horizon=100.0,
    )
    m = compute(obs)
    assert m.gang_deadlocked == 1
    assert m.gang_deadlock_rate == 1.0
    assert m.gang_stranded_gpu_hours > 0


def test_fairness_is_one_when_every_account_is_served_equally() -> None:
    jobs = [
        Job(job_id=1, account="a", submit_time=0.0, duration=100.0, gpus=1),
        Job(job_id=2, account="b", submit_time=0.0, duration=100.0, gpus=1),
    ]
    obs = Observation(
        config="t", fleet=FLEET, jobs=jobs,
        pods=[
            PodEvent(1, 0, scheduled_time=0.0, start_time=0.0, end_time=100.0, node="big-0"),
            PodEvent(2, 0, scheduled_time=0.0, start_time=0.0, end_time=100.0, node="big-0"),
        ],
        horizon=100.0,
    )
    assert compute(obs).fairness_ratio == 1.0


def test_reference_model_is_flagged_not_measured() -> None:
    """A model row must never be mistaken for a cluster row."""
    m = compute(run(load("fleets/default.yaml"),
                    generate(WorkloadProfile(job_count=40), seed=0), "D-fifo"))
    assert m.measured_on_cluster is False


# ---- gang stranding (M2) ------------------------------------------------------

H = 3600.0


def _gang(pods: list[PodEvent], gpus: int = 1, horizon: float = 1000.0) -> Observation:
    job = Job(job_id=1, account="a", submit_time=0.0, duration=100.0, gpus=gpus,
              gang_size=len(pods))
    return Observation(config="t", fleet=FLEET, jobs=[job], pods=pods, horizon=horizon)


def test_stranded_gang_time_matches_phase_1_waste_for_a_stalled_gang() -> None:
    """The reconciliation. For a gang whose pods bind and start together and
    whose spread exceeds the old 60 s threshold, the replacement metric equals
    Phase 1's gang_wasted_gpu_hours exactly: sum of g * (max T - t)."""
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=10.0, start_time=10.0, end_time=110.0, node="big-0"),
        PodEvent(1, 1, scheduled_time=90.0, start_time=90.0, end_time=190.0, node="big-0"),
    ], gpus=2))
    assert m.gang_stranded_gpu_hours == 2 * (90.0 - 10.0) / H
    assert m.gang_assembled == 1
    assert m.gang_assembly_p50 == m.gang_assembly_p95 == 80.0
    assert m.gang_stranded_share == pytest.approx((2 * 80.0) / (2 * 200.0))


def test_short_assembly_is_not_rounded_to_zero_by_a_threshold() -> None:
    """Phase 1 charged nothing below 60 s. The continuous metric does."""
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=0.0, start_time=0.0, end_time=100.0, node="big-0"),
        PodEvent(1, 1, scheduled_time=30.0, start_time=30.0, end_time=130.0, node="big-0"),
    ]))
    assert m.gang_stalled == 0  # deprecated binary: below threshold
    assert m.gang_stranded_gpu_hours == 30.0 / H


def test_held_time_counts_from_bind_and_assembly_from_running() -> None:
    """Bound pods hold GPUs whether or not they run; the gang is assembled
    only when every member is Running (PodEvent.start_time)."""
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=0.0, start_time=30.0, end_time=130.0, node="big-0"),
        PodEvent(1, 1, scheduled_time=0.0, start_time=50.0, end_time=150.0, node="big-0"),
    ]))
    assert m.gang_stranded_gpu_hours == (50.0 + 50.0) / H
    assert m.gang_assembly_p50 == 50.0  # last Running (50) - first bind (0)


def test_member_that_finished_before_assembly_was_stranded_for_its_whole_life() -> None:
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=0.0, start_time=0.0, end_time=50.0, node="big-0"),
        PodEvent(1, 1, scheduled_time=80.0, start_time=80.0, end_time=180.0, node="big-0"),
    ]))
    assert m.gang_stranded_gpu_hours == 50.0 / H


def test_never_assembled_gang_is_stranded_until_each_member_ends() -> None:
    """Phase 1 charged g * |placed| * (H - first bind) -- here 1 * 1 * 1000 s,
    although the pod ended at 100 s. The replacement charges what was held."""
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=0.0, start_time=0.0, end_time=100.0, node="big-0"),
        PodEvent(1, 1),
    ], horizon=1000.0))
    assert m.gang_deadlocked == 1 and m.gang_assembled == 0
    assert m.gang_stranded_gpu_hours == 100.0 / H
    assert m.gang_assembly_p50 is None and m.gang_assembly_p95 is None


def test_unfinished_member_is_held_until_the_horizon() -> None:
    m = compute(_gang([
        PodEvent(1, 0, scheduled_time=100.0, start_time=100.0, node="big-0"),
        PodEvent(1, 1),
    ], horizon=400.0))
    assert m.gang_stranded_gpu_hours == 300.0 / H


def test_gang_assembly_percentiles_over_several_gangs() -> None:
    jobs = [Job(i, "a", 0.0, 10.0, 1, gang_size=2) for i in range(1, 5)]
    pods = []
    for i, delay in zip(range(1, 5), (0.0, 10.0, 20.0, 400.0), strict=True):
        pods += [PodEvent(i, 0, 0.0, 0.0, 10.0, "big-0"),
                 PodEvent(i, 1, delay, delay, delay + 10.0, "big-0")]
    m = compute(Observation("t", FLEET, jobs, pods, horizon=500.0))
    assert m.gang_assembly_p50 == 10.0 and m.gang_assembly_p95 == 400.0


def test_evicted_gang_attempts_are_stranded_too() -> None:
    """The final PodEvents describe the last attempt only. An evicted and
    requeued attempt held GPUs as well, by the same rule: an attempt that
    assembled is stranded before its assembly, one that never assembled for
    every bound member's whole held time, each member ending at the eviction."""
    job = Job(1, "a", 0.0, 100.0, gpus=2, gang_size=2)

    def ev(i: int, bind: float, start: float | None, evict: float) -> Eviction:
        return Eviction(1, i, "big-0", 2, bind, start, evict, evict + 30.0, "test")

    evictions = [
        # attempt 1 assembled at 60 (latest start), evicted at 100
        ev(0, 0.0, 10.0, 100.0), ev(1, 50.0, 60.0, 100.0),
        # attempt 2: only member 0 was bound when it was evicted at 300
        ev(0, 200.0, 200.0, 300.0),
    ]
    pods = [PodEvent(1, 0, 400.0, 400.0, 500.0, "big-0"),
            PodEvent(1, 1, 400.0, 400.0, 500.0, "big-0")]
    m = compute(Observation("t", FLEET, [job], pods, horizon=500.0, evictions=evictions))
    assert m.gang_stranded_gpu_hours == pytest.approx(2 * ((60 - 0) + (60 - 50) + 100) / H)
    # Assembly counts and percentiles describe each gang's final attempt.
    assert m.gang_assembled == 1 and m.gang_assembly_p50 == 0.0


def test_a_destroyed_gang_member_is_not_counted_twice() -> None:
    """A pod the control plane destroyed (requeued=False) IS its final
    PodEvent, which already covers that attempt."""
    job = Job(1, "a", 0.0, 100.0, gpus=1, gang_size=2)
    pods = [PodEvent(1, 0, 0.0, 0.0, 50.0, "big-0", preempted=True),
            PodEvent(1, 1, 30.0, 30.0, 130.0, "big-0")]
    gone = Eviction(1, 0, "big-0", 1, 0.0, 0.0, 50.0, 80.0, "control-plane", requeued=False)
    with_record = compute(Observation("t", FLEET, [job], pods, horizon=200.0,
                                      evictions=[gone]))
    assert with_record.gang_stranded_gpu_hours == 30.0 / H


def test_evicted_gang_stranding_end_to_end_on_the_model() -> None:
    """A 2 x 4 gang with a constant 60 s startup gap, evicted once: both the
    evicted attempt and the requeued one strand 2 x 4 GPUs x 60 s."""
    from k8slab.sim import EvictionRequest

    fleet = Fleet("t", (NodeClass("n", 2, 8),))
    jobs = [Job(1, "a", 0.0, 1000.0, gpus=4, gang_size=2)]
    info = RunInfo(speedup=60.0, startup_delay_ms=(1000.0, 1000.0))
    obs = run(fleet, jobs, "D-fifo", run_info=info, evictions=[EvictionRequest((1, 0), 200.0)])
    assert len(obs.evictions) == 2 and all(e.start_time == 60.0 for e in obs.evictions)
    m = compute(obs)
    assert m.gang_stranded_gpu_hours == pytest.approx(2 * (2 * 4 * 60.0) / H)


def test_no_gang_means_undefined_assembly_and_zero_stranding() -> None:
    m = compute(Observation("t", FLEET, [Job(1, "a", 0.0, 10.0, 1)],
                            [PodEvent(1, 0, 0.0, 0.0, 10.0, "big-0")], horizon=10.0))
    assert m.gang_jobs == 0 and m.gang_stranded_gpu_hours == 0.0
    assert m.gang_assembly_p50 is None


def test_stranded_gang_time_is_worst_for_random_placement_on_the_model() -> None:
    """Random pod order scatters a gang's members across passes; the metric
    must see that, as gang deadlock rate did."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=250, arrival_interval=12.0), seed=0)
    by = {p: compute(run(fleet, jobs, p)) for p in ("D-fifo", "D-random", "D-largest")}
    assert by["D-random"].gang_stranded_gpu_hours > 2 * by["D-fifo"].gang_stranded_gpu_hours
    for m in by.values():
        assert 0.0 <= m.gang_stranded_share < 1.0


# ---- size bias: head-of-line blocking / starvation (M3) ------------------------


def test_footprint_buckets() -> None:
    cases = {0: None, 1: "1", 2: "2", 3: "3-4", 4: "3-4", 5: "5-8", 8: "5-8",
             9: "9-16", 16: "9-16", 17: "17+", 64: "17+"}
    for gpus, label in cases.items():
        assert footprint_bucket(gpus) == label
    assert [b for b, _, _ in FOOTPRINT_BUCKETS] == ["1", "2", "3-4", "5-8", "9-16", "17+"]


def test_average_ranks_share_ties() -> None:
    assert average_ranks([10.0, 20.0, 20.0, 5.0]) == [2.0, 3.5, 3.5, 1.0]
    assert average_ranks([]) == []


def test_spearman_by_hand_with_ties() -> None:
    # ranks x: 1.5 1.5 3 4 ; ranks y: 1 2 4 3 -> Pearson of ranks
    rho = spearman([1.0, 1.0, 2.0, 8.0], [5.0, 6.0, 30.0, 20.0])
    assert rho == pytest.approx(0.7378647873726218, rel=1e-12)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([1.0], [1.0]) is None
    assert spearman([4.0, 4.0, 4.0], [1.0, 2.0, 3.0]) is None


@pytest.mark.skipif(sys.version_info < (3, 12), reason="method='ranked' is 3.12+")
def test_spearman_agrees_with_the_stdlib_ranked_correlation() -> None:
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=200), seed=1)
    obs = run(fleet, jobs, "D-random")
    m = compute(obs)
    last: dict[int, float] = {}
    for p in obs.pods:
        assert p.scheduled_time is not None
        last[p.job_id] = max(last.get(p.job_id, 0.0), p.scheduled_time)
    xs = [float(j.total_gpus) for j in jobs]
    ys = [max(0.0, last[j.job_id] - j.submit_time) for j in jobs]
    expected = statistics.correlation(xs, ys, method="ranked")  # type: ignore[call-arg,unused-ignore]
    assert m.footprint_wait_spearman == pytest.approx(expected, rel=1e-9)


def _waits_obs(spec: list[tuple[int, int, float | None]]) -> Observation:
    """spec: (gpus per pod, gang size, wait or None=never bound)."""
    fleet = Fleet("t", (NodeClass("big", 4, 8),))
    jobs, pods = [], []
    for i, (gpus, gang, wait) in enumerate(spec, start=1):
        jobs.append(Job(i, "a", 0.0, 10.0, gpus, gang_size=gang))
        for k in range(gang):
            if wait is None:
                pods.append(PodEvent(i, k))
            else:
                pods.append(PodEvent(i, k, wait, wait, wait + 10.0, f"big-{k % 4}"))
    return Observation("t", fleet, jobs, pods, horizon=1000.0)


def test_starvation_ratio_and_buckets() -> None:
    m = compute(_waits_obs([
        (1, 1, 10.0), (1, 1, 30.0),  # 1-GPU: mean 20
        (8, 1, 100.0),  # 8 GPUs = largest node: large
        (4, 4, 140.0),  # 16 GPUs: large
        (2, 1, 50.0),
        (4, 1, None),  # never admitted
    ]))
    assert m.large_job_gpus == 8
    assert m.large_job_starvation_ratio == pytest.approx(120.0 / 20.0)
    b = m.wait_by_footprint
    assert b["1"] == {"jobs": 2.0, "admitted": 2.0, "mean": 20.0, "p95": 30.0}
    assert b["3-4"] == {"jobs": 1.0, "admitted": 0.0, "mean": None, "p95": None}
    assert b["9-16"]["mean"] == 140.0 and b["17+"]["jobs"] == 0.0
    assert m.footprint_wait_spearman is not None and m.footprint_wait_spearman > 0.9


def test_starvation_ratio_is_undefined_rather_than_infinite() -> None:
    assert compute(_waits_obs([(8, 1, 100.0)])).large_job_starvation_ratio is None
    zero = compute(_waits_obs([(1, 1, 0.0), (8, 1, 100.0)]))
    assert zero.large_job_starvation_ratio is None


def test_largest_first_inverts_the_size_bias_on_the_model() -> None:
    """D-largest serves big jobs first: small jobs wait, the correlation goes
    negative and the starvation ratio drops below 1. D-random is the opposite."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=300, arrival_interval=12.0), seed=0)
    largest = compute(run(fleet, jobs, "D-largest"))
    rnd = compute(run(fleet, jobs, "D-random"))
    assert largest.footprint_wait_spearman is not None and largest.footprint_wait_spearman < 0
    assert rnd.footprint_wait_spearman is not None and rnd.footprint_wait_spearman > 0
    assert (largest.large_job_starvation_ratio or 0) < 1 < (rnd.large_job_starvation_ratio or 0)


# ---- placement (M4) ------------------------------------------------------------


def _placed(nodes: list[str], gpus: int = 2) -> Observation:
    fleet = load("fleets/default.yaml")
    job = Job(1, "a", 0.0, 10.0, gpus, gang_size=len(nodes))
    pods = [PodEvent(1, i, 0.0, 0.0, 10.0, n) for i, n in enumerate(nodes)]
    return Observation("t", fleet, [job], pods, horizon=10.0)


def test_placement_is_measured_from_bound_nodes() -> None:
    m = compute(_placed(["dgx8-0", "dgx8-4"]))  # rack-0, rack-1 -> one switch
    assert m.placement_multi_pod_jobs == 1
    assert m.placement_tier_share["switch"] == 1.0
    assert m.placement_penalty_mean == 1.8
    assert m.topology_declared
    assert m.penalty_factors["cross_switch"] == 2.2


def test_placement_penalty_uses_the_given_factors_and_records_them() -> None:
    f = PenaltyFactors(switch=2.0, cross_switch=2.5)
    m = compute(_placed(["dgx8-0", "dgx8-4"]), factors=f)
    assert m.placement_penalty_mean == 2.0 and m.penalty_factors["switch"] == 2.0


def test_observation_topology_overrides_the_fleet_derivation() -> None:
    obs = _placed(["dgx8-0", "dgx8-4"])
    topo = derive(obs.fleet)
    moved = dict(topo.nodes)
    # As if the labels read back from the cluster put dgx8-4 in rack-0.
    moved["dgx8-4"] = NodeTopology("dgx8-4", "dgx8", 8, True, rack="rack-0", switch="switch-0")
    obs.topology = Topology(nodes=moved, declared=True)
    assert compute(obs).placement_tier_share["rack"] == 1.0


def test_pod_on_a_node_outside_the_fleet_fails_loudly() -> None:
    with pytest.raises(KeyError, match="outside the fleet"):
        compute(_placed(["dgx8-0", "ghost-1"]))


def test_partially_placed_jobs_have_no_placement_tier() -> None:
    obs = _placed(["dgx8-0", "dgx8-8"])
    obs.pods[1] = PodEvent(1, 1)
    m = compute(obs)
    assert m.placement_multi_pod_jobs == 0 and m.placement_penalty_mean is None


# ---- identity and flattening ---------------------------------------------------


def test_trace_digest_identifies_the_trace() -> None:
    a = generate(WorkloadProfile(job_count=30), seed=0)
    assert trace_digest(a) == trace_digest(generate(WorkloadProfile(job_count=30), seed=0))
    assert trace_digest(a) != trace_digest(generate(WorkloadProfile(job_count=30), seed=1))
    m = compute(run(load("fleets/default.yaml"), a, "D-fifo"))
    assert m.trace_digest == trace_digest(a) and m.fleet == "default"


def test_scalars_flatten_breakdowns_and_skip_identity() -> None:
    m = compute(_waits_obs([(1, 1, 10.0), (8, 1, 100.0)]))
    s = m.scalars()
    assert s["wait_by_footprint.1.mean"] == 10.0
    assert s["wait_by_footprint.3-4.mean"] is None
    # No multi-pod job: the tier shares are undefined (None), never 0.0.
    assert s["placement_tier_share.cross-switch"] is None
    assert s["gang_assembly_p50"] is None
    assert "config" not in s and "trace_digest" not in s and "topology_declared" not in s
    assert "penalty_factors" not in s


# ---- definition A with evictions ----------------------------------------------------


def _pending_brute(obs: Observation, t: float) -> int | None:
    """The definition, scanned pod by pod: pending = submitted and not bound
    at t, where a requeued eviction means the pod was bound on
    [bind, evict) of that attempt, and a pod the control plane destroyed
    before any bind left the queue at its destruction."""
    jobs = {j.job_id: j for j in obs.jobs}
    sizes: list[int] = []
    for p in obs.pods:
        job = jobs[p.job_id]
        if job.gpus <= 0 or t < job.submit_time:
            continue
        if p.preempted and p.scheduled_time is None and p.end_time is not None \
                and t >= p.end_time:
            continue
        bound_now = p.scheduled_time is not None and p.scheduled_time <= t
        for e in obs.evictions:
            if e.requeued and (e.job_id, e.pod_index) == (p.job_id, p.pod_index):
                bound_now = bound_now or e.bind_time <= t < e.evict_time
        if not bound_now:
            sizes.append(job.gpus)
    return min(sizes) if sizes else None


def test_definition_a_does_not_count_an_evicted_pod_as_pending_while_it_was_bound() -> None:
    jobs = [Job(1, "a", 0.0, 100.0, gpus=1), Job(2, "a", 0.0, 100.0, gpus=4)]
    pods = [PodEvent(1, 0, 100.0, 100.0, 200.0, "big-0"),
            PodEvent(2, 0, 150.0, 150.0, 250.0, "big-0")]
    evicted = Eviction(1, 0, "small-0", 1, 10.0, 10.0, 50.0, 80.0, "test")
    obs = Observation("t", FLEET, jobs, pods, horizon=250.0, evictions=[evicted])
    times = [0.0, 20.0, 60.0, 120.0, 160.0]
    got = _min_pending_at(obs, {j.job_id: j for j in jobs}, times)
    # At t=20 the 1-GPU pod is bound (its first attempt): only the 4-GPU pod
    # is pending. Treating [submit, final bind) as pending said 1.
    assert got == {0.0: 1, 20.0: 4, 60.0: 1, 120.0: 4, 160.0: None}
    assert all(got[t] == _pending_brute(obs, t) for t in times)


def test_definition_a_sweep_matches_the_definition_on_a_preempting_run() -> None:
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=160, arrival_interval=10.0), seed=3)
    obs = run(fleet, jobs, "D-preempt")
    assert any(e.requeued for e in obs.evictions)  # the case under test
    times = sorted({t for t, _ in obs.gpu_free_samples})
    got = _min_pending_at(obs, {j.job_id: j for j in jobs}, times)
    assert all(got[t] == _pending_brute(obs, t) for t in times[::7])


def test_definition_a_stops_counting_a_pod_destroyed_before_its_bind_was_seen() -> None:
    """The runner's _destroyed with no bound attempt (a pod seen Terminating
    before its binding was observed, or one that vanished never seen bound):
    ``preempted``, ``end_time`` = the destruction, never scheduled, no
    Eviction record. It left the queue when it was destroyed; its request
    stayed in m(t) until the horizon instead. One 8-GPU node with 2 GPUs free
    for 100 s behind a 6-GPU job, a 4-GPU pod destroyed at 10: stranded
    2 GPUs x 10 s, not x 100 s (200 GPU-s, 20% of free GPU-time)."""
    fleet = Fleet("one", (NodeClass("n", 1, 8),))
    jobs = [Job(1, "a", 0.0, 100.0, gpus=6), Job(2, "a", 0.0, 100.0, gpus=4)]
    pods = [PodEvent(1, 0, 0.0, 0.0, 100.0, "n-0"),
            PodEvent(2, 0, None, None, 10.0, None, preempted=True)]
    samples = [(float(t), {"n-0": 2 if t < 100 else 8}) for t in range(0, 201, 10)]
    obs = Observation("t", fleet, jobs, pods, gpu_free_samples=samples, horizon=200.0)
    times = [t for t, _ in samples]
    got = _min_pending_at(obs, {j.job_id: j for j in jobs}, times)
    assert got[0.0] == 4 and all(got[t] is None for t in times[1:])
    assert all(got[t] == _pending_brute(obs, t) for t in times)
    m = compute(obs)
    assert m.stranded_gpu_hours * H == pytest.approx(2 * 10.0)
    assert m.fragmentation_rate == pytest.approx(20.0 / (2 * 100.0 + 8 * 100.0))
    # A pod never bound and never destroyed is still pending to the horizon.
    alive = dataclasses.replace(obs, pods=[pods[0], PodEvent(2, 0)])
    assert compute(alive).stranded_gpu_hours * H == pytest.approx(2 * 100.0)


# ---- the two denominators of definition A ---------------------------------------------


def test_the_two_definition_a_denominators_differ_by_the_idle_share() -> None:
    """fragmentation_of_fleet = fragmentation_rate x (free GPU-time / fleet
    GPU-time), exactly. The free share is close to 1 - utilization -- not to
    utilization, the factor docs/metrics.md used to name."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=250, arrival_interval=12.0), seed=0)
    obs = run(fleet, jobs, "D-fifo")
    m = compute(obs)
    s = obs.gpu_free_samples
    free = math.fsum(
        sum(f.values()) * (t1 - t0) for (t0, f), (t1, _) in zip(s, s[1:], strict=False)
    )
    share = free / (fleet.total_gpus * obs.horizon)
    assert m.fragmentation_rate > 0
    assert m.fragmentation_of_fleet == pytest.approx(m.fragmentation_rate * share, rel=1e-12)
    assert abs(share - (1 - m.utilization)) < abs(share - m.utilization)
    assert share == pytest.approx(1 - m.utilization, abs=0.02)


# ---- fairness counts the trace's work, not the topology stretch ------------------------


def test_fairness_subtracts_each_accounts_topology_extension() -> None:
    """Under topology_penalty=extend a stretched pod's Running time includes
    the ASSUMED stretch. V_a is the trace's work: account a's 1-GPU job ran
    150 s for 100 s of work, so a got exactly what it asked for -- a ratio of
    1.0, not 1.5 -- while delivered GPU-hours still include the stretch."""
    jobs = [Job(1, "a", 0.0, 100.0, gpus=1), Job(2, "b", 0.0, 100.0, gpus=1)]
    pods = [PodEvent(1, 0, 0.0, 0.0, 150.0, "big-0"), PodEvent(2, 0, 0.0, 0.0, 100.0, "big-0")]
    obs = Observation("t", FLEET, jobs, pods, horizon=150.0,
                      run=RunInfo(topology_penalty="extend"),
                      extension_gpu_seconds=50.0, extension_by_account={"a": 50.0})
    m = compute(obs)
    assert m.service_ratio == {"a": 1.0, "b": 1.0} and m.fairness_ratio == 1.0
    assert m.gpu_hours_used == pytest.approx(250.0 / H)
    assert m.topology_extension_gpu_hours == pytest.approx(50.0 / H)
    # A total with no per-account split cannot be taken out: refused, not
    # silently scored as service.
    with pytest.raises(ValueError, match="extension_by_account"):
        compute(dataclasses.replace(obs, extension_by_account={}))
