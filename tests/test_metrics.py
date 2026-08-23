from __future__ import annotations

from k8slab.fleet import load
from k8slab.metrics import _stranded_at, compute, percentile
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent
from k8slab.sim import run
from k8slab.trace import WorkloadProfile, generate

FLEET = Fleet("t", (NodeClass("big", 1, 8), NodeClass("small", 2, 2)))


def test_percentile_nearest_rank() -> None:
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0
    assert percentile([], 95) == 0.0


def test_idle_cluster_is_not_fragmented() -> None:
    """Capacity nobody wants is idle, not stranded."""
    assert _stranded_at({"a": 4, "b": 2}, None) == 0


def test_stranded_counts_only_unusable_remainders() -> None:
    # smallest pending pod needs 4: the node with 2 free is stranded, 4 is not.
    assert _stranded_at({"a": 2, "b": 4, "c": 0}, 4) == 2


def test_gpu_hours_equal_trace_demand_when_all_jobs_complete() -> None:
    """The check that caught a +11.9% GPU-hour inflation from polling lag."""
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
    assert m.gang_wasted_gpu_hours > 0


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
