"""Golden values for the reference model, so that drift in the published
reference-model numbers fails CI.

CI's bench-model job checks only the results schema and plumbing. Neither it
nor any other test noticed when a mechanism the README's table depends on was
removed (the capacity-freed event, the nomination reservation, the backoffQ
expiry check). These pin the numbers themselves.

Reference model, not a cluster. Kube queue model, speedup 60, trace seed 0,
harness seed 0 -- the settings of `make bench-model`, whose D-fifo, D-largest
and D-preempt rows are deterministic (identical on every repeat), so one run
reproduces them. D-random is not pinned: its repeats vary by design. If a
change moves these deliberately, re-run `make bench-model`, update the
README's reference-model table and docs/metrics.md from it, then update the
values here.
"""

from __future__ import annotations

import pytest

from k8slab import trace
from k8slab.fleet import load
from k8slab.metrics import Metrics, compute
from k8slab.model import RunInfo
from k8slab.sim import run

FLEET = load("fleets/default.yaml")
KUBE = RunInfo(queue_model="kube", speedup=60.0)


def _score(profile: str, policy: str) -> Metrics:
    jobs = trace.generate(trace.profile_named(profile), seed=0)
    return compute(run(FLEET, jobs, policy, seed=0, run_info=KUBE))


#: Light profile (the CI smoke trace): exact values, to 1e-12.
LIGHT: dict[str, dict[str, float]] = {
    "D-fifo": {
        "makespan_hours": 9.29861111111111,
        "mean_wait": 623.4283597096615,
        "p95_wait": 5485.4115939147305,
        "fragmentation_rate": 0.17031598742508713,
        "fragmentation_structural": 0.14707139397277968,
        "gang_stranded_gpu_hours": 23.741733601740986,
        "preemptions": 0,
    },
    "D-preempt": {
        "makespan_hours": 9.666666666666666,
        "mean_wait": 612.6116930429948,
        "p95_wait": 3416.432889287371,
        "fragmentation_rate": 0.13477847462999068,
        "fragmentation_structural": 0.2044064964127585,
        "gang_stranded_gpu_hours": 18.911071202175787,
        "preemptions": 74,
    },
}


@pytest.mark.parametrize("policy", sorted(LIGHT))
def test_light_profile_golden(policy: str) -> None:
    m = _score("light", policy)
    for name, expected in LIGHT[policy].items():
        assert getattr(m, name) == pytest.approx(expected, rel=1e-12, abs=0.0), name


#: The README's reference-model table, as displayed (default profile): makespan
#: h, util %, mean and p95 wait in minutes, frag A/B/C %, gang strand GPU-h,
#: size-wait rho, starvation ratio.
README_TABLE: dict[str, tuple[float, ...]] = {
    "D-fifo": (15.8, 63.3, 122, 520, 29.0, 55.7, 14.0, 71.2, 0.56, 7.85),
    "D-largest": (15.4, 65.3, 126, 476, 23.1, 53.6, 14.6, 80.7, 0.58, 7.20),
    "D-preempt": (15.0, 66.8, 202, 435, 4.0, 42.5, 12.4, 71.3, 0.28, 1.59),
}


@pytest.mark.parametrize("policy", sorted(README_TABLE))
def test_the_readme_reference_model_table_reproduces(policy: str) -> None:
    m = _score("default", policy)
    assert m.large_job_starvation_ratio is not None
    assert m.footprint_wait_spearman is not None
    shown = (
        round(m.makespan_hours, 1), round(m.utilization * 100, 1),
        round(m.mean_wait / 60), round(m.p95_wait / 60),
        round(m.fragmentation_rate * 100, 1), round(m.fragmentation_ref * 100, 1),
        round(m.fragmentation_structural * 100, 1), round(m.gang_stranded_gpu_hours, 1),
        round(m.footprint_wait_spearman, 2), round(m.large_job_starvation_ratio, 2),
    )
    assert shown == README_TABLE[policy]
