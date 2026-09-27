"""Definition C: structural fragmentation.

``test_golden_vector`` is shared VERBATIM with slurm-scheduler-lab: both labs
implement the same pure function and must produce the same numbers on the same
input. Do not edit the vector or the expected values in one lab only.
"""

from __future__ import annotations

import random

import pytest

from k8slab.fleet import load
from k8slab.fragmentation import structural_fragmentation
from k8slab.metrics import compute
from k8slab.model import Job
from k8slab.sim import run
from k8slab.topology import derive
from k8slab.trace import WorkloadProfile, generate

# ---- DEFINITION C GOLDEN TEST VECTOR (shared with slurm-scheduler-lab) ------
# Nodes: n0 (rack r0, switch s0, capacity 8 GPUs), n1 (rack r0, switch s0,
# capacity 8), n2 (rack r1, switch s0, capacity 4), n3 (rack r1, switch s0,
# capacity 4).
# Free-GPU samples (step function: each sample holds until the next sample):
#   t=0:  {n0: 8, n1: 3, n2: 4, n3: 0}
#   t=10: {n0: 8, n1: 8, n2: 4, n3: 4}
#   t=20: {n0: 8, n1: 8, n2: 4, n3: 4}   (horizon)
# Expected: frag_C^node = 30/390 = 1/13; frag_C^rack = 150/390 = 5/13;
#           frag_C^switch = 150/390 = 5/13.
GOLDEN_CAPACITY = {"n0": 8, "n1": 8, "n2": 4, "n3": 4}
GOLDEN_DOMAINS = {
    "rack": {"n0": "r0", "n1": "r0", "n2": "r1", "n3": "r1"},
    "switch": {"n0": "s0", "n1": "s0", "n2": "s0", "n3": "s0"},
}
GOLDEN_SAMPLES: list[tuple[float, dict[str, int]]] = [
    (0.0, {"n0": 8, "n1": 3, "n2": 4, "n3": 0}),
    (10.0, {"n0": 8, "n1": 8, "n2": 4, "n3": 4}),
    (20.0, {"n0": 8, "n1": 8, "n2": 4, "n3": 4}),
]


def test_golden_vector() -> None:
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    # The integrals themselves, so a disagreement between labs can be located.
    assert r.free_gpu_seconds == 390.0
    assert r.carved_gpu_seconds == {"node": 30.0, "rack": 150.0, "switch": 150.0}
    assert r.rate("node") == 30 / 390 == pytest.approx(1 / 13, abs=1e-15)
    assert r.rate("rack") == 150 / 390 == pytest.approx(5 / 13, abs=1e-15)
    assert r.rate("switch") == 150 / 390 == pytest.approx(5 / 13, abs=1e-15)
    assert round(r.rate("node"), 7) == 0.0769231
    assert round(r.rate("rack"), 7) == 0.3846154
    assert r.levels == ("node", "rack", "switch")


# ---- the properties the definition promises --------------------------------


def _random_nested(rng: random.Random, nodes: int) -> tuple[
    dict[str, int], dict[str, dict[str, str]], list[tuple[float, dict[str, int]]]
]:
    names = [f"x{i}" for i in range(nodes)]
    capacity = {n: rng.choice([0, 1, 2, 4, 8]) for n in names}
    racks = {n: f"r{i // rng.randint(1, 4)}" for i, n in enumerate(names)}
    # Racks nest into switches by construction: switch is a function of rack.
    switch_of_rack = {r: f"s{int(r[1:]) // 2}" for r in set(racks.values())}
    domains = {"rack": racks, "switch": {n: switch_of_rack[racks[n]] for n in names}}
    t = 0.0
    samples = []
    for _ in range(rng.randint(2, 12)):
        samples.append((t, {n: rng.randint(0, c) for n, c in capacity.items()}))
        t += rng.choice([0.0, 1.0, 2.5, 7.0])
    return capacity, domains, samples


def test_levels_are_monotone_on_random_nested_topologies() -> None:
    rng = random.Random(7)
    for _ in range(300):
        cap, dom, samples = _random_nested(rng, rng.randint(1, 12))
        r = structural_fragmentation(cap, dom, samples).rates()
        assert r["node"] <= r["rack"] + 1e-12
        assert r["rack"] <= r["switch"] + 1e-12
        assert all(0.0 <= v <= 1.0 for v in r.values())


@pytest.mark.parametrize("policy", ["D-fifo", "D-random", "D-largest"])
def test_levels_are_monotone_on_reference_model_runs(policy: str) -> None:
    fleet = load("fleets/default.yaml")
    m = compute(run(fleet, generate(WorkloadProfile(job_count=150), seed=2), policy))
    assert m.fragmentation_structural <= m.fragmentation_structural_rack
    assert m.fragmentation_structural_rack <= m.fragmentation_structural_switch


def test_idle_fleet_is_zero_at_every_level() -> None:
    samples = [(0.0, dict(GOLDEN_CAPACITY)), (50.0, dict(GOLDEN_CAPACITY))]
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples)
    assert r.free_gpu_seconds > 0
    assert r.rates() == {"node": 0.0, "rack": 0.0, "switch": 0.0}


def test_fully_allocated_fleet_has_no_denominator_and_reports_zero() -> None:
    full = dict.fromkeys(GOLDEN_CAPACITY, 0)
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, full), (9.0, full)])
    assert r.free_gpu_seconds == 0.0
    assert r.rates() == {"node": 0.0, "rack": 0.0, "switch": 0.0}


def test_homogeneous_fleet_running_only_whole_node_jobs_has_zero_node_level() -> None:
    """The control. Whole-node jobs can never leave a node partly allocated."""
    fleet = load("fleets/homogeneous.yaml")
    jobs = [
        Job(job_id=i, account="a", submit_time=30.0 * i, duration=600.0 + 97.0 * (i % 5),
            gpus=8, gang_size=1 + i % 3)
        for i in range(1, 60)
    ]
    for policy in ("D-fifo", "D-random", "D-largest"):
        m = compute(run(fleet, jobs, policy))
        assert m.jobs_completed == len(jobs)
        assert m.fragmentation_structural == 0.0
        # Not a vacuous zero: whole nodes were busy and whole racks were carved.
        assert m.fragmentation_structural_rack > 0.0


def test_node_level_is_definition_b_with_the_node_as_reference_on_one_shape() -> None:
    """On a single-shape fleet whose largest pod equals the node size, a node
    is carved exactly when 0 < free < reference -- so C-node and B coincide.
    On a mixed fleet they do not, which is why both are reported."""
    jobs = generate(WorkloadProfile(job_count=150), seed=4)
    hom = compute(run(load("fleets/homogeneous.yaml"), jobs, "D-fifo"))
    assert hom.reference_request == 8
    assert hom.fragmentation_structural == pytest.approx(hom.fragmentation_ref, rel=1e-12)
    het = compute(run(load("fleets/default.yaml"), jobs, "D-fifo"))
    assert het.fragmentation_structural != pytest.approx(het.fragmentation_ref, rel=1e-3)


def test_zero_length_intervals_and_the_final_sample_are_not_integrated() -> None:
    doubled = [GOLDEN_SAMPLES[0], GOLDEN_SAMPLES[0], *GOLDEN_SAMPLES[1:]]
    # A last sample with absurd-but-valid values changes nothing: it only
    # closes the final interval.
    tail = [*GOLDEN_SAMPLES[:-1], (20.0, {"n0": 0, "n1": 0, "n2": 0, "n3": 0})]
    base = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    for samples in (doubled, tail):
        assert structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples) == base


def test_fewer_than_two_samples_integrate_nothing() -> None:
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES[:1])
    assert r.free_gpu_seconds == 0.0 and r.rates()["node"] == 0.0


def test_node_level_alone_needs_no_domains() -> None:
    r = structural_fragmentation(GOLDEN_CAPACITY, {}, GOLDEN_SAMPLES)
    assert r.levels == ("node",)
    assert r.rate("node") == 30 / 390


def test_fleet_topology_feeds_the_function_directly() -> None:
    topo = derive(load("fleets/default.yaml"))
    cap = topo.capacity()
    r = structural_fragmentation(cap, topo.domains(), [(0.0, cap), (1.0, cap)])
    assert r.levels == ("node", "rack", "switch")


@pytest.mark.parametrize(
    ("capacity", "domains", "samples", "match"),
    [
        (GOLDEN_CAPACITY, {"node": {}}, GOLDEN_SAMPLES, "implicit"),
        (GOLDEN_CAPACITY, {"rack": {"n0": "r0"}}, GOLDEN_SAMPLES, "no domain"),
        (
            GOLDEN_CAPACITY,
            {"rack": {**GOLDEN_DOMAINS["rack"], "zz": "r9"}},
            GOLDEN_SAMPLES,
            "unknown node",
        ),
        (
            # r0 split across two switches: not nested, monotonicity would break.
            GOLDEN_CAPACITY,
            {"rack": GOLDEN_DOMAINS["rack"],
             "switch": {"n0": "s0", "n1": "s1", "n2": "s1", "n3": "s1"}},
            GOLDEN_SAMPLES,
            "does not nest",
        ),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, {"n0": 8}), (1.0, {"n0": 8})], "no free count"),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {"n0": 9, "n1": 0, "n2": 0, "n3": 0}), (1.0, dict(GOLDEN_CAPACITY))],
            "outside",
        ),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(5.0, dict(GOLDEN_CAPACITY)), (1.0, dict(GOLDEN_CAPACITY))],
            "out of time order",
        ),
        ({"n0": -1}, {}, [], "capacity"),
        # Samples that are never integrated must still be valid: the final
        # sample, one followed by a zero-length interval, and a lone sample.
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [GOLDEN_SAMPLES[0], (10.0, {"n0": 99, "n1": 8, "n2": 4, "n3": 4})],
            "outside",
        ),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {"n0": -5, "n1": 0, "n2": 0, "n3": 0}), *GOLDEN_SAMPLES],
            "outside",
        ),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [GOLDEN_SAMPLES[0], (10.0, {"n0": 8})], "no free count"),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, {"bogus": 1})], "no free count"),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {**GOLDEN_CAPACITY, "zz": 1}), (1.0, dict(GOLDEN_CAPACITY))],
            "unknown node",
        ),
    ],
)
def test_rejects_malformed_input(
    capacity: dict[str, int],
    domains: dict[str, dict[str, str]],
    samples: list[tuple[float, dict[str, int]]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        structural_fragmentation(capacity, domains, samples)
