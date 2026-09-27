from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

import pytest

from k8slab.fleet import load
from k8slab.metrics import Metrics, compute
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent
from k8slab.sim import run
from k8slab.stats import (
    DEFAULT_CV_TOLERANCE,
    GATED_METRICS,
    MIN_RUNS_TO_RESOLVE,
    Spread,
    aggregate,
    bootstrap_diff_interval,
    compare,
    compare_all,
    dataset_verdict,
    tail_count,
)
from k8slab.trace import WorkloadProfile, generate


def _base() -> Metrics:
    fleet = Fleet("t", (NodeClass("big", 2, 8),))
    obs = Observation(
        "X", fleet, [Job(1, "a", 0.0, 100.0, 1)],
        [PodEvent(1, 0, 0.0, 0.0, 100.0, "big-0")],
        [(0.0, {"big-0": 7, "big-1": 8}), (100.0, {"big-0": 8, "big-1": 8})],
        horizon=100.0,
    )
    return compute(obs)


BASE = _base()


def _run(config: str = "X", **values: Any) -> Metrics:
    return dataclasses.replace(BASE, config=config, **values)


def _runs(config: str, metric: str, values: list[float]) -> list[Metrics]:
    return [_run(config, **{metric: v}) for v in values]


# ---- Spread and the stability rule --------------------------------------------


def test_single_run_is_never_stable() -> None:
    agg = aggregate("X", [_run()])
    assert agg.n == 1 and agg.verdict == "unreplicated" and agg.unstable
    assert set(agg.unstable_metrics) == {m for m in GATED_METRICS if m in agg.spreads}
    assert agg.get("makespan_hours").sd is None
    assert "n=1" in agg.stability_note() and "do not quote" in agg.stability_note()
    assert agg.get("utilization").format(scale=100) == f"{BASE.utilization * 100:.1f}"


def test_cv_rule_is_relative_so_one_tolerance_serves_every_unit() -> None:
    # The same 3% relative spread is stable in hours and in seconds alike ...
    for metric, centre in (("makespan_hours", 20.0), ("mean_wait", 9000.0)):
        agg = aggregate("X", _runs("X", metric, [centre * 0.97, centre, centre * 1.03]))
        assert metric not in agg.unstable_metrics
        assert agg.get(metric).cv == pytest.approx(0.03, rel=1e-9)
    # ... and a 10% one is unstable in both.
    for metric, centre in (("makespan_hours", 20.0), ("mean_wait", 9000.0)):
        agg = aggregate("X", _runs("X", metric, [centre * 0.9, centre, centre * 1.1]))
        assert agg.unstable_metrics == [metric]
        assert agg.verdict == "unstable"
        assert "UNSTABLE" in agg.stability_note()


def test_resolution_floor_stops_cv_exploding_near_zero() -> None:
    """Definition A reads ~0.003% for D-largest. A spread in the fifth decimal
    is a CV of ~70% and invisible in every table; it must not trip the gate."""
    agg = aggregate("X", _runs("X", "fragmentation_rate", [0.00001, 0.00005, 0.00003]))
    s = agg.get("fragmentation_rate")
    assert s.cv is not None and s.cv > 0.5
    assert s.sd is not None and s.sd < GATED_METRICS["fragmentation_rate"]
    assert "fragmentation_rate" not in agg.unstable_metrics
    # A spread a reader could see is still caught.
    agg = aggregate("X", _runs("X", "fragmentation_rate", [0.001, 0.005, 0.003]))
    assert "fragmentation_rate" in agg.unstable_metrics


def test_zero_mean_with_visible_spread_is_unstable() -> None:
    s = Spread("footprint_wait_spearman", mean=0.0, sd=0.2, n=3, minimum=-0.2, maximum=0.2)
    assert s.cv == math.inf
    assert s.is_unstable(DEFAULT_CV_TOLERANCE, GATED_METRICS["footprint_wait_spearman"])
    still = Spread("x", mean=0.0, sd=0.0, n=3, minimum=0.0, maximum=0.0)
    assert still.cv is None and not still.is_unstable()


def test_identical_repeats_are_labelled_deterministic_not_just_stable() -> None:
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=60), seed=0)
    runs = [compute(run(fleet, jobs, "D-fifo")) for _ in range(3)]
    agg = aggregate("D-fifo", runs)
    assert not agg.unstable
    assert agg.deterministic and agg.verdict == "deterministic"
    assert "deterministic process" in agg.stability_note()


def test_every_scalar_is_summarised_and_raw_runs_are_kept() -> None:
    runs = _runs("X", "makespan_hours", [1.0, 1.1, 1.2])
    agg = aggregate("X", runs)
    assert agg.runs == runs
    assert "wait_by_footprint.1.mean" in agg.spreads
    assert "service_ratio.a" in agg.spreads
    # These runs placed no multi-pod job: the tier shares are undefined, not
    # 0%, so they are not summarised ...
    assert "placement_tier_share.rack" not in agg.spreads
    # ... and a run that did place one is not averaged with zeros.
    placed = {"node": 0.0, "rack": 1.0, "switch": 0.0, "cross-switch": 0.0}
    mixed = aggregate("X", [*runs, _run(placement_tier_share=placed)])
    assert mixed.get("placement_tier_share.rack").mean == 1.0
    assert mixed.get("placement_tier_share.rack").n == 1
    assert "config" not in agg.spreads and "measured_on_cluster" not in agg.spreads


def test_undefined_values_are_skipped_not_counted_as_zero() -> None:
    runs = [
        _run(footprint_wait_spearman=None),
        _run(footprint_wait_spearman=0.4),
        _run(footprint_wait_spearman=0.5),
    ]
    agg = aggregate("X", runs)
    assert agg.get("footprint_wait_spearman").n == 2
    assert agg.get("footprint_wait_spearman").mean == pytest.approx(0.45)
    assert agg.values("footprint_wait_spearman") == [0.4, 0.5]
    # Undefined in every run: no spread at all, and it does not gate.
    agg = aggregate("X", [_run(gang_assembly_p50=None)] * 2)
    assert "gang_assembly_p50" not in agg.spreads


@pytest.mark.parametrize(
    ("runs", "match"),
    [
        ([], "zero runs"),
        ([_run("A"), _run("B")], "disagree"),
        ([_run(), _run(measured_on_cluster=True)], "cluster and reference-model"),
        ([_run(), _run(fleet="other")], "different fleets"),
        ([_run(), _run(penalty_factors={**BASE.penalty_factors, "rack": 1.5})], "penalty"),
    ],
)
def test_aggregate_refuses_runs_that_are_not_one_experiment(
    runs: list[Metrics], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate(runs[0].config if runs else "X", runs)


def test_mixed_traces_are_reported_not_hidden() -> None:
    agg = aggregate("X", [_run(trace_digest="aaa"), _run(trace_digest="bbb")])
    assert agg.trace_digests == ["aaa", "bbb"]
    assert "workload variance" in agg.stability_note()


def test_dataset_verdict() -> None:
    good = aggregate("A", _runs("A", "makespan_hours", [10.0, 10.1, 10.2]))
    bad = aggregate("B", [_run("B")])
    assert dataset_verdict([good]) == (True, [good.stability_note()])
    stable, notes = dataset_verdict([good, bad])
    assert not stable and len(notes) == 2


# ---- bootstrap ------------------------------------------------------------------


def test_bootstrap_is_reproducible_and_seed_sensitive() -> None:
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.5, 3.0, 3.5, 9.0]
    first = bootstrap_diff_interval(a, b, rng=random.Random("s"), resamples=500)
    again = bootstrap_diff_interval(a, b, rng=random.Random("s"), resamples=500)
    other = bootstrap_diff_interval(a, b, rng=random.Random("t"), resamples=500)
    assert first == again
    assert first != other


def test_bootstrap_interval_brackets_the_point_difference() -> None:
    a, b = [10.0, 11.0, 12.0, 13.0, 14.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_diff_interval(a, b, rng=random.Random(0))
    assert 5.0 < lo <= 9.0 <= hi < 13.0


def test_bootstrap_rejects_empty_sides() -> None:
    with pytest.raises(ValueError):
        bootstrap_diff_interval([], [1.0], rng=random.Random(0))


def _resampled(a: list[float], b: list[float], seed: int, resamples: int) -> list[float]:
    """The sorted resampled differences bootstrap_diff_interval draws."""
    rng = random.Random(seed)
    return sorted(
        math.fsum(rng.choices(a, k=len(a))) / len(a) - math.fsum(rng.choices(b, k=len(b))) / len(b)
        for _ in range(resamples)
    )


@pytest.mark.parametrize(
    ("resamples", "confidence", "k"),
    [(4000, 0.95, 100), (4000, 0.90, 200), (4000, 0.80, 400), (1000, 0.99, 5), (1, 0.95, 0)],
)
def test_bootstrap_cuts_the_same_count_from_each_tail(
    resamples: int, confidence: float, k: int
) -> None:
    """The interval is [d[k], d[R-1-k]]. Two separately rounded indices used to
    cut 199 values below and 200 above at confidence 0.90 (1 - 0.9 is
    0.09999999999999998), and were documented as nearest-rank quantiles that
    the lower one was not."""
    # Irrational-valued samples, so resampled means (almost) never tie and
    # neighbouring ranks are distinguishable.
    a = [math.sqrt(i + 2) for i in range(12)]
    b = [math.log(i + 3) for i in range(9)]
    assert tail_count(resamples, confidence) == k
    diffs = _resampled(a, b, 1, resamples)
    if resamples > 1:
        assert diffs[k] != diffs[k - 1] if k else diffs[k] != diffs[k + 1]
    got = bootstrap_diff_interval(a, b, rng=random.Random(1), resamples=resamples,
                                  confidence=confidence)
    assert got == (diffs[k], diffs[resamples - 1 - k])


def test_the_default_interval_is_unchanged_by_the_tail_rule() -> None:
    """At 0.95 and 4000 resamples the old indices (floor(0.025 R) = 100 and
    ceil(0.975 R) - 1 = 3899) are exactly k and R-1-k: no published interval
    moved."""
    assert tail_count(4000, 0.95) == 100 and 4000 - 1 - 100 == 3899


def test_tail_count_rejects_what_has_no_interval() -> None:
    for confidence in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            tail_count(100, confidence)
    with pytest.raises(ValueError, match="resamples"):
        tail_count(0, 0.95)
    assert tail_count(3, 0.01) == 1  # capped: never an inverted interval


def _agg(config: str, metric: str, values: list[float], **extra: Any) -> Any:
    return aggregate(config, [_run(config, **{metric: v}, **extra) for v in values])


def test_separated_samples_are_resolvable_and_overlapping_are_not() -> None:
    hi = _agg("K0", "utilization", [0.70, 0.71, 0.69, 0.72, 0.70])
    lo = _agg("D-random", "utilization", [0.60, 0.61, 0.59, 0.62, 0.60])
    c = compare(hi, lo, "utilization")
    assert c.resolvable and c.low is not None and c.high is not None
    assert c.low > 0 and c.diff == pytest.approx(0.10)
    same = _agg("D-fifo", "utilization", [0.61, 0.59, 0.62, 0.58, 0.60])
    c = compare(same, lo, "utilization")
    assert not c.resolvable and c.low is not None and c.low < 0 < (c.high or 0)


def test_small_n_never_resolves_even_when_the_interval_excludes_zero() -> None:
    n = MIN_RUNS_TO_RESOLVE - 1
    a = _agg("K0", "utilization", [0.9 + 0.001 * i for i in range(n)])
    b = _agg("D-fifo", "utilization", [0.1 + 0.001 * i for i in range(n)])
    c = compare(a, b, "utilization")
    assert c.low is not None and c.low > 0  # excludes zero ...
    assert not c.resolvable  # ... and is still not trusted
    assert f"n<{MIN_RUNS_TO_RESOLVE}" in c.note
    one = compare(_agg("K0", "utilization", [0.9]), b, "utilization")
    assert one.low is None and not one.resolvable and one.diff is not None


@pytest.mark.parametrize(("n_a", "n_b"), [(10, 2), (2, 10), (5, 4), (4, 5)])
def test_the_minimum_run_count_applies_to_the_smaller_side(n_a: int, n_b: int) -> None:
    """Unequal n is normal: results.json is rewritten after every interleaved
    run, so an interrupted bench leaves some configurations one run short. A
    side below the minimum is never resolvable, however many runs the other
    side has -- the rule is min(n_a, n_b), not max."""
    a = _agg("K0", "utilization", [0.9 + 0.001 * i for i in range(n_a)])
    b = _agg("D-fifo", "utilization", [0.1 + 0.001 * i for i in range(n_b)])
    c = compare(a, b, "utilization")
    assert c.low is not None and c.low > 0  # the interval excludes zero ...
    assert not c.resolvable and f"n<{MIN_RUNS_TO_RESOLVE}" in c.note  # ... and is not trusted
    assert (c.n_config, c.n_baseline) == (n_a, n_b)


def test_comparisons_that_mean_nothing_are_refused() -> None:
    a = _agg("K0", "utilization", [0.7] * 5, measured_on_cluster=True)
    b = _agg("D-fifo", "utilization", [0.1] * 5)
    assert "never comparable" in compare(a, b, "utilization").note
    c = _agg("D-random", "utilization", [0.1] * 5, trace_digest="other")
    d = _agg("D-fifo", "utilization", [0.7] * 5)
    refused = compare(d, c, "utilization")
    assert not refused.resolvable and "same experiment" in refused.note


def test_deterministic_difference_is_resolvable_but_labelled() -> None:
    a = _agg("D-fifo", "utilization", [0.7] * 5)
    b = _agg("D-random", "utilization", [0.6] * 5)
    c = compare(a, b, "utilization")
    assert c.resolvable and c.low == c.high == pytest.approx(0.1)
    assert "deterministic" in c.note


def test_compare_all_pairs_every_config_with_every_baseline_once() -> None:
    aggs = [_agg(cfg, "utilization", [0.5, 0.6])
            for cfg in ("K0", "K1", "D-fifo", "D-random", "D-largest")]
    pairs = {(c.config, c.baseline) for c in compare_all(aggs, resamples=50)}
    assert pairs == {
        ("K0", "D-fifo"), ("K0", "D-random"), ("K0", "D-largest"),
        ("K1", "D-fifo"), ("K1", "D-random"), ("K1", "D-largest"), ("K1", "K0"),
        ("D-fifo", "D-random"), ("D-fifo", "D-largest"), ("D-random", "D-largest"),
    }
    metrics = {c.metric for c in compare_all(aggs, resamples=50)}
    assert metrics == {"makespan_hours", "utilization", "mean_wait", "fragmentation_structural"}


def test_adding_a_config_never_moves_another_interval() -> None:
    a = _agg("K0", "utilization", [0.70, 0.72, 0.69, 0.73, 0.71])
    b = _agg("D-fifo", "utilization", [0.60, 0.65, 0.61, 0.63, 0.62])
    extra = _agg("D-random", "utilization", [0.5, 0.52, 0.51, 0.49, 0.5])
    only = [c for c in compare_all([a, b]) if c.metric == "utilization"]
    more = [c for c in compare_all([a, b, extra])
            if c.metric == "utilization" and {c.config, c.baseline} == {"K0", "D-fifo"}]
    assert only == more


def test_documented_small_n_weakness_holds() -> None:
    """docs/metrics.md states the percentile bootstrap false-alarms well above
    its nominal 5% at n=5. This regression-guards that statement on a smaller,
    seeded replica of the measurement (400 null trials, 500 resamples)."""
    gen = random.Random(12345)
    trials = 400
    hits = 0
    for trial in range(trials):
        a = [gen.gauss(0, 1) for _ in range(5)]
        b = [gen.gauss(0, 1) for _ in range(5)]
        lo, hi = bootstrap_diff_interval(a, b, rng=random.Random(trial), resamples=500)
        hits += lo > 0 or hi < 0
    assert 0.08 < hits / trials < 0.20


def test_a_side_unstable_on_the_metric_is_never_resolvable() -> None:
    """One rule in one report: the stability section says an unstable metric
    is not resolvable by this harness, so no comparison on it may say it is --
    even when the bootstrap interval excludes zero."""
    noisy = _agg("D-random", "fragmentation_structural", [0.20, 0.14, 0.24, 0.12, 0.18])
    fixed = _agg("D-fifo", "fragmentation_structural", [0.10] * 5)
    assert "fragmentation_structural" in noisy.unstable_metrics
    c = compare(fixed, noisy, "fragmentation_structural")
    assert c.high is not None and c.high < 0  # the interval excludes zero ...
    assert not c.resolvable  # ... and the metric is still not resolvable
    assert "unstable on D-random" in c.note
    assert "never call them resolvable" in noisy.stability_note()
    # The same pair on a metric neither side is unstable on still resolves.
    steady = compare(_agg("D-fifo", "utilization", [0.70, 0.71, 0.69, 0.72, 0.70]),
                     _agg("D-random", "utilization", [0.60, 0.61, 0.59, 0.62, 0.60]),
                     "utilization")
    assert steady.resolvable


def test_paired_comparison_refuses_what_does_not_pair() -> None:
    from k8slab.stats import compare_paired

    def agg(config: str, digests: list[str], values: list[float]) -> Any:
        return aggregate(config, [_run(config, utilization=v, trace_digest=d)
                                  for d, v in zip(digests, values, strict=True)])

    a = agg("K0", ["t1", "t2"], [0.5, 0.6])
    other = compare_paired(a, agg("D-fifo", ["t1", "t3"], [0.4, 0.5]), "utilization")
    assert other.diff is None and "different traces" in other.note and other.paired
    twice = compare_paired(a, agg("D-fifo", ["t1", "t1"], [0.4, 0.5]), "utilization")
    assert twice.diff is None
    # compare_all routes the study through the paired method.
    two = compare_paired(a, agg("D-fifo", ["t2", "t1"], [0.55, 0.4]), "utilization")
    assert two.diff == pytest.approx(0.075) and not two.resolvable
    assert "2 traces < 5" in two.note
