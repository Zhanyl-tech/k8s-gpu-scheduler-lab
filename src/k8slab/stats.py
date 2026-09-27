"""Repeated runs, aggregated into a mean and a spread, and compared honestly.

Why this module is the most important one in the repo
----------------------------------------------------

``docs/limitations.md`` records two runs of the *same* configuration, on the
*same* seeded trace, on the same machine, reporting 33.7% and 54.2% utilization
(29.8 h and 18.5 h makespan). The trace is deterministic; the harness is not,
because the replay is driven by real wall-clock and API-server latency feeds
back into the simulated clock.

A single point estimate drawn from that distribution is not a measurement of a
scheduler. It is one sample from a spread wider than every difference the
results table is trying to report. So the unit of reporting here is not a run --
it is :math:`N` runs, summarised as ``mean ± sd``, with an explicit stability
verdict attached, and differences between configurations are called
*resolvable* only when a bootstrap interval says so.

What the stability check does and does not claim
-----------------------------------------------

:meth:`Aggregate.unstable_metrics` flags a gated metric whose **coefficient of
variation** (sd / |mean|) exceeds a tolerance (default 5%). That is a statement
about the harness, not about the scheduler: an unstable metric means this
apparatus cannot yet resolve differences in that quantity, and any ranking
built on it is unsupported.

The check is relative rather than absolute. An absolute :math:`\\sigma > 0.05`
threshold is meaningless across the quantities here -- utilization is a
fraction in [0, 1], makespan is hours, and waits are seconds, so one tolerance
cannot serve all three: 0.05 would be a 5-percentage-point band on utilization,
three minutes on makespan, and a twentieth of a second on wait. The
coefficient of variation is dimensionless and comparable.

It has one failure mode, and the first draft of this module fell into it:
near a zero mean, CV explodes. Definition A reads ~0.003% for ``D-largest``
in the reference model under the Phase 1 binder (``--queue-model none``);
a correlation coefficient can sit near 0. A spread of 0.00002 on a mean of
0.00003 is a CV of 67% and is also invisible in every table the lab prints.
So each gated metric carries an absolute **resolution floor** in its own unit
(:data:`GATED_METRICS`): half of the last digit the report displays. A metric is
unstable only if its sd exceeds that floor *and* its CV exceeds the tolerance.
The floor never rescues a metric whose spread a reader could see.

A single run is never called stable. With :math:`N = 1` the sample standard
deviation is undefined, and reporting 0.0 would assert perfect reproducibility
from the one piece of evidence that cannot demonstrate it.

Identical repeats are called what they are. The reference model is
deterministic, so N repeats of it with the same seed are N copies of one number:
sd = 0 and "stable" would be true and uninformative. :attr:`Aggregate.deterministic`
labels that case so it is never mistaken for evidence about the cluster harness.

What may be pooled, and what may be ranked
------------------------------------------

Runs are pooled into one aggregate only if they are one experiment: same
source (cluster or model), fleet, penalty factors and harness settings
(:meth:`k8slab.model.RunInfo.harness` -- queue model, speedup, startup delay,
topology mode, grace period, gates). The harness *seed* may differ; that is
what a repeat is. Two kinds of row are never ranked against measured rows:
SCENARIO rows (``topology_penalty=extend``: runtimes stretched by assumed
factors, labelled ``+topo``) and GATED rows (admission gate or serialised
kube-scheduler, labelled ``+gated``), which answer a diagnostic question and are
left out of comparisons altogether.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .baselines import DEGENERATE_IDS
from .metrics import Metrics

#: Coefficient of variation (sd / |mean|) above which a metric is unstable.
#: 0.05 is the tolerance the optimisation plan named as "sigma > 0.05", read as a
#: CV for the reasons in the module docstring.
DEFAULT_CV_TOLERANCE = 0.05

#: The metrics whose stability gates the dataset, each with its absolute
#: resolution floor in the metric's own unit: half the last digit the report
#: displays (``tests/test_report.py`` checks the two stay in step). These are
#: the quantities results tables rank configurations on; a spread in, say,
#: ``reference_request`` is structurally impossible and not worth checking.
#: ``gang_deadlock_rate`` was gated in the draft and is not any more: it is
#: deprecated (docs/metrics.md) and ``gang_stranded_gpu_hours`` replaces it.
GATED_METRICS: dict[str, float] = {
    "makespan_hours": 0.05,  # shown to 0.1 h
    "utilization": 0.0005,  # shown to 0.1 %
    "mean_wait": 30.0,  # seconds; shown in whole minutes
    "p95_wait": 30.0,
    "fragmentation_rate": 0.0005,
    "fragmentation_ref": 0.0005,
    "fragmentation_structural": 0.0005,
    "gang_stranded_gpu_hours": 0.05,  # shown to 0.1 GPU-h
    "footprint_wait_spearman": 0.005,  # shown to 0.01
}


@dataclass(frozen=True)
class Spread:
    """One metric, summarised across repeated runs."""

    name: str
    mean: float
    #: Sample standard deviation (n-1). ``None`` when n < 2, where it is
    #: undefined -- not 0.0, which would falsely assert reproducibility.
    sd: float | None
    n: int
    minimum: float
    maximum: float

    @property
    def cv(self) -> float | None:
        """Coefficient of variation, sd / |mean|. ``None`` when undefined."""
        if self.sd is None:
            return None
        if self.mean == 0.0:
            # A zero mean with any spread is maximally unstable; with no spread
            # it is perfectly stable. Neither is expressible as a ratio.
            return None if self.sd == 0.0 else math.inf
        return self.sd / abs(self.mean)

    def is_unstable(
        self, tolerance: float = DEFAULT_CV_TOLERANCE, floor: float = 0.0
    ) -> bool:
        """True when this apparatus cannot resolve this metric.

        A single run is unstable by construction: one sample cannot demonstrate
        reproducibility. Otherwise unstable iff ``sd > floor`` and
        ``cv > tolerance``.
        """
        if self.n < 2 or self.sd is None:
            return True
        if self.sd <= floor:
            return False
        cv = self.cv
        return cv is not None and cv > tolerance

    def format(self, *, scale: float = 1.0, unit: str = "", precision: int = 1) -> str:
        """``mean±sd`` in display units, or a bare mean when n == 1."""
        mean = self.mean * scale
        if self.sd is None:
            return f"{mean:.{precision}f}{unit}"
        return f"{mean:.{precision}f}±{self.sd * scale:.{precision}f}{unit}"


def _spread(name: str, values: list[float]) -> Spread:
    return Spread(
        name=name,
        mean=statistics.fmean(values),
        sd=statistics.stdev(values) if len(values) > 1 else None,
        n=len(values),
        minimum=min(values),
        maximum=max(values),
    )


@dataclass
class Aggregate:
    """Every repeat of one configuration, reduced to spreads.

    ``runs`` keeps the individual :class:`~k8slab.metrics.Metrics` so a results
    file can carry the raw samples alongside the summary. Discarding them would
    make the aggregate unfalsifiable.
    """

    config: str
    measured_on_cluster: bool
    runs: list[Metrics]
    spreads: dict[str, Spread] = field(default_factory=dict)
    tolerance: float = DEFAULT_CV_TOLERANCE

    @property
    def n(self) -> int:
        return len(self.runs)

    def get(self, name: str) -> Spread:
        if name not in self.spreads:
            raise KeyError(f"no spread for {name!r}; have {sorted(self.spreads)}")
        return self.spreads[name]

    def values(self, name: str) -> list[float]:
        """The per-run values of one scalar, skipping runs where it is undefined."""
        out: list[float] = []
        for m in self.runs:
            v = m.scalars().get(name)
            if v is not None:
                out.append(v)
        return out

    @property
    def harness(self) -> dict[str, Any]:
        """The harness settings every run shares (aggregate() enforces it)."""
        return dict(self.runs[0].harness)

    @property
    def scenario(self) -> bool:
        """Runtimes stretched by ASSUMED topology factors: a scenario row."""
        return self.runs[0].scenario

    @property
    def gated(self) -> bool:
        """Produced under a diagnostic gate: never in headline comparisons."""
        return self.runs[0].gated

    @property
    def src(self) -> str:
        """``cluster`` or ``model``, plus ``+topo`` for a scenario row and
        ``+gated`` for a diagnostic row. Rows with different labels are never
        pooled, and never ranked against each other."""
        label = "cluster" if self.measured_on_cluster else "model"
        if self.scenario:
            label += "+topo"
        if self.gated:
            label += "+gated"
        return label

    @property
    def trace_digests(self) -> list[str]:
        """Distinct traces among the runs. More than one means the spread
        includes workload variance, not just harness variance."""
        return sorted({m.trace_digest for m in self.runs})

    @property
    def fleets(self) -> list[str]:
        return sorted({m.fleet for m in self.runs})

    @property
    def unstable_metrics(self) -> list[str]:
        """Gated metrics this dataset cannot resolve, worst first."""
        bad: list[tuple[float, str]] = []
        for name, floor in GATED_METRICS.items():
            s = self.spreads.get(name)
            if s is None:
                continue  # undefined in every run (e.g. no admitted job)
            if s.is_unstable(self.tolerance, floor):
                bad.append((s.cv if s.cv is not None else 0.0, name))
        return [name for _, name in sorted(bad, key=lambda p: -p[0])]

    @property
    def unstable(self) -> bool:
        return bool(self.unstable_metrics)

    @property
    def deterministic(self) -> bool:
        """n >= 2 and every scalar identical across runs: repeats of a
        deterministic process, which carry no information about variance."""
        return self.n >= 2 and all(
            s.sd == 0.0 for s in self.spreads.values() if s.sd is not None
        )

    @property
    def verdict(self) -> str:
        """One word: ``unreplicated`` (n=1), ``unstable``, ``deterministic``
        (identical repeats) or ``stable``."""
        if self.n < 2:
            return "unreplicated"
        if self.unstable:
            return "unstable"
        if self.deterministic:
            return "deterministic"
        return "stable"

    def stability_note(self) -> str:
        """One line stating whether this configuration's numbers can be quoted."""
        mixed = ""
        if len(self.trace_digests) > 1:
            mixed = (
                f" Runs span {len(self.trace_digests)} distinct traces, so the spread "
                f"includes workload variance as well as harness variance."
            )
        if self.n < 2:
            return (
                f"{self.config}: n=1 — no spread measured. This is a smoke test, "
                f"not a measurement; do not quote or rank on it."
            )
        bad = self.unstable_metrics
        if bad:
            detail = ", ".join(_cv_text(self.spreads[name]) for name in bad)
            return (
                f"{self.config}: n={self.n}, UNSTABLE — {detail} "
                f"(tolerance CV {self.tolerance:.0%}). Differences in these metrics are "
                f"not resolvable by this harness, and the comparisons below never call "
                f"them resolvable.{mixed}"
            )
        if self.deterministic:
            where = (
                " On the reference model sd = 0 means the model is deterministic for "
                "this policy and seed set; it says nothing about whether a cluster run "
                "would be stable."
                if not self.measured_on_cluster else ""
            )
            return (
                f"{self.config}: n={self.n} bit-identical runs — a deterministic process "
                f"repeated, not evidence of a stable harness.{where}{mixed}"
            )
        return (
            f"{self.config}: n={self.n}, every gated metric within CV "
            f"{self.tolerance:.0%} (or below display resolution).{mixed}"
        )


def _cv_text(s: Spread) -> str:
    cv = s.cv
    if s.n < 2:
        return f"{s.name} defined in only {s.n} run"
    if cv is None:
        return s.name
    if math.isinf(cv):
        return f"{s.name} CV inf (mean 0)"
    return f"{s.name} CV {cv:.1%}"


def aggregate(
    config: str, runs: list[Metrics], tolerance: float = DEFAULT_CV_TOLERANCE
) -> Aggregate:
    """Reduce repeated runs of one configuration to per-metric spreads.

    Every scalar :meth:`Metrics.scalars` yields is summarised, not just the
    gated ones, so the results file records the full distribution. A scalar
    that is undefined (``None``) in some runs is summarised over the runs where
    it is defined, and its ``n`` says how many that was.
    """
    if not runs:
        raise ValueError(f"{config}: cannot aggregate zero runs")
    if len({m.config for m in runs}) != 1:
        raise ValueError(f"{config}: runs disagree on config id: {[m.config for m in runs]}")

    sources = {m.measured_on_cluster for m in runs}
    if len(sources) != 1:
        # Averaging a cluster run together with a reference-model run would
        # produce a number that describes neither. This is the specific
        # dishonesty report.py exists to prevent, so it is an error here.
        raise ValueError(
            f"{config}: cannot aggregate cluster and reference-model runs together"
        )
    if len({m.fleet for m in runs}) != 1:
        raise ValueError(f"{config}: cannot aggregate runs on different fleets")
    if len({tuple(sorted(m.penalty_factors.items())) for m in runs}) != 1:
        # placement_penalty_mean is only meaningful under one stated premise.
        raise ValueError(f"{config}: runs were scored under different penalty factors")
    if len({m.scenario for m in runs}) != 1:
        # A stretched runtime is an assumption, not a measurement; averaging
        # it with measured runs would launder the assumption into a number.
        raise ValueError(
            f"{config}: cannot aggregate scenario runs (topology_penalty=extend) "
            f"with measured runs"
        )
    harnesses = {json.dumps(m.harness, sort_keys=True) for m in runs}
    if len(harnesses) != 1:
        raise ValueError(
            f"{config}: runs were produced under different harness settings: "
            f"{sorted(harnesses)}"
        )

    per_run = [m.scalars() for m in runs]
    names: list[str] = []
    for scalars in per_run:
        names.extend(k for k in scalars if k not in names)
    spreads: dict[str, Spread] = {}
    for name in names:
        values = [v for s in per_run if (v := s.get(name)) is not None]
        if values:
            spreads[name] = _spread(name, values)

    return Aggregate(
        config=config,
        measured_on_cluster=runs[0].measured_on_cluster,
        runs=list(runs),
        spreads=spreads,
        tolerance=tolerance,
    )


def dataset_verdict(aggregates: list[Aggregate]) -> tuple[bool, list[str]]:
    """``(stable, notes)`` for a whole benchmark dataset.

    Returned rather than raised: an unstable dataset is still worth writing out,
    with the instability recorded next to it. Suppressing the numbers would lose
    the evidence that the harness needs work.
    """
    notes = [a.stability_note() for a in aggregates]
    return (not any(a.unstable for a in aggregates), notes)


# ---------------------------------------------------------------------------
# Pairwise comparison: seeded bootstrap of the difference in means
# ---------------------------------------------------------------------------

#: What every configuration is compared on: the headline quantities.
COMPARED_METRICS: tuple[str, ...] = (
    "makespan_hours",
    "utilization",
    "mean_wait",
    "fragmentation_structural",
)
DEFAULT_RESAMPLES = 4000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIDENCE = 0.95
#: Fewest runs per side before a difference may be called resolvable. Below it
#: the interval is still computed (from n >= 2) and shown, but never trusted.
#: The percentile bootstrap is anti-conservative at small n: under a null of
#: two identical normal distributions, its 95% interval excluded 0 in 33% of
#: trials at n=2, 20% at n=3, 13% at n=5 and 9% at n=10 (2000 trials each,
#: 4000 resamples; the command is in docs/metrics.md). 5 is the plan's repeat
#: count and the smallest n here with a false-alarm rate near one in eight.
MIN_RUNS_TO_RESOLVE = 5


@dataclass(frozen=True)
class Comparison:
    """``mean(config) - mean(baseline)`` for one metric, with its interval."""

    config: str
    baseline: str
    metric: str
    n_config: int
    n_baseline: int
    #: Point difference in means; ``None`` when either side has no data.
    diff: float | None
    #: Bootstrap percentile interval; ``None`` when not computed (n < 2).
    low: float | None
    high: float | None
    #: True only if the interval excludes 0 AND both sides have at least
    #: :data:`MIN_RUNS_TO_RESOLVE` runs AND neither side is unstable on this
    #: metric AND the two are comparable at all.
    resolvable: bool
    #: Why it is not resolvable, or a caveat when it is ("deterministic").
    note: str = ""
    #: True for the across-trace study's comparisons (:func:`compare_paired`):
    #: runs paired by trace, the bootstrap over per-trace differences.
    paired: bool = False


def bootstrap_diff_interval(
    a: Sequence[float],
    b: Sequence[float],
    *,
    rng: random.Random,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``mean(a) - mean(b)``.

    Independent resampling of each side with replacement: repeats of two
    configurations are separate runs, not paired observations. Each resample
    draws ``len(a)`` from ``a`` and ``len(b)`` from ``b``. The interval spans
    the resampled differences left after excluding :func:`tail_count` of them
    from EACH end: ``[d[k], d[R-1-k]]`` of the ``R`` sorted differences. Stdlib
    only; reproducible for a given ``rng`` state.
    """
    if not a or not b:
        raise ValueError("both sides need at least one value")
    na, nb = len(a), len(b)
    k = tail_count(resamples, confidence)
    diffs = sorted(
        math.fsum(rng.choices(a, k=na)) / na - math.fsum(rng.choices(b, k=nb)) / nb
        for _ in range(resamples)
    )
    return diffs[k], diffs[resamples - 1 - k]


def tail_count(resamples: int, confidence: float) -> int:
    """Resampled differences excluded from each tail of the interval.

    ``k = floor(R * (1 - confidence) / 2)``, the same count at both ends, with
    a 1e-9 guard against float error and capped so the interval never inverts.
    This replaced two separately rounded indices -- ``floor(alpha/2 * R)`` and
    ``ceil((1 - alpha/2) * R) - 1`` -- that were documented as nearest-rank
    quantiles but were not: at confidence 0.95 and R = 4000 the lower bound
    was the 101st smallest value (nearest-rank 2.5% is the 100th), and at 0.90
    ``1 - 0.9`` is 0.09999999999999998, so 199 values were cut below and 200
    above. At 0.95 with the default R both rules cut 100 from each tail, so no
    default interval moved.
    """
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    k = math.floor(resamples * (1.0 - confidence) / 2.0 + 1e-9)
    return min(k, (resamples - 1) // 2)


def compare(
    config: Aggregate,
    baseline: Aggregate,
    metric: str,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison:
    """Bootstrap one difference, refusing comparisons that mean nothing.

    The generator is seeded from ``(seed, config, baseline, metric)`` -- a
    string seed, which :class:`random.Random` hashes with SHA-512, so the
    result is independent of ``PYTHONHASHSEED`` and of which other comparisons
    ran. Adding a configuration to a dataset never moves another's interval.
    """
    a, b = config.values(metric), baseline.values(metric)

    def result(
        diff: float | None,
        low: float | None = None,
        high: float | None = None,
        resolvable: bool = False,
        note: str = "",
    ) -> Comparison:
        return Comparison(
            config=config.config, baseline=baseline.config, metric=metric,
            n_config=len(a), n_baseline=len(b), diff=diff, low=low, high=high,
            resolvable=resolvable, note=note,
        )

    refused = _refusal(config, baseline)
    if refused is not None:
        return result(None, note=refused)
    if set(config.trace_digests) != set(baseline.trace_digests) or config.fleets != baseline.fleets:
        return result(None, note="different trace or fleet: not the same experiment")
    if not a or not b:
        return result(None, note="undefined on one side")
    diff = statistics.fmean(a) - statistics.fmean(b)
    if len(a) < 2 or len(b) < 2:
        return result(diff, note="n<2: no spread, no interval")
    rng = random.Random(f"{seed}|{config.config}|{baseline.config}|{metric}")
    low, high = bootstrap_diff_interval(a, b, rng=rng, resamples=resamples,
                                        confidence=confidence)
    if min(len(a), len(b)) < MIN_RUNS_TO_RESOLVE:
        return result(diff, low, high,
                      note=f"n<{MIN_RUNS_TO_RESOLVE}: interval shown, not trusted")
    # One rule everywhere: a metric a side is unstable on is one this harness
    # "cannot resolve" (stability_note, the UNSTABLE banner). Calling a
    # difference on it resolvable here said the opposite in the same report.
    unstable = [
        agg.config for agg in (config, baseline) if metric in agg.unstable_metrics
    ]
    if unstable:
        return result(diff, low, high,
                      note=f"unstable on {' and '.join(unstable)}: interval shown, not trusted")
    excludes_zero = low > 0 or high < 0
    note = ""
    if excludes_zero and statistics.pstdev(a) == 0 and statistics.pstdev(b) == 0:
        note = "deterministic: both sides identical repeats"
    return result(diff, low, high, resolvable=excludes_zero, note=note)


def _refusal(config: Aggregate, baseline: Aggregate) -> str | None:
    """Why two aggregates may not be compared at all, or ``None``."""
    if config.measured_on_cluster != baseline.measured_on_cluster:
        return "cluster vs reference model: never comparable"
    if config.gated or baseline.gated:
        return "gated diagnostic row: excluded from comparisons"
    if config.scenario != baseline.scenario:
        return "scenario (+topo) vs measured: never ranked together"
    if config.harness != baseline.harness:
        differing = sorted(
            k for k in set(config.harness) | set(baseline.harness)
            if config.harness.get(k) != baseline.harness.get(k)
        )
        return f"different harness settings ({', '.join(differing)})"
    return None


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    rng: random.Random,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``mean(values)``, with the same tail
    rule as :func:`bootstrap_diff_interval` (:func:`tail_count`)."""
    if not values:
        raise ValueError("need at least one value")
    n = len(values)
    k = tail_count(resamples, confidence)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[k], means[resamples - 1 - k]


def _per_trace(agg: Aggregate, metric: str) -> dict[str, float] | None:
    """``{trace digest: value}`` for one metric; ``None`` if some trace has
    more than one run (then the runs are not one-per-trace and cannot pair)."""
    out: dict[str, float] = {}
    for m in agg.runs:
        if m.trace_digest in out:
            return None
        v = m.scalars().get(metric)
        if v is not None:
            out[m.trace_digest] = v
    return out


def compare_paired(
    config: Aggregate,
    baseline: Aggregate,
    metric: str,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison:
    """Compare two configurations across TRACES, paired by trace.

    For the across-trace-seed study (``--trace-seeds``): every configuration
    is replayed once on each of the same traces, so its runs pair with the
    baseline's by trace digest. The statistic is the mean of the per-trace
    differences, and the bootstrap resamples those differences, so the
    workload's own variation -- the same trace is easy or hard for both --
    cancels instead of swamping the interval as it does when the two sides
    are resampled independently.

    Resolvable when the interval excludes 0 over at least
    :data:`MIN_RUNS_TO_RESOLVE` traces. The harness-stability rule is NOT
    applied: the spread across traces is not harness noise, and whatever
    harness noise each run carries is already inside the per-trace
    differences the interval is built from.
    """
    def result(
        diff: float | None,
        n: tuple[int, int] = (0, 0),
        low: float | None = None,
        high: float | None = None,
        resolvable: bool = False,
        note: str = "",
    ) -> Comparison:
        return Comparison(
            config=config.config, baseline=baseline.config, metric=metric,
            n_config=n[0], n_baseline=n[1], diff=diff, low=low, high=high,
            resolvable=resolvable, note=note, paired=True,
        )

    refused = _refusal(config, baseline)
    if refused is not None:
        return result(None, note=refused)
    if config.fleets != baseline.fleets:
        return result(None, note="different fleet: not the same experiment")
    if set(config.trace_digests) != set(baseline.trace_digests):
        return result(None, note="different traces: nothing to pair")
    a, b = _per_trace(config, metric), _per_trace(baseline, metric)
    if a is None or b is None:
        return result(None, note="more than one run on a trace: not a paired design")
    # Pairs are the traces where the metric is defined on both sides.
    traces = sorted(t for t in a if t in b)
    n = (len(traces), len(traces))
    if not traces:
        return result(None, n, note="undefined on one side")
    diffs = [a[t] - b[t] for t in traces]
    diff = statistics.fmean(diffs)
    if len(diffs) < 2:
        return result(diff, n, note="one trace: no spread, no interval")
    rng = random.Random(f"{seed}|{config.config}|{baseline.config}|{metric}|paired")
    low, high = bootstrap_mean_interval(diffs, rng=rng, resamples=resamples,
                                        confidence=confidence)
    if len(diffs) < MIN_RUNS_TO_RESOLVE:
        return result(diff, n, low, high,
                      note=f"{len(diffs)} traces < {MIN_RUNS_TO_RESOLVE}: interval shown, "
                           f"not trusted")
    excludes_zero = low > 0 or high < 0
    note = "same difference on every trace" if len(set(diffs)) == 1 else ""
    return result(diff, n, low, high, resolvable=excludes_zero, note=note)


def baseline_ids(aggregates: Sequence[Aggregate]) -> list[str]:
    """The configurations everything is compared against: every degenerate
    baseline present, then K0 when present."""
    present = {a.config for a in aggregates}
    ids = [d for d in DEGENERATE_IDS if d in present]
    if "K0" in present:
        ids.append("K0")
    return ids


def compare_all(
    aggregates: Sequence[Aggregate],
    *,
    metrics: Sequence[str] = COMPARED_METRICS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    paired: bool = False,
) -> list[Comparison]:
    """Every configuration against every baseline (degenerate, and K0).

    A pair of baselines is compared once, in dataset order, not twice with the
    sign flipped. ``paired`` uses :func:`compare_paired` (the across-trace
    study) instead of the independent-repeats :func:`compare`.
    """
    method = compare_paired if paired else compare
    # Gated rows answer a diagnostic question; they are shown, never ranked.
    aggregates = [a for a in aggregates if not a.gated]
    by_id = {a.config: a for a in aggregates}
    bases = baseline_ids(aggregates)
    done: set[frozenset[str]] = set()
    out: list[Comparison] = []
    for agg in aggregates:
        for base_id in bases:
            if base_id == agg.config:
                continue
            pair = frozenset((agg.config, base_id))
            if pair in done:
                continue
            done.add(pair)
            for metric in metrics:
                out.append(
                    method(agg, by_id[base_id], metric, seed=seed,
                           resamples=resamples, confidence=confidence)
                )
    return out
