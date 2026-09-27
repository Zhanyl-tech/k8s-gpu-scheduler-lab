"""Render results.

Four rules. The first two are inherited from the sibling repos:

* Degenerate baselines appear in every table. A configuration that does not
  clearly beat ``D-random`` has not been shown to schedule.
* Every row states whether it was measured on a cluster or produced by the
  in-process reference model. A table that mixes the two without saying so is
  the specific dishonesty this module exists to prevent.

The next two exist because one run of this harness is not a measurement
(docs/limitations.md, "Run-to-run variance"):

* The unit of reporting is a configuration's repeated runs, shown as
  ``mean±sd`` with ``n``. ``n=1`` is printed as a bare value and is never
  called stable; if any configuration has an unstable gated metric, results.md
  opens with an UNSTABLE banner.
* A difference between two configurations is called *resolvable* only when a
  seeded bootstrap interval excludes zero and neither side is unstable on that
  metric (:mod:`k8slab.stats`). Everything else is a difference this apparatus
  cannot tell from noise.

Phase 2 adds labels to the ``src`` column: ``+topo`` for a scenario row whose
runtimes were stretched by assumed topology factors, ``+gated`` for a
diagnostic row (admission gate, serialised scheduler) that is shown but never
ranked. results.json (schema 3) also records how the runs were produced
(``harness``), the time-scaling record and its quantisation residual
(``timescale``) and, when requested, a separate across-trace-seed study.

The input is simply every run's :class:`~k8slab.metrics.Metrics`, in any order;
runs are grouped by config id, in first-seen order. A caller that repeats
configurations just passes more runs.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import FOOTPRINT_BUCKETS, Metrics
from .stats import (
    COMPARED_METRICS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_CV_TOLERANCE,
    DEFAULT_RESAMPLES,
    GATED_METRICS,
    MIN_RUNS_TO_RESOLVE,
    Aggregate,
    Comparison,
    Spread,
    aggregate,
    compare_all,
    dataset_verdict,
)
from .topology import DEFAULT_FACTORS, TIERS

#: A configuration whose utilization beats D-random while its large jobs wait
#: more than this many times as long as its 1-GPU jobs -- and more than
#: D-random's large jobs do, relative to D-random's 1-GPU jobs -- is flagged as
#: the "inflates utilization by starving large jobs" pattern. 2.0 is a judgment
#: call, NOT a calibrated value: it reads "large jobs wait more than twice as
#: long as single-GPU jobs". Some excess is expected from any scheduler, since
#: a large job needs a whole node to drain; how much is not something this lab
#: has measured. For scale only, docs/metrics.md lists the reference model's
#: ratios for the degenerate policies. The ratio itself is always reported, so
#: a reader can apply their own line; ``build_report`` takes another threshold.
STARVATION_FLAG_THRESHOLD = 2.0

#: 3: Phase 2 execution layer -- harness/timescale/trace_seed_study keys,
#: ``src`` labels with ``+topo``/``+gated``, and the execution-layer metrics.
JSON_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class Column:
    """One numeric column: which scalar, and how it is displayed."""

    header: str
    metric: str
    scale: float = 1.0
    precision: int = 1

    def cell(self, agg: Aggregate) -> str:
        s = agg.spreads.get(self.metric)
        if s is None:
            return "—"
        return s.format(scale=self.scale, precision=self.precision)

    @property
    def resolution(self) -> float:
        """Half the last displayed digit, in the metric's own unit."""
        return 0.5 * 10.0 ** (-self.precision) / self.scale


#: The headline table. Units live in the headers; cells are ``mean±sd``.
HEADLINE: tuple[Column, ...] = (
    Column("makespan h", "makespan_hours"),
    Column("util %", "utilization", 100.0),
    Column("GPU-h used", "gpu_hours_used"),
    Column("mean wait m", "mean_wait", 1 / 60, 0),
    Column("p95 wait m", "p95_wait", 1 / 60, 0),
    Column("frag A %", "fragmentation_rate", 100.0),
    Column("frag B %", "fragmentation_ref", 100.0),
    Column("frag C %", "fragmentation_structural", 100.0),
    Column("gang strand GPU-h", "gang_stranded_gpu_hours"),
    Column("size-wait ρ", "footprint_wait_spearman", 1.0, 2),
    Column("starve ×", "large_job_starvation_ratio", 1.0, 2),
    Column("place pen.*", "placement_penalty_mean", 1.0, 2),
    Column("fair", "fairness_ratio", 1.0, 2),
)

#: How a compared metric is displayed in the differences table.
_DIFF_DISPLAY: dict[str, tuple[str, float, int]] = {
    "makespan_hours": ("makespan h", 1.0, 2),
    "utilization": ("util pp", 100.0, 2),
    "mean_wait": ("mean wait m", 1 / 60, 1),
    "fragmentation_structural": ("frag C pp", 100.0, 2),
}


@dataclass
class Report:
    """Everything results.md and results.json are rendered from."""

    aggregates: list[Aggregate]
    comparisons: list[Comparison]
    flags: list[str]
    stable: bool
    notes: list[str]
    tolerance: float
    resamples: int
    seed: int
    confidence: float
    starvation_threshold: float
    #: How the runs were produced: harness settings, repeats, seeds.
    harness: dict[str, Any] | None = None
    #: :meth:`k8slab.timescale.TimeScale.summary` for the run's speedup.
    timescale: dict[str, Any] | None = None
    #: The optional across-trace-seed study (``--trace-seeds``).
    trace_seeds: list[int] | None = None
    trace_seed_aggregates: list[Aggregate] | None = None
    trace_seed_comparisons: list[Comparison] | None = None
    #: Over the starvation threshold with a utilization lead, but not flagged
    #: because D-random starves large jobs at least as much
    #: (:func:`starvation_exemptions`).
    flag_exemptions: list[str] = field(default_factory=list)


def aggregate_runs(
    runs: Sequence[Metrics], tolerance: float = DEFAULT_CV_TOLERANCE
) -> list[Aggregate]:
    """Group runs by config id (first-seen order) and aggregate each group."""
    groups: dict[str, list[Metrics]] = {}
    for m in runs:
        groups.setdefault(m.config, []).append(m)
    return [aggregate(cfg, ms, tolerance) for cfg, ms in groups.items()]


def build_report(
    runs: Sequence[Metrics],
    *,
    tolerance: float = DEFAULT_CV_TOLERANCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    starvation_threshold: float = STARVATION_FLAG_THRESHOLD,
    harness: Mapping[str, Any] | None = None,
    timescale: Mapping[str, Any] | None = None,
    trace_seed_runs: Sequence[Metrics] = (),
    trace_seeds: Sequence[int] = (),
) -> Report:
    aggs = aggregate_runs(runs, tolerance)
    comparisons = compare_all(aggs, seed=seed, resamples=resamples, confidence=confidence)
    stable, notes = dataset_verdict(aggs)
    seed_aggs: list[Aggregate] | None = None
    seed_cmps: list[Comparison] | None = None
    if trace_seed_runs:
        seed_aggs = aggregate_runs(trace_seed_runs, tolerance)
        # One run per configuration per trace, the same traces for every
        # configuration: compared paired by trace, not as independent repeats.
        seed_cmps = compare_all(seed_aggs, seed=seed, resamples=resamples,
                                confidence=confidence, paired=True)
    return Report(
        aggregates=aggs,
        comparisons=comparisons,
        flags=starvation_flags(aggs, comparisons, starvation_threshold),
        stable=stable,
        notes=notes,
        tolerance=tolerance,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        starvation_threshold=starvation_threshold,
        harness=dict(harness) if harness is not None else None,
        timescale=dict(timescale) if timescale is not None else None,
        trace_seeds=list(trace_seeds) if trace_seed_runs else None,
        trace_seed_aggregates=seed_aggs,
        trace_seed_comparisons=seed_cmps,
        flag_exemptions=starvation_exemptions(aggs, starvation_threshold),
    )


def starvation_flags(
    aggs: Sequence[Aggregate],
    comparisons: Sequence[Comparison],
    threshold: float = STARVATION_FLAG_THRESHOLD,
) -> list[str]:
    """Utilization bought by starving large jobs.

    Raised when a configuration's mean utilization beats D-random's, its mean
    large-job starvation ratio exceeds ``threshold``, AND that ratio exceeds
    D-random's own. The pattern is relative: a lead cannot have been bought
    from D-random by starving large jobs if D-random starves them more. (It
    used to fire on the absolute threshold alone and assert "This is the ...
    pattern": the reference model flagged D-largest, 7.20×, against a D-random
    at 19.14×, on a lead that was not resolvable.) Configurations over the
    threshold that this rule leaves out are listed by
    :func:`starvation_exemptions`, so nothing is hidden.

    The wording follows the evidence: "matches the pattern" only when the
    utilization lead is itself resolvable; otherwise the pattern "is not
    established" -- a flag on a lead that is noise is a weaker statement.
    """
    flags: list[str] = []
    for agg, _, lead, starve, rnd_starve in _starvation_candidates(aggs, threshold):
        if rnd_starve is not None and starve <= rnd_starve:
            continue
        resolvable = _lead_resolvable(agg, comparisons)
        baseline = (
            f"D-random's {rnd_starve:.2f}×" if rnd_starve is not None
            else "D-random's, which is undefined"
        )
        text = (
            f"**{agg.config}**: utilization lead over D-random of {lead * 100:+.1f} pp "
            f"({'resolvable' if resolvable else 'NOT resolvable'}) with a large-job "
            f"starvation ratio of {starve:.2f}× against {baseline} (threshold "
            f"{threshold:.1f}×): jobs needing at least {agg.runs[0].large_job_gpus} GPUs "
            f"waited that many times as long as 1-GPU jobs. "
        )
        if resolvable:
            text += (
                "This matches the 'inflates utilization by starving large jobs' "
                "pattern; read the utilization lead with it."
            )
        else:
            text += (
                "It coincides with the 'inflates utilization by starving large jobs' "
                "pattern, but the lead itself is not resolvable, so the pattern is not "
                "established."
            )
        flags.append(text)
    return flags


def starvation_exemptions(
    aggs: Sequence[Aggregate], threshold: float = STARVATION_FLAG_THRESHOLD
) -> list[str]:
    """Configurations that lead D-random on utilization with a starvation ratio
    above ``threshold`` but are NOT flagged, because D-random's own ratio is at
    least as high. Printed under the flags so the rule hides nothing."""
    out: list[str] = []
    for agg, _, lead, starve, rnd_starve in _starvation_candidates(aggs, threshold):
        if rnd_starve is not None and starve <= rnd_starve:
            out.append(
                f"{agg.config} leads D-random on utilization by {lead * 100:+.1f} pp with a "
                f"starvation ratio of {starve:.2f}× (above the {threshold:.1f}× threshold), "
                f"but D-random's own ratio is {rnd_starve:.2f}×: the lead cannot have been "
                f"bought by starving large jobs relative to D-random. Not flagged."
            )
    return out


def _starvation_candidates(
    aggs: Sequence[Aggregate], threshold: float
) -> list[tuple[Aggregate, Aggregate, float, float, float | None]]:
    """``(config, D-random, utilization lead, ratio, D-random's ratio)`` for
    every like-for-like configuration that leads D-random with a ratio above
    ``threshold``."""
    by_id = {a.config: a for a in aggs}
    rnd = by_id.get("D-random")
    if rnd is None:
        return []
    out: list[tuple[Aggregate, Aggregate, float, float, float | None]] = []
    rnd_starve = rnd.spreads.get("large_job_starvation_ratio")
    for agg in aggs:
        if agg is rnd or agg.src != rnd.src or agg.harness != rnd.harness:
            continue
        u, u_rnd = agg.spreads.get("utilization"), rnd.spreads.get("utilization")
        starve = agg.spreads.get("large_job_starvation_ratio")
        if u is None or u_rnd is None or starve is None:
            continue
        lead = u.mean - u_rnd.mean
        if lead <= 0 or starve.mean <= threshold:
            continue
        out.append((agg, rnd, lead, starve.mean,
                    rnd_starve.mean if rnd_starve is not None else None))
    return out


def _lead_resolvable(agg: Aggregate, comparisons: Sequence[Comparison]) -> bool:
    cmp = next(
        (
            c for c in comparisons
            if c.metric == "utilization" and {c.config, c.baseline} == {agg.config, "D-random"}
        ),
        None,
    )
    return cmp is not None and cmp.resolvable


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |")
    return "\n".join(out)


def _src(agg: Aggregate) -> str:
    return agg.src


def headline_table(aggs: Sequence[Aggregate], *, across_traces: bool = False) -> str:
    """The headline table. ``across_traces`` is the trace-seed study's form:
    a ``traces`` column instead of the harness-stability ``verdict``, which
    does not apply to a spread across different workloads."""
    if across_traces:
        headers = ["config", "src", "n", "traces", *(c.header for c in HEADLINE)]
        rows = [
            [a.config, _src(a), str(a.n), str(len(a.trace_digests)),
             *(c.cell(a) for c in HEADLINE)]
            for a in aggs
        ]
        return _table(headers, rows)
    headers = ["config", "src", "n", "verdict", *(c.header for c in HEADLINE)]
    rows = [
        [a.config, _src(a), str(a.n), a.verdict, *(c.cell(a) for c in HEADLINE)] for a in aggs
    ]
    return _table(headers, rows)


def markdown_table(
    runs: Sequence[Metrics], tolerance: float = DEFAULT_CV_TOLERANCE
) -> str:
    """The headline table alone, for a terminal. ``runs`` may repeat configs.

    ``tolerance`` must be the one results.md was written with: the verdict
    column depends on it, and the CLI once printed a 5%-tolerance verdict here
    next to a results.md written under ``--sigma-tolerance``.
    """
    return headline_table(aggregate_runs(runs, tolerance))


def text_table(runs: Sequence[Metrics], tolerance: float = DEFAULT_CV_TOLERANCE) -> str:
    return markdown_table(runs, tolerance)


def _fmt(value: float | None, scale: float, precision: int, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed else ""
    return f"{value * scale:{sign}.{precision}f}"


def _differences_table(comparisons: Sequence[Comparison]) -> str:
    rows: list[list[str]] = []
    for c in comparisons:
        label, scale, prec = _DIFF_DISPLAY.get(c.metric, (c.metric, 1.0, 3))
        ci = (
            "—" if c.low is None or c.high is None
            else f"[{_fmt(c.low, scale, prec, True)}, {_fmt(c.high, scale, prec, True)}]"
        )
        rows.append([
            c.config, c.baseline, label, _fmt(c.diff, scale, prec, True), ci,
            f"{c.n_config}/{c.n_baseline}", "yes" if c.resolvable else "no", c.note,
        ])
    return _table(
        ["config", "vs", "metric", "diff", "95% CI", "n", "resolvable", "note"], rows
    )


def _spread_cell(agg: Aggregate, name: str, scale: float, precision: int) -> str:
    s: Spread | None = agg.spreads.get(name)
    return "—" if s is None else s.format(scale=scale, precision=precision)


#: Where results.md points for definitions when written somewhere its own
#: relative link cannot be computed (a directory outside this repository).
DOCS_REFERENCE = "docs/metrics.md in the k8s-gpu-scheduler-lab repository"


def docs_link(directory: str | Path) -> str | None:
    """Relative link from ``directory`` to this repository's docs/metrics.md.

    ``None`` when the directory is outside the repository (or the package is
    not running from a checkout): there is no link that would resolve, and a
    hard-coded ``../docs/metrics.md`` was already broken for the Makefile's
    own default, ``results/phase2``.
    """
    repo = Path(__file__).resolve().parents[2]
    docs = repo / "docs" / "metrics.md"
    target = Path(directory).resolve()
    if not docs.is_file() or not target.is_relative_to(repo):
        return None
    return Path(os.path.relpath(docs, target)).as_posix()


def render_markdown(report: Report, docs: str | None = "../docs/metrics.md") -> str:
    """results.md. ``docs`` is the link to docs/metrics.md relative to where
    the file will be written (:func:`docs_link`); ``None`` names it in plain
    text instead."""
    aggs = report.aggregates
    definitions = f"[docs/metrics.md]({docs})" if docs is not None else DOCS_REFERENCE
    out: list[str] = ["# Results", ""]

    unstable = [a for a in aggs if a.unstable]
    if unstable:
        names = ", ".join(f"`{a.config}`" for a in unstable)
        out += [
            f"> **UNSTABLE.** {len(unstable)} of {len(aggs)} configurations ({names}) "
            f"have a gated metric this harness cannot resolve — a coefficient of "
            f"variation above {report.tolerance:.0%} across repeats, or a single run "
            f"(n=1 is never stable). Do not rank configurations on the affected "
            f"metrics. See *Stability* below.",
            "",
        ]

    out += [
        "Cells are `mean±sd` over `n` repeated runs (a bare value when n=1). "
        "`src=cluster` rows came from a real control plane, `src=model` rows from "
        "the in-process reference model; never compare across the two. A `+topo` "
        "suffix marks a SCENARIO row whose runtimes were stretched by assumed "
        "topology factors; `+gated` marks a diagnostic row (admission gate or "
        "serialised scheduler) that is shown but never ranked. Definitions: "
        f"{definitions}.",
        "",
    ]
    out += _produced_section(report)
    out += [
        "## Headline",
        "",
        headline_table(aggs),
        "",
        "\\* `place pen.` is the GPU-weighted mean of ASSUMED topology penalty factors "
        f"({_factors_text(aggs)}) — scenario parameters, not measurements; the lab "
        "sends no NCCL traffic. `frag C` is Definition C at the node level. "
        "`gang DL` (gang_deadlock_rate) is deprecated and kept in results.json only.",
        "",
        "## Stability",
        "",
    ]
    out += [f"- {note}" for note in report.notes]
    out.append("")
    if any(a.deterministic and not a.measured_on_cluster for a in aggs):
        out += [
            "_`deterministic` on a model row means the reference model produced "
            "bit-identical numbers on every repeat (sd = 0): the policy ignores the "
            "harness seed and, with a zero startup delay, nothing else in the model "
            "draws from it (a non-zero `--startup-delay` is drawn per harness seed, "
            "so every policy's repeats then vary). It is a property of the model, not "
            "evidence that a cluster run would be stable._",
            "",
        ]
    gated = [a.config for a in aggs if a.gated]
    if gated:
        out += [
            f"_Gated diagnostic rows ({', '.join(gated)}) changed what is measured — "
            f"the admission gate and a serialised scheduler remove the concurrency "
            f"that queue-pressure throughput is a property of — so they are excluded "
            f"from the comparisons below (docs/limitations.md)._",
            "",
        ]

    out += [
        "## Differences that survive repetition",
        "",
        f"Seeded bootstrap ({report.resamples} resamples, seed {report.seed}) "
        f"{report.confidence:.0%} interval of mean(config) − mean(baseline), each "
        f"configuration against every degenerate baseline and K0. **Resolvable** only "
        f"if the interval excludes 0, both sides have n ≥ {MIN_RUNS_TO_RESOLVE}, and "
        f"neither side is unstable on that metric (see *Stability*). "
        f"Even then this is a screen, not a test at 5%: under a null of no difference "
        f"the method false-alarms about 13% of the time at n=5 (docs/metrics.md).",
        "",
    ]
    if report.comparisons:
        out.append(_differences_table(report.comparisons))
    else:
        out.append("_No baseline present in this dataset; nothing to compare against._")
    out.append("")

    out += ["## Flags", ""]
    if "D-random" not in {a.config for a in aggs}:
        out.append("_D-random is not in this dataset, so the starvation flag was not evaluated._")
    elif report.flags:
        out += [f"- {f}" for f in report.flags]
    else:
        out.append(
            f"_None: no configuration beat D-random on utilization with a large-job "
            f"starvation ratio above {report.starvation_threshold:.1f}× and above "
            f"D-random's own._"
        )
    if "D-random" in {a.config for a in aggs} and report.flag_exemptions:
        out += ["", *(f"- _{e}_" for e in report.flag_exemptions)]
    out.append("")

    out += [
        "## Fragmentation, Definition C by level",
        "",
        "Share of free GPU-time inside a carved domain (any GPU allocated). "
        "Node is the headline; rack and switch depend on the declared topology.",
        "",
        _table(
            ["config", "src", "n", "node %", "rack %", "switch %"],
            [
                [a.config, _src(a), str(a.n),
                 _spread_cell(a, "fragmentation_structural", 100.0, 1),
                 _spread_cell(a, "fragmentation_structural_rack", 100.0, 1),
                 _spread_cell(a, "fragmentation_structural_switch", 100.0, 1)]
                for a in aggs
            ],
        ),
        "",
        "## Gang assembly",
        "",
        "Stranded = GPU-time held by gang members before every member was running. "
        "Assembly delay = last member Running − first member bound.",
        "",
        _table(
            ["config", "n", "gangs", "assembled", "stranded GPU-h", "% of delivered",
             "assembly p50 m", "assembly p95 m"],
            [
                [a.config, str(a.n),
                 _spread_cell(a, "gang_jobs", 1.0, 0),
                 _spread_cell(a, "gang_assembled", 1.0, 0),
                 _spread_cell(a, "gang_stranded_gpu_hours", 1.0, 1),
                 _spread_cell(a, "gang_stranded_share", 100.0, 1),
                 _spread_cell(a, "gang_assembly_p50", 1 / 60, 1),
                 _spread_cell(a, "gang_assembly_p95", 1 / 60, 1)]
                for a in aggs
            ],
        ),
        "",
        "## Admission delay by job footprint",
        "",
        "Mean wait in minutes (to the last pod; for an evicted job, only the time it "
        "was pending) by total GPUs per job, over admitted "
        "jobs. `size-wait ρ` in the headline is the Spearman correlation of footprint "
        "and wait; `starve ×` is mean wait of jobs needing at least the largest node "
        "over mean wait of 1-GPU jobs.",
        "",
        _table(
            ["config", "n", *(f"{b} GPU" for b, _, _ in FOOTPRINT_BUCKETS)],
            [
                [a.config, str(a.n),
                 *(_spread_cell(a, f"wait_by_footprint.{b}.mean", 1 / 60, 0)
                   for b, _, _ in FOOTPRINT_BUCKETS)]
                for a in aggs
            ],
        ),
        "",
    ]
    unadmitted = _unadmitted_note(aggs)
    if unadmitted:
        out += [unadmitted, ""]

    out += [
        "## Placement",
        "",
        "Widest domain spanned by each fully placed multi-pod job — a measurement of "
        "where the scheduler put pods. The penalty column applies ASSUMED factors.",
        "",
        _table(
            ["config", "n", "multi-pod jobs", *(f"{t} %" for t in TIERS), "penalty*"],
            [
                [a.config, str(a.n),
                 _spread_cell(a, "placement_multi_pod_jobs", 1.0, 0),
                 *(_spread_cell(a, f"placement_tier_share.{t}", 100.0, 1) for t in TIERS),
                 _spread_cell(a, "placement_penalty_mean", 1.0, 2)]
                for a in aggs
            ],
        ),
        "",
    ]
    undeclared = [a.config for a in aggs if not all(m.topology_declared for m in a.runs)]
    if undeclared:
        out += [
            f"_Topology was not declared in the fleet file for {', '.join(undeclared)}; "
            f"rack/switch tiers use the flat defaults (one rack per node class, one "
            f"switch)._",
            "",
        ]
    out += _execution_section(aggs)
    notes = [f"{a.config}: {n}" for a in aggs for m in a.runs for n in m.notes]
    if notes:
        out += ["## Execution notes", "", *(f"- {n}" for n in dict.fromkeys(notes)), ""]
    out += _trace_seed_section(report)
    return "\n".join(out)


def _produced_section(report: Report) -> list[str]:
    """How the runs were produced: settings, seeds, and the time-scaling residual."""
    if report.harness is None and report.timescale is None:
        return []
    out = ["## How these runs were produced", ""]
    if report.harness is not None:
        for key, value in report.harness.items():
            if key == "argv" and isinstance(value, list):
                out.append(f"- command: `{' '.join(str(v) for v in value)}`")
            elif key == "startup_delay_ms" and isinstance(value, list):
                out.append(f"- `{key}`: {_plain(value[0])}:{_plain(value[1])} (real ms)")
            else:
                out.append(f"- `{key}`: {_plain(value)}")
    ts = report.timescale
    if ts is not None:
        out.append(_timescale_line(ts))
    out.append("")
    return out


#: ``timescale["status"]`` values (cli._bench). ``recorded``: a cluster run
#: whose kube-scheduler configuration is the one ``make up`` wrote and recorded
#: in cluster-state.json -- a record of INTENT, written before ``kind create
#: cluster`` runs. Nothing reads the running scheduler's flags back, so a
#: results file must not say the values were applied or experienced. (It said
#: ``applied`` until that was pointed out.) A record without a status predates
#: the field and is read the same, conservative way.
TIMESCALE_RECORDED = "recorded"
TIMESCALE_MODEL = "model"
TIMESCALE_NOT_APPLIED = "not applied"
TIMESCALE_UNKNOWN = "unknown"


def _timescale_line(ts: Mapping[str, Any]) -> str:
    """One line saying what time scaling was in force -- and only that.

    A results file must not state a scheduler configuration nobody verified
    (cluster without its ``make up`` record) or one that did not exist (the
    reference model with no queue model), and states a recorded one only as
    configured, never as applied.
    """
    status = ts.get("status", TIMESCALE_RECORDED)
    speedup = _plain(ts.get("speedup"))
    if status == TIMESCALE_UNKNOWN:
        return (
            f"- time scaling at speedup {speedup}: kube-scheduler configuration "
            f"**UNKNOWN** — {ts.get('note', 'no record of how the cluster was brought up')}. "
            f"The scaled values `make up` would have set are recorded as "
            f"`timescale.intended` in results.json; nothing verified they were in force."
        )
    if status == TIMESCALE_NOT_APPLIED:
        return f"- time scaling: not applied — {ts.get('note', '')}".rstrip(" —")
    who = (
        "as experienced by the in-process kube queue model (reference model: no "
        "kube-scheduler ran)"
        if status == TIMESCALE_MODEL else
        "as `make up` configured kube-scheduler (from its record, cluster-state.json; "
        "the running scheduler's flags were not read back)"
    )
    exp = ts.get("experienced_sim_seconds", {})
    line = (
        f"- time scaling at speedup {speedup}, {who}: backoff "
        f"configured at {_plain(exp.get('initial_backoff'))}–"
        f"{_plain(exp.get('max_backoff'))} simulated s (uncompressed: 1–10 s; "
        f"integer-quantisation residual "
        f"{_plain(ts.get('backoff_residual_error_sim_seconds'))} s on the ceiling, "
        f"{_plain(ts.get('initial_backoff_residual_sim_seconds'))} s on the initial "
        f"value); unscalable flush timers every "
        f"{_plain(exp.get('backoff_flush_interval'))} s (backoffQ) and "
        f"{_plain(exp.get('unschedulable_flush_interval'))} s (unschedulable pool)."
    )
    ranges = ts.get("experienced_range_sim_seconds")
    worst = ts.get("worst_case_residual_sim_seconds")
    if isinstance(ranges, Mapping) and isinstance(worst, Mapping):
        i = ranges.get("initial_backoff", [None, None])
        b = ranges.get("max_backoff", [None, None])
        u = ranges.get("max_in_unschedulable", [None, None])
        line += (
            f" The flush ticks gate the configured values: a pod an event releases "
            f"while it is still backing off waits for the next backoffQ tick, so the "
            f"initial backoff is experienced as [{_plain(i[0])}, {_plain(i[1])}) s and "
            f"the ceiling as [{_plain(b[0])}, {_plain(b[1])}) s (worst case "
            f"+{_plain(worst.get('initial_backoff'))} s and "
            f"+{_plain(worst.get('max_backoff'))} s against 1 s and 10 s); a pod no "
            f"event releases leaves the unschedulable pool after ({_plain(u[0])}, "
            f"{_plain(u[1])}] s (worst case +{_plain(worst.get('max_in_unschedulable'))} s "
            f"against 300 s; uncompressed at most +30 s)."
        )
    return line


def _plain(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(v) for v in value) if value else "—"
    return str(value)


def _execution_section(aggs: Sequence[Aggregate]) -> list[str]:
    return [
        "## Execution layer",
        "",
        "Startup overhead = GPU-time held between bind and Running. Preemption "
        "columns are non-zero only for configurations that evict (D-preempt, or a "
        "control plane that preempted). `topo ext.` is delivered GPU-time that exists "
        "only because runtimes were stretched by ASSUMED topology factors "
        "(`--topology-penalty extend`); delivered = demanded + topo ext. when every "
        "job completes and nothing was checkpointed.",
        "",
        _table(
            ["config", "src", "n", "demanded GPU-h", "delivered GPU-h", "startup GPU-h",
             "preemptions", "lost GPU-h", "grace-locked GPU-h", "topo ext. GPU-h"],
            [
                [a.config, _src(a), str(a.n),
                 _spread_cell(a, "gpu_hours_demanded", 1.0, 1),
                 _spread_cell(a, "gpu_hours_used", 1.0, 1),
                 _spread_cell(a, "startup_overhead_gpu_hours", 1.0, 2),
                 _spread_cell(a, "preemptions", 1.0, 0),
                 _spread_cell(a, "preempted_gpu_hours_lost", 1.0, 1),
                 _spread_cell(a, "grace_locked_gpu_hours", 1.0, 1),
                 _spread_cell(a, "topology_extension_gpu_hours", 1.0, 1)]
                for a in aggs
            ],
        ),
        "",
    ]


#: What the across-trace-seed study's spread is -- and is not. It used to be
#: called "WORKLOAD variance, not harness variance" while each configuration's
#: stability note under it said the opposite ("not resolvable by this
#: harness"). With one run per configuration per trace neither is right.
TRACE_SEED_SPREAD = (
    "With one run per configuration per trace, the spread across traces MIXES "
    "workload variance (the traces differ), harness variance (each run on a "
    "cluster carries the run-to-run noise docs/limitations.md describes) and, "
    "for D-random, the policy's own random choices, which differ on every "
    "trace. This design cannot separate them, so no stability verdict is given "
    "here."
)


def trace_seed_note(agg: Aggregate) -> str:
    """The trace-seed study's per-configuration note (in place of
    :meth:`~k8slab.stats.Aggregate.stability_note`, whose verdict is about
    repeats of one trace)."""
    traces = len(agg.trace_digests)
    return (
        f"{agg.config}: {agg.n} run(s) on {traces} distinct trace(s). The sd is "
        f"across traces -- workload, harness and policy randomness together -- and "
        f"is not a stability verdict."
    )


def _trace_seed_section(report: Report) -> list[str]:
    aggs = report.trace_seed_aggregates
    if not aggs:
        return []
    seeds = ", ".join(str(s) for s in report.trace_seeds or [])
    out = [
        "## Across trace seeds",
        "",
        f"A separate study, not pooled with the repeats above: each configuration "
        f"replayed once on each of the traces generated with seeds {seeds} (harness "
        f"seed of repeat 1). {TRACE_SEED_SPREAD}",
        "",
        headline_table(aggs, across_traces=True),
        "",
    ]
    out += [f"- {trace_seed_note(a)}" for a in aggs]
    out.append("")
    if report.trace_seed_comparisons:
        out += [
            f"Differences are PAIRED by trace: every configuration replayed the same "
            f"traces, so the statistic is the mean per-trace difference and the "
            f"bootstrap ({report.resamples} resamples) resamples those differences. "
            f"Resolvable only if the interval excludes 0 over at least "
            f"{MIN_RUNS_TO_RESOLVE} traces. It says whether one configuration beat "
            f"another across these workloads, under this harness; it says nothing "
            f"about whether the harness is stable.",
            "",
            _differences_table(report.trace_seed_comparisons),
            "",
        ]
    return out


def _factors_text(aggs: Sequence[Aggregate]) -> str:
    """The factor set(s) the runs were actually scored under."""
    sets: dict[tuple[tuple[str, float], ...], list[str]] = {}
    for a in aggs:
        key = tuple(a.runs[0].penalty_factors.items())
        sets.setdefault(key, []).append(a.config)
    texts = []
    for key, cfgs in sets.items():
        f = dict(key)
        text = (
            f"node+NVLink {f['node_nvlink']}, node w/o NVLink {f['node_no_nvlink']}, "
            f"rack {f['rack']}, switch {f['switch']}, cross-switch {f['cross_switch']}"
        )
        if key != tuple(asdict(DEFAULT_FACTORS).items()):
            text += " — NOT the defaults"
        texts.append(text if len(sets) == 1 else f"{', '.join(cfgs)}: {text}")
    return "; ".join(texts)


def _unadmitted_note(aggs: Sequence[Aggregate]) -> str:
    parts: list[str] = []
    for a in aggs:
        missing = []
        for b, _, _ in FOOTPRINT_BUCKETS:
            jobs = a.spreads.get(f"wait_by_footprint.{b}.jobs")
            admitted = a.spreads.get(f"wait_by_footprint.{b}.admitted")
            if jobs is not None and admitted is not None and admitted.mean < jobs.mean:
                missing.append(f"{b}: {jobs.mean - admitted.mean:.0f}")
        if missing:
            parts.append(f"{a.config} ({', '.join(missing)})")
    if not parts:
        return ""
    return (
        "_Never admitted, mean per run, by bucket — excluded from the waits above, "
        "which therefore flatter these configurations: " + "; ".join(parts) + "._"
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _finite(value: Any) -> Any:
    """JSON has no inf/nan; record them as null rather than emit invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def to_json(report: Report) -> dict[str, Any]:
    """Aggregates next to every raw run: the summary stays falsifiable."""
    configs: list[dict[str, Any]] = []
    for a in report.aggregates:
        configs.append({
            "config": a.config,
            "src": _src(a),
            "scenario": a.scenario,
            "gated": a.gated,
            "harness": a.harness,
            "n": a.n,
            "verdict": a.verdict,
            "note": a.stability_note(),
            "unstable_metrics": a.unstable_metrics,
            "trace_digests": a.trace_digests,
            "aggregate": {
                name: {
                    "mean": s.mean,
                    "sd": s.sd,
                    "n": s.n,
                    "min": s.minimum,
                    "max": s.maximum,
                    "cv": s.cv,
                    "gated": name in GATED_METRICS,
                    "unstable": (
                        s.is_unstable(a.tolerance, GATED_METRICS[name])
                        if name in GATED_METRICS else None
                    ),
                }
                for name, s in a.spreads.items()
            },
            "runs": [asdict(m) for m in a.runs],
        })
    doc: dict[str, Any] = {
        "schema": JSON_SCHEMA_VERSION,
        "stable": report.stable,
        "cv_tolerance": report.tolerance,
        "gated_metrics": dict(GATED_METRICS),
        "bootstrap": {
            "method": "percentile, independent resampling of each side",
            "resamples": report.resamples,
            "seed": report.seed,
            "confidence": report.confidence,
            "min_runs_to_resolve": MIN_RUNS_TO_RESOLVE,
            "metrics": list(COMPARED_METRICS),
        },
        "starvation_flag_threshold": report.starvation_threshold,
        "penalty_factors_default": asdict(DEFAULT_FACTORS),
        "deprecated": ["gang_stalled", "gang_deadlocked", "gang_deadlock_rate"],
        "configs": configs,
        "comparisons": [asdict(c) for c in report.comparisons],
        "flags": report.flags,
        "flag_exemptions": report.flag_exemptions,
        "harness": report.harness,
        "timescale": report.timescale,
    }
    if report.trace_seed_aggregates:
        doc["trace_seed_study"] = {
            "trace_seeds": report.trace_seeds,
            "design": "one run per configuration per trace; " + TRACE_SEED_SPREAD,
            "bootstrap": "percentile, paired by trace: resamples per-trace differences",
            "configs": [
                {
                    "config": a.config,
                    "src": _src(a),
                    "n": a.n,
                    "trace_digests": a.trace_digests,
                    "note": trace_seed_note(a),
                    "aggregate": {
                        name: {"mean": sp.mean, "sd": sp.sd, "n": sp.n}
                        for name, sp in a.spreads.items()
                    },
                    "runs": [asdict(m) for m in a.runs],
                }
                for a in report.trace_seed_aggregates
            ],
            "comparisons": [asdict(c) for c in report.trace_seed_comparisons or []],
        }
    cleaned: dict[str, Any] = _finite(doc)
    return cleaned


def write_results(
    runs: Sequence[Metrics],
    directory: str | Path,
    **options: Any,
) -> Path:
    """Write results.md and results.json for every run given.

    ``runs`` may contain several runs per config; they are aggregated.
    ``options`` are passed to :func:`build_report` (tolerance, seed, resamples,
    confidence, starvation_threshold, harness, timescale, trace_seed_runs,
    trace_seeds).

    Safe to call after every run: each file is written to a temporary name and
    renamed into place, so a crash mid-write leaves the previous complete file
    rather than a truncated one.
    """
    report = build_report(runs, **options)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    _write_atomic(d / "results.md", render_markdown(report, docs_link(d)) + "\n")
    _write_atomic(
        d / "results.json", json.dumps(to_json(report), indent=2, allow_nan=False) + "\n"
    )
    return d / "results.md"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
