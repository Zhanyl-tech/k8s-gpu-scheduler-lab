from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from k8slab.fleet import load
from k8slab.metrics import Metrics, compute
from k8slab.model import Fleet, Job, NodeClass, Observation, PodEvent
from k8slab.report import (
    DOCS_REFERENCE,
    HEADLINE,
    STARVATION_FLAG_THRESHOLD,
    aggregate_runs,
    build_report,
    docs_link,
    markdown_table,
    render_markdown,
    starvation_flags,
    text_table,
    to_json,
    write_results,
)
from k8slab.sim import run
from k8slab.stats import GATED_METRICS, compare_all
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


def _run(config: str, **values: Any) -> Metrics:
    return dataclasses.replace(BASE, config=config, **values)


def _repeats(config: str, util: float, n: int = 5, **extra: Any) -> list[Metrics]:
    """n runs whose makespan and utilization jitter well inside CV 5%."""
    return [
        _run(config, utilization=util * (1 + 0.004 * i), makespan_hours=10.0 + 0.05 * i, **extra)
        for i in range(n)
    ]


def test_every_gated_floor_is_half_the_displayed_resolution() -> None:
    """The stability floor is defined as 'invisible in the table'. If a column's
    precision changes, the floor must move with it."""
    shown = {c.metric: c for c in HEADLINE}
    for metric, floor in GATED_METRICS.items():
        assert metric in shown, f"gated metric {metric} is not in the headline table"
        assert floor == pytest.approx(shown[metric].resolution), metric


def test_single_runs_open_results_md_with_an_unstable_banner() -> None:
    md = render_markdown(build_report([_run("D-fifo"), _run("D-random")]))
    assert md.splitlines()[2].startswith("> **UNSTABLE.**")
    assert "n=1 is never stable" in md
    assert "unreplicated" in md


def test_stable_repeats_have_no_banner_and_show_mean_sd_with_n() -> None:
    runs = _repeats("D-fifo", 0.6) + _repeats("D-random", 0.5)
    md = render_markdown(build_report(runs))
    assert "UNSTABLE" not in md
    row = next(line for line in md.splitlines() if line.startswith("| D-fifo "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:4] == ["model", "5", "stable"]
    assert "±" in cells[4] and "±" in cells[5]  # makespan, utilization
    assert cells[4] == "10.1±0.1"


def test_one_unstable_config_raises_the_banner_for_the_dataset() -> None:
    wobbly = [_run("K0", makespan_hours=h) for h in (10.0, 14.0, 18.0, 12.0, 16.0)]
    report = build_report(_repeats("D-fifo", 0.6) + wobbly)
    assert not report.stable
    md = render_markdown(report)
    assert "> **UNSTABLE.** 1 of 2 configurations (`K0`)" in md
    assert "makespan_hours CV" in md


def test_results_json_keeps_every_raw_run_next_to_the_aggregate(tmp_path: Path) -> None:
    runs = _repeats("D-fifo", 0.6, n=3) + [_run("K0")]
    write_results(runs, tmp_path)
    doc = json.loads(
        (tmp_path / "results.json").read_text(encoding="utf-8"),
        parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"),
    )
    assert doc["schema"] == 3  # 3: Phase 2 execution layer (CHANGELOG.md)
    by_cfg = {c["config"]: c for c in doc["configs"]}
    assert list(by_cfg) == ["D-fifo", "K0"]  # first-seen order
    assert by_cfg["D-fifo"]["n"] == 3 and len(by_cfg["D-fifo"]["runs"]) == 3
    assert [r["utilization"] for r in by_cfg["D-fifo"]["runs"]] == [
        m.utilization for m in runs[:3]
    ]
    agg = by_cfg["D-fifo"]["aggregate"]["utilization"]
    assert set(agg) == {"mean", "sd", "n", "min", "max", "cv", "gated", "unstable"}
    assert by_cfg["K0"]["verdict"] == "unreplicated"
    # Deprecated, but still in every raw run for backwards compatibility.
    assert "gang_deadlock_rate" in by_cfg["K0"]["runs"][0]
    assert "gang_deadlock_rate" in doc["deprecated"]
    assert (tmp_path / "results.md").read_text(encoding="utf-8").startswith("# Results")


def test_infinite_cv_is_written_as_null_not_invalid_json() -> None:
    runs = [_run("X", footprint_wait_spearman=v) for v in (-0.3, 0.3, 0.0)]
    doc = to_json(build_report(runs))
    cell = doc["configs"][0]["aggregate"]["footprint_wait_spearman"]
    assert cell["cv"] is None and cell["unstable"] is True
    json.dumps(doc, allow_nan=False)


def test_deprecated_gang_deadlock_rate_is_not_a_headline_column() -> None:
    headers = markdown_table([_run("X")]).splitlines()[0]
    assert "gang DL" not in headers and "gang strand GPU-h" in headers
    assert all(c.metric != "gang_deadlock_rate" for c in HEADLINE)


def test_comparisons_cover_baselines_and_k0_and_mark_resolvable() -> None:
    runs = _repeats("K0", 0.70) + _repeats("D-random", 0.50) + _repeats("K1", 0.705)
    report = build_report(runs, resamples=500)
    util = {(c.config, c.baseline): c for c in report.comparisons if c.metric == "utilization"}
    assert set(util) == {("K0", "D-random"), ("K1", "D-random"), ("K1", "K0")}
    assert util[("K0", "D-random")].resolvable
    assert not util[("K1", "K0")].resolvable
    md = render_markdown(report)
    assert "## Differences that survive repetition" in md
    assert "| K0     | D-random | util pp" in md


def test_starvation_flag_fires_only_on_lead_plus_starvation() -> None:
    rnd = _repeats("D-random", 0.50, large_job_starvation_ratio=1.0)
    starving = _repeats("K0", 0.70, large_job_starvation_ratio=STARVATION_FLAG_THRESHOLD + 1)
    fair = _repeats("K1", 0.70, large_job_starvation_ratio=STARVATION_FLAG_THRESHOLD - 0.5)
    slow = _repeats("K2", 0.40, large_job_starvation_ratio=STARVATION_FLAG_THRESHOLD + 5)
    report = build_report(rnd + starving + fair + slow, resamples=300)
    assert len(report.flags) == 1
    assert report.flags[0].startswith("**K0**")
    assert "(resolvable)" in report.flags[0] and "This matches" in report.flags[0]
    assert "inflates utilization by starving large jobs" in render_markdown(report)
    assert report.flag_exemptions == []


def test_starvation_flag_needs_more_starvation_than_d_random() -> None:
    """The reference model flagged D-largest (7.20x) as 'the inflates
    utilization by starving large jobs pattern' against a D-random at 19.14x:
    a lead cannot be bought from D-random by starving large jobs if D-random
    starves them more. Such a configuration is listed, not flagged."""
    rnd = _repeats("D-random", 0.50, large_job_starvation_ratio=19.14)
    less = _repeats("D-largest", 0.70, large_job_starvation_ratio=7.20)
    report = build_report(rnd + less, resamples=300)
    assert report.flags == []
    (exempt,) = report.flag_exemptions
    assert "D-largest" in exempt and "19.14×" in exempt and "Not flagged" in exempt
    md = render_markdown(report)
    assert "_None:" in md and "19.14×" in md
    assert to_json(report)["flag_exemptions"] == report.flag_exemptions


def test_a_flag_on_a_lead_that_is_not_resolvable_does_not_assert_the_pattern() -> None:
    """Flagged (more starvation than D-random, above the threshold), but the
    utilization lead overlaps zero: the wording says the pattern is not
    established, and prints D-random's ratio next to the flagged one."""
    rnd = _repeats("D-random", 0.50, large_job_starvation_ratio=1.0)
    noisy = [_run("K0", utilization=u, makespan_hours=10.0,
                  large_job_starvation_ratio=STARVATION_FLAG_THRESHOLD + 1)
             for u in (0.30, 0.70, 0.45, 0.62, 0.55)]
    report = build_report(rnd + noisy, resamples=300)
    (flag,) = report.flags
    assert "(NOT resolvable)" in flag and "not established" in flag
    assert "against D-random's 1.00×" in flag
    assert "This matches" not in flag


def test_starvation_flag_needs_d_random() -> None:
    aggs = aggregate_runs(_repeats("K0", 0.7, large_job_starvation_ratio=9.0))
    assert starvation_flags(aggs, compare_all(aggs)) == []
    md = render_markdown(build_report(_repeats("K0", 0.7)))
    assert "D-random is not in this dataset" in md


def test_text_table_accepts_the_cli_shape() -> None:
    """cli.py passes one Metrics per config; PART B may pass many."""
    fleet = load("fleets/default.yaml")
    jobs = generate(WorkloadProfile(job_count=40), seed=0)
    results = [compute(run(fleet, jobs, cfg)) for cfg in ("D-fifo", "D-random")]
    table = text_table(results)
    assert table.count("\n") == 3 and "| D-fifo " in table


def test_report_carries_the_factors_actually_used(tmp_path: Path) -> None:
    custom = {**BASE.penalty_factors, "cross_switch": 3.0}
    md = render_markdown(build_report([_run("X", penalty_factors=custom)]))
    assert "cross-switch 3.0 — NOT the defaults" in md
    assert "ASSUMED" in md


def test_undeclared_topology_is_called_out() -> None:
    md = render_markdown(build_report([_run("X", topology_declared=False)]))
    assert "Topology was not declared" in md


ROOT = Path(__file__).resolve().parent.parent


def test_text_table_uses_the_tolerance_results_md_was_written_with() -> None:
    """The CLI prints this table next to a results.md written under
    --sigma-tolerance; the verdict column must agree with it."""
    runs = [_run("D-random", utilization=u) for u in (0.5, 0.6, 0.7)]  # CV ~17%
    row = next(r for r in text_table(runs).splitlines() if r.startswith("| D-random"))
    assert "| unstable" in row
    row = next(r for r in text_table(runs, 10.0).splitlines() if r.startswith("| D-random"))
    assert "| stable" in row and "unstable" not in row
    assert markdown_table(runs, 10.0) == text_table(runs, 10.0)


def test_the_definitions_link_resolves_from_where_results_md_is_written(tmp_path: Path) -> None:
    assert docs_link(ROOT / "results-model") == "../docs/metrics.md"
    assert docs_link(ROOT / "results") == "../docs/metrics.md"
    link = docs_link(ROOT / "results" / "phase2")  # make bench's default
    assert link == "../../docs/metrics.md"
    assert (ROOT / "results" / "phase2" / link).resolve() == (ROOT / "docs" / "metrics.md")
    # Outside the repository no relative link would resolve: name it instead.
    assert docs_link(tmp_path) is None
    md = write_results([_run("X")], tmp_path / "out").read_text()
    assert DOCS_REFERENCE in md and "](../docs/metrics.md)" not in md


def _ts_line(timescale: dict[str, Any]) -> str:
    md = render_markdown(build_report([_run("X")], timescale=timescale))
    return next(line for line in md.splitlines() if "time scaling" in line)


def test_time_scaling_line_states_only_what_was_in_force() -> None:
    from k8slab.timescale import TimeScale

    summary = TimeScale(60.0).summary()
    recorded = _ts_line({**summary, "status": "recorded"})
    # make up's record is intent: stated as configured, never as experienced.
    assert "as `make up` configured kube-scheduler" in recorded
    assert "not read back" in recorded and "experienced by kube-scheduler" not in recorded
    # The flush ticks gate the configured values (docs/limitations.md).
    assert "[60, 120) s" in recorded and "(300, 2100] s" in recorded
    assert "+1800 s" in recorded and "+119 s" in recorded
    model = _ts_line({**summary, "status": "model"})
    assert "in-process kube queue model" in model and "kube-scheduler ran" in model
    unknown = _ts_line({"status": "unknown", "speedup": 60.0, "note": "no record",
                        "intended": summary})
    assert "UNKNOWN" in unknown and "configured at" not in unknown
    assert "timescale.intended" in unknown
    off = _ts_line({"status": "not applied", "speedup": 60.0, "note": "queue model none"})
    assert off == "- time scaling: not applied — queue model none"
    # A record written before `status` existed is read as recorded, too.
    legacy = {k: v for k, v in summary.items() if k != "experienced_range_sim_seconds"}
    assert "as `make up` configured kube-scheduler" in _ts_line(legacy)
