"""Phase 2 additions to metrics, stats and report: execution-layer metrics,
harness identity, scenario and gated rows."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from k8slab.metrics import Metrics, compute
from k8slab.model import Eviction, Fleet, Job, NodeClass, Observation, PodEvent, RunInfo
from k8slab.report import build_report, render_markdown, to_json
from k8slab.stats import aggregate, compare, compare_all

FLEET = Fleet("t", (NodeClass("big", 2, 8),))
H = 3600.0


def _base() -> Metrics:
    obs = Observation(
        "X", FLEET, [Job(1, "a", 0.0, 100.0, 1)],
        [PodEvent(1, 0, 0.0, 0.0, 100.0, "big-0")],
        [(0.0, {"big-0": 7, "big-1": 8}), (100.0, {"big-0": 8, "big-1": 8})],
        horizon=100.0,
    )
    return compute(obs)


BASE = _base()


def _run(config: str, harness: dict[str, Any] | None = None, **values: Any) -> Metrics:
    h = dict(BASE.harness)
    h.update(harness or {})
    return dataclasses.replace(BASE, config=config, harness=h, **values)


def _repeats(config: str, util: float, n: int = 5, **kw: Any) -> list[Metrics]:
    return [_run(config, utilization=util * (1 + 0.004 * i), **kw) for i in range(n)]


# ---- metrics ---------------------------------------------------------------------------


def test_default_run_info_is_phase_1_and_recorded() -> None:
    assert BASE.harness == RunInfo().harness()
    assert BASE.harness["queue_model"] == "none" and BASE.harness["topology_penalty"] == "report"
    assert BASE.startup_overhead_gpu_hours == 0.0 and BASE.preemptions == 0
    assert BASE.gpu_hours_demanded == pytest.approx(100.0 / H)
    assert "harness_seed" not in BASE.scalars()
    assert not BASE.scenario and not BASE.gated


def test_startup_overhead_counts_bind_to_running_and_never_started_pods() -> None:
    jobs = [Job(1, "a", 0.0, 50.0, 2), Job(2, "a", 0.0, 50.0, 4)]
    pods = [
        PodEvent(1, 0, scheduled_time=10.0, start_time=16.0, end_time=66.0, node="big-0"),
        PodEvent(2, 0, scheduled_time=90.0, node="big-1"),  # bound, never Running
    ]
    m = compute(Observation("X", FLEET, jobs, pods, horizon=100.0))
    assert m.startup_overhead_gpu_hours == pytest.approx((2 * 6.0 + 4 * 10.0) / H)


def test_eviction_metrics() -> None:
    job = Job(1, "a", 0.0, 100.0, 4)
    evictions = [
        Eviction(1, 0, "big-0", 4, bind_time=0.0, start_time=5.0, evict_time=45.0,
                 release_time=75.0, reason="preempted by j9-p0", lost_seconds=30.0,
                 retained_seconds=10.0),
    ]
    pods = [PodEvent(1, 0, 80.0, 80.0, 170.0, "big-1")]
    obs = Observation("X", FLEET, [job], pods, horizon=170.0, evictions=evictions)
    m = compute(obs)
    assert m.preemptions == 1
    assert m.preempted_gpu_hours_lost == pytest.approx(4 * 30.0 / H)
    assert m.grace_locked_gpu_hours == pytest.approx(4 * 30.0 / H)
    assert m.startup_overhead_gpu_hours == pytest.approx(4 * 5.0 / H)
    # Delivered = final attempt + the checkpointed part of the evicted one.
    assert m.gpu_hours_used == pytest.approx(4 * (90.0 + 10.0) / H)
    assert m.gpu_hours_used == pytest.approx(m.gpu_hours_demanded)


def test_a_destroyed_bare_pod_is_lost_not_delivered() -> None:
    jobs = [Job(1, "a", 0.0, 100.0, 4), Job(2, "a", 0.0, 100.0, 1)]
    pods = [
        PodEvent(1, 0, 0.0, 0.0, 40.0, "big-0", preempted=True),
        PodEvent(2, 0, 0.0, 0.0, 100.0, "big-1"),
    ]
    ev = Eviction(1, 0, "big-0", 4, 0.0, 0.0, 40.0, 70.0, "control-plane (vanished)",
                  lost_seconds=40.0, requeued=False)
    m = compute(Observation("X", FLEET, jobs, pods, horizon=100.0, evictions=[ev]))
    assert m.gpu_hours_used == pytest.approx(100.0 / H)
    assert m.preempted_gpu_hours_lost == pytest.approx(4 * 40.0 / H)
    assert m.startup_overhead_gpu_hours == 0.0  # counted once, from the PodEvent


def test_wait_counts_only_pending_time_for_an_evicted_job() -> None:
    """Wait is time spent pending, to the last pod's bind. The final PodEvents
    describe only the last attempt; "last bind - submit" also counted the time
    an evicted attempt was bound and Running as waiting. Every size-bias
    metric is built on the same per-job wait."""
    fleet = Fleet("t", (NodeClass("big", 4, 8),))
    jobs = [
        Job(1, "a", 0.0, 100.0, 1),  # single pod: bound 10, evicted 50, re-bound 80
        Job(2, "a", 0.0, 100.0, 1, gang_size=2),  # gang: whole attempt evicted at 30
        Job(3, "a", 0.0, 100.0, 1, gang_size=2),  # partial gang evicted: never admitted
        Job(4, "a", 0.0, 100.0, 1),  # never evicted: Phase 1's formula
    ]

    def ev(job: int, pod: int, bind: float, evict: float) -> Eviction:
        return Eviction(job, pod, "big-0", 1, bind, bind, evict, evict + 30.0, "x",
                        lost_seconds=evict - bind)

    evictions = [ev(1, 0, 10.0, 50.0), ev(2, 0, 10.0, 30.0), ev(2, 1, 20.0, 30.0),
                 ev(3, 0, 10.0, 15.0)]
    pods = [
        PodEvent(1, 0, 80.0, 80.0, 180.0, "big-0"),
        PodEvent(2, 0, 40.0, 40.0, 140.0, "big-1"),
        PodEvent(2, 1, 50.0, 50.0, 150.0, "big-1"),
        PodEvent(3, 0, 40.0, 40.0, 140.0, "big-2"),
        PodEvent(3, 1, 50.0, 50.0, 150.0, "big-2"),
        PodEvent(4, 0, 70.0, 70.0, 170.0, "big-3"),
    ]
    obs = Observation("X", fleet, jobs, pods, horizon=200.0, evictions=evictions)
    m = compute(obs)
    # 1: [0,10) + [50,80) = 40 (was 80).
    # 2: [0,20) u [30,50) = 40 -- admitted 20-30, evicted, re-admitted at 50 (was 50).
    # 3: member 1 pending throughout [0,50) = 50: never fully admitted until 50.
    # 4: 70, unchanged.
    waits = {"1": 40.0, "2": 40.0, "3": 50.0, "4": 70.0}
    assert m.mean_wait == pytest.approx(sum(waits.values()) / 4)
    assert m.wait_by_footprint["1"]["mean"] == pytest.approx((40.0 + 70.0) / 2)
    assert m.wait_by_footprint["2"]["mean"] == pytest.approx((40.0 + 50.0) / 2)
    # Without the eviction records the same PodEvents give Phase 1's formula.
    plain = compute(dataclasses.replace(obs, evictions=[]))
    assert plain.mean_wait == pytest.approx((80.0 + 50.0 + 50.0 + 70.0) / 4)


def test_a_destroyed_pod_s_wait_is_to_its_only_bind() -> None:
    """requeued=False: the PodEvent is the only attempt, so nothing changes."""
    jobs = [Job(1, "a", 0.0, 100.0, 1)]
    pods = [PodEvent(1, 0, 30.0, 30.0, 60.0, "big-0", preempted=True)]
    ev = Eviction(1, 0, "big-0", 1, 30.0, 30.0, 60.0, 90.0, "control-plane (vanished)",
                  lost_seconds=30.0, requeued=False)
    m = compute(Observation("X", FLEET, jobs, pods, horizon=100.0, evictions=[ev]))
    assert m.mean_wait == 30.0


def test_grace_lock_is_capped_at_the_horizon() -> None:
    ev = Eviction(1, 0, "big-0", 8, 0.0, 0.0, 90.0, 120.0, "x", lost_seconds=90.0)
    obs = Observation("X", FLEET, [Job(1, "a", 0.0, 10.0, 8)], [PodEvent(1, 0)],
                      horizon=100.0, evictions=[ev])
    assert compute(obs).grace_locked_gpu_hours == pytest.approx(8 * 10.0 / H)


def test_run_info_validates_its_settings() -> None:
    bad: dict[str, Any]
    for bad in (
        dict(queue_model="fifo"), dict(topology_penalty="on"),
        dict(queue_model="kube"),  # needs a speedup
        dict(startup_delay_ms=(5.0, 1.0)), dict(startup_delay_ms=(1.0, 5.0)),  # no speedup
        dict(checkpoint_fraction=1.5), dict(grace_seconds=-1.0),
        # Non-finite values pass every `< 0` check; a NaN grace lock never releases.
        dict(grace_seconds=float("nan")), dict(grace_seconds=float("inf")),
        dict(cycle_latency=float("nan"), queue_model="kube", speedup=60.0),
        dict(speedup=float("inf")), dict(speedup=float("nan")),
        dict(startup_delay_ms=(float("nan"), 1.0), speedup=60.0),
        dict(startup_delay_ms=(0.0, float("inf")), speedup=60.0),
        dict(checkpoint_fraction=float("nan")),
    ):
        with pytest.raises(ValueError):
            RunInfo(**bad)
    assert RunInfo(admission_gate=True).gated and RunInfo(serialise=True).gated
    assert RunInfo(topology_penalty="extend").scenario


# ---- stats: pooling and comparison refusals ----------------------------------------------


def test_scenario_runs_are_never_pooled_with_measured_runs() -> None:
    with pytest.raises(ValueError, match="scenario"):
        aggregate("X", [_run("X"), _run("X", {"topology_penalty": "extend"})])


@pytest.mark.parametrize(
    "change",
    [{"queue_model": "kube"}, {"speedup": 120.0}, {"startup_delay_ms": [50.0, 200.0]},
     {"grace_seconds": 5.0}, {"admission_gate": True}],
)
def test_runs_from_different_harness_settings_are_never_pooled(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="harness settings"):
        aggregate("X", [_run("X"), _run("X", change)])


def test_harness_seeds_may_differ_across_repeats() -> None:
    agg = aggregate("X", [_run("X", harness_seed=0), _run("X", harness_seed=7)])
    assert agg.n == 2


def test_src_labels() -> None:
    assert aggregate("X", [_run("X")]).src == "model"
    assert aggregate("X", [_run("X", {"topology_penalty": "extend"})]).src == "model+topo"
    assert aggregate("X", [_run("X", {"serialise": True})]).src == "model+gated"
    both = {"topology_penalty": "extend", "admission_gate": True}
    assert aggregate("X", [_run("X", both, measured_on_cluster=True)]).src == "cluster+topo+gated"


def test_compare_refuses_gated_scenario_and_mismatched_rows() -> None:
    base = aggregate("D-fifo", _repeats("D-fifo", 0.6))
    gated = aggregate("K0", _repeats("K0", 0.7, harness={"serialise": True}))
    topo = aggregate("K1", _repeats("K1", 0.7, harness={"topology_penalty": "extend"}))
    kube = aggregate("K2", _repeats("K2", 0.7, harness={"queue_model": "kube"}))
    assert "gated" in compare(gated, base, "utilization").note
    assert "scenario" in compare(topo, base, "utilization").note
    refused = compare(kube, base, "utilization")
    assert "queue_model" in refused.note and not refused.resolvable and refused.diff is None


def test_compare_all_leaves_gated_rows_out() -> None:
    aggs = [
        aggregate("D-fifo", _repeats("D-fifo", 0.6)),
        aggregate("D-random", _repeats("D-random", 0.5)),
        aggregate("K0", _repeats("K0", 0.7, harness={"serialise": True})),
    ]
    pairs = {(c.config, c.baseline) for c in compare_all(aggs, resamples=50)}
    assert pairs == {("D-fifo", "D-random")}


def test_scenario_rows_compare_among_themselves() -> None:
    topo = {"topology_penalty": "extend"}
    aggs = [aggregate("D-fifo", _repeats("D-fifo", 0.6, harness=topo)),
            aggregate("D-random", _repeats("D-random", 0.5, harness=topo))]
    (c, *_) = [c for c in compare_all(aggs, resamples=200) if c.metric == "utilization"]
    assert c.resolvable


# ---- report --------------------------------------------------------------------------------


def test_report_records_how_the_runs_were_produced() -> None:
    harness = {"source": "reference model", "argv": ["k8slab", "bench"], "repeat": 5,
               "startup_delay_ms": [0.0, 0.0]}
    timescale = {"speedup": 60.0, "experienced_sim_seconds": {
        "initial_backoff": 60.0, "max_backoff": 60.0, "backoff_flush_interval": 60.0,
        "unschedulable_flush_interval": 1800.0},
        "backoff_residual_error_sim_seconds": 50.0, "initial_backoff_residual_sim_seconds": 59.0}
    report = build_report(_repeats("D-fifo", 0.6), harness=harness, timescale=timescale)
    md = render_markdown(report)
    assert "- command: `k8slab bench`" in md and "residual 50 s" in md
    doc = to_json(report)
    assert doc["harness"]["repeat"] == 5 and doc["timescale"]["speedup"] == 60.0
    json.dumps(doc, allow_nan=False)


def test_report_explains_deterministic_model_rows_and_gated_rows() -> None:
    same = [_run("D-fifo")] * 3
    gated = _repeats("K0", 0.7, harness={"admission_gate": True})
    md = render_markdown(build_report(same + gated))
    assert "It is a property of the model, not evidence that a cluster run" in md
    assert "Gated diagnostic rows (K0)" in md
    k0_rows = [line for line in md.splitlines() if line.startswith("| K0 ")]
    assert "model+gated" in k0_rows[0]  # the headline row carries the label


def test_execution_layer_table_is_always_rendered() -> None:
    md = render_markdown(build_report([_run("X", preemptions=3, grace_locked_gpu_hours=1.5)]))
    assert "## Execution layer" in md
    section = md.split("## Execution layer", 1)[1]
    row = next(line for line in section.splitlines() if line.startswith("| X "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:3] == ["X", "model", "1"] and cells[6] == "3" and cells[8] == "1.5"


def test_starvation_flag_only_compares_like_with_like() -> None:
    rnd = _repeats("D-random", 0.5, large_job_starvation_ratio=1.0)
    other = _repeats("K0", 0.7, harness={"queue_model": "kube"}, large_job_starvation_ratio=9.0)
    assert build_report(rnd + other, resamples=100).flags == []


def test_execution_notes_reach_results() -> None:
    obs = Observation("X", FLEET, [Job(1, "a", 0.0, 10.0, 1)],
                      [PodEvent(1, 0, 0.0, 0.0, 10.0, "big-0")], horizon=10.0,
                      notes=["2 sample(s) had a node where a grace lock overlapped"])
    m = compute(obs)
    assert m.notes == obs.notes and "notes" not in m.scalars()
    report = build_report([m])
    assert "## Execution notes" in render_markdown(report)
    assert to_json(report)["configs"][0]["runs"][0]["notes"] == obs.notes


def _trace_study() -> tuple[list[Metrics], list[Metrics]]:
    """Main repeats, plus a 5-trace study with a large trace effect: D-fifo's
    utilization is D-random's + 0.020..0.024 on every trace."""
    main = _repeats("D-fifo", 0.6) + _repeats("D-random", 0.5)
    seed_runs: list[Metrics] = []
    for i, level in enumerate([0.3, 0.5, 0.7, 0.4, 0.6]):
        seed_runs.append(_run("D-fifo", utilization=level + 0.02 + 0.001 * i,
                              trace_digest=f"t{i}"))
        seed_runs.append(_run("D-random", utilization=level, trace_digest=f"t{i}"))
    return main, seed_runs


def test_the_trace_seed_section_says_what_its_spread_is() -> None:
    """It used to open with "Its spread is WORKLOAD variance, not harness
    variance" and then print, per configuration, "UNSTABLE ... not resolvable
    by this harness". One run per configuration per trace cannot separate
    workload, harness and policy randomness, and the harness-stability verdict
    does not apply to a spread across different workloads."""
    main, seed_runs = _trace_study()
    report = build_report(main, trace_seed_runs=seed_runs, trace_seeds=[0, 1, 2, 3, 4],
                          resamples=500)
    md = render_markdown(report)
    assert "WORKLOAD variance, not harness variance" not in md
    section = md.split("## Across trace seeds", 1)[1]
    assert "MIXES workload variance" in section and "cannot separate them" in section
    assert "UNSTABLE" not in section and "not resolvable by this harness" not in section
    header = next(line for line in section.splitlines() if line.startswith("| config"))
    assert "| traces |" in header and "verdict" not in header
    study = to_json(report)["trace_seed_study"]
    assert "MIXES" in study["design"]
    assert all("UNSTABLE" not in c["note"] and "not a stability verdict" in c["note"]
               for c in study["configs"])


def test_trace_seed_comparisons_are_paired_by_trace() -> None:
    """Every configuration replays the same traces, so the difference is taken
    per trace: the trace effect cancels. Resampled as independent repeats the
    same runs cannot resolve a consistent 2-point lead."""
    main, seed_runs = _trace_study()
    report = build_report(main, trace_seed_runs=seed_runs, trace_seeds=[0, 1, 2, 3, 4],
                          resamples=500)
    util = next(c for c in report.trace_seed_comparisons or [] if c.metric == "utilization")
    assert util.paired and util.resolvable and (util.n_config, util.n_baseline) == (5, 5)
    assert util.low is not None and util.low >= 0.02 - 1e-12
    assert util.diff == pytest.approx(0.022)
    fifo = aggregate("D-fifo", [m for m in seed_runs if m.config == "D-fifo"])
    rnd = aggregate("D-random", [m for m in seed_runs if m.config == "D-random"])
    assert not compare(fifo, rnd, "utilization", resamples=500).resolvable
    # The main comparisons are unchanged: independent repeats, not paired.
    assert report.comparisons and not any(c.paired for c in report.comparisons)
