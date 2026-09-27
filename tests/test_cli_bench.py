"""The bench command (repeats, seeds, refusals) and the generated cluster files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from k8slab import clusterconfig
from k8slab.cli import EXIT_MISMATCH, EXIT_UNSTABLE, main
from k8slab.execution import StartupDelay, harness_seed
from k8slab.timescale import CONFIG_DIR_IN_NODE, TimeScale

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def _bench(tmp_path: Path, *extra: str) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "res"
    code = main(["bench", "--reference-model", "--profile", "light",
                 "--config", "D-fifo", "--config", "D-random", "--results", str(out), *extra])
    doc = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    return code, doc


def test_repeats_are_aggregated_with_derived_harness_seeds(tmp_path: Path) -> None:
    code, doc = _bench(tmp_path, "--repeat", "3")
    assert code == 0 and doc["schema"] == 3
    by = {c["config"]: c for c in doc["configs"]}
    assert by["D-fifo"]["n"] == by["D-random"]["n"] == 3
    # Deterministic policy, fixed trace: bit-identical repeats, labelled so.
    assert by["D-fifo"]["verdict"] == "deterministic"
    assert "says nothing about whether a cluster run" in by["D-fifo"]["note"]
    # D-random's own randomness is measured across repeats.
    assert by["D-random"]["aggregate"]["mean_wait"]["sd"] > 0
    assert [r["harness_seed"] for r in by["D-random"]["runs"]] == [
        harness_seed(0, r) for r in range(3)
    ]
    # One trace across repeats.
    assert len(by["D-fifo"]["trace_digests"]) == 1
    assert doc["harness"]["repeat"] == 3 and doc["harness"]["interleaved"] is True
    assert doc["harness"]["queue_model"] == "kube"  # the default for new results
    assert doc["timescale"]["backoff_residual_error_sim_seconds"] == 50.0
    md = (tmp_path / "res" / "results.md").read_text()
    assert "## How these runs were produced" in md and "## Execution layer" in md


def test_results_are_written_after_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k8slab import report

    seen: list[int] = []
    real = report.write_results

    def spy(runs: Any, directory: Any, **options: Any) -> Path:
        seen.append(len(runs))
        return real(runs, directory, **options)

    monkeypatch.setattr(report, "write_results", spy)
    code, _ = _bench(tmp_path, "--repeat", "2")
    assert code == 0 and seen == [1, 2, 3, 4]


def test_repeats_are_interleaved_across_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap treats repeats as independent draws; running every repeat
    of one configuration before the next would let machine drift pose as a
    difference. The order is recorded where the runs happen: results.json
    groups runs by configuration, so the order cannot be read back from it
    (the previous form of this test compared the grouped seeds, which are the
    same either way, and passed with the loops swapped)."""
    from k8slab import sim

    order: list[tuple[str, int]] = []
    real = sim.run

    def spy(fleet: Any, jobs: Any, config: str, seed: int = 0, **kw: Any) -> Any:
        order.append((config, seed))
        return real(fleet, jobs, config, seed, **kw)

    monkeypatch.setattr(sim, "run", spy)
    code, doc = _bench(tmp_path, "--repeat", "2")
    assert code == 0
    h1 = harness_seed(0, 1)
    assert order == [("D-fifo", 0), ("D-random", 0), ("D-fifo", h1), ("D-random", h1)]
    assert doc["harness"]["interleaved"] is True


def test_a_repeated_config_is_refused_not_pooled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--config D-fifo --config D-fifo` at one repeat used to report n=2
    ("deterministic") from two back-to-back copies with the same seed, while
    results.json said repeat 1; five copies would reach the n needed to call a
    difference resolvable."""
    code, doc = _bench(tmp_path, "--config", "D-fifo")
    assert code == 1 and doc == {}
    assert "more than once for ['D-fifo']" in capsys.readouterr().err


def test_trace_seed_study_is_a_separate_section(tmp_path: Path) -> None:
    code, doc = _bench(tmp_path, "--trace-seeds", "3,4")
    assert code == 0
    study = doc["trace_seed_study"]
    assert study["trace_seeds"] == [3, 4]
    by = {c["config"]: c for c in study["configs"]}
    assert by["D-fifo"]["n"] == 2 and len(by["D-fifo"]["trace_digests"]) == 2
    # The main table is untouched by the study.
    assert {c["config"]: c["n"] for c in doc["configs"]} == {"D-fifo": 1, "D-random": 1}
    assert "## Across trace seeds" in (tmp_path / "res" / "results.md").read_text()


def test_fail_on_unstable(tmp_path: Path) -> None:
    code, _ = _bench(tmp_path, "--fail-on-unstable")  # n=1 is never stable
    assert code == EXIT_UNSTABLE
    code, _ = _bench(tmp_path, "--repeat", "2", "--config", "D-largest", "--fail-on-unstable",
                     "--sigma-tolerance", "10")
    assert code == 0


def test_scenario_and_gated_rows_are_labelled(tmp_path: Path) -> None:
    code, doc = _bench(tmp_path, "--topology-penalty", "extend")
    assert code == 0 and {c["src"] for c in doc["configs"]} == {"model+topo"}
    code, doc = _bench(tmp_path, "--admission-gate")
    assert code == 0 and {c["src"] for c in doc["configs"]} == {"model+gated"}
    assert doc["comparisons"] == []  # gated rows are never ranked


def test_model_refuses_serialise(tmp_path: Path) -> None:
    code, _ = _bench(tmp_path, "--serialise")
    assert code == 1


def test_cluster_bench_refuses_a_speedup_the_cluster_was_not_built_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gen = tmp_path / "gen"
    clusterconfig.write(gen, TimeScale(60.0), kind_template=ROOT / "cluster/kind.yaml")
    code = main(["bench", "--speedup", "120", "--cluster-state", str(gen / "cluster-state.json"),
                 "--results", str(tmp_path / "r")])
    assert code == EXIT_MISMATCH
    assert "SPEEDUP 120 differs from the 60" in capsys.readouterr().err
    code = main(["bench", "--speedup", "60", "--startup-delay", "50:200",
                 "--cluster-state", str(gen / "cluster-state.json"), "--results", str(tmp_path)])
    assert code == EXIT_MISMATCH


# ---- generated cluster files ----------------------------------------------------------


def test_cluster_config_write_generates_every_file(tmp_path: Path) -> None:
    gen = tmp_path / "gen"
    assert main(["cluster-config", "write", "--dir", str(gen), "--speedup", "60",
                 "--startup-delay", "50:200"]) == 0
    sched = yaml.safe_load((gen / "scheduler-config.yaml").read_text())
    assert sched["podInitialBackoffSeconds"] == 1 and sched["parallelism"] == 16
    kind = yaml.safe_load((gen / "kind.yaml").read_text())
    node = kind["nodes"][0]
    assert node["extraMounts"] == [{"hostPath": "cluster/generated",
                                    "containerPath": CONFIG_DIR_IN_NODE, "readOnly": True}]
    patch = yaml.safe_load(node["kubeadmConfigPatches"][0])
    args = patch["scheduler"]["extraArgs"]
    assert args["config"] == f"{CONFIG_DIR_IN_NODE}/scheduler-config.yaml"
    assert args["leader-elect"] == "false"
    assert args["pod-max-in-unschedulable-pods-duration"] == "5s"
    assert "kube-api-qps" not in args  # ignored under --config; moved into the file
    assert patch["scheduler"]["extraVolumes"][0]["mountPath"] == CONFIG_DIR_IN_NODE
    delay = yaml.safe_load((gen / "pod-ready-delay.patch.yaml").read_text())
    assert delay == {"spec": {"delay": {"durationMilliseconds": 50,
                                        "jitterDurationMilliseconds": 200}}}
    state = json.loads((gen / "cluster-state.json").read_text())
    assert state["speedup"] == 60.0 and state["startup_delay_ms"] == [50.0, 200.0]
    assert state["timescale"]["backoff_residual_error_sim_seconds"] == 50.0


def test_a_zero_delay_removes_a_stale_patch(tmp_path: Path) -> None:
    gen = tmp_path / "gen"
    clusterconfig.write(gen, TimeScale(60.0), delay=StartupDelay(50, 200))
    clusterconfig.write(gen, TimeScale(60.0))
    assert not (gen / "pod-ready-delay.patch.yaml").exists()
    const = clusterconfig.delay_patch(StartupDelay(80, 80))
    assert const == {"spec": {"delay": {"durationMilliseconds": 80}}}


def test_cluster_config_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gen = tmp_path / "gen"
    assert main(["cluster-config", "check", "--dir", str(gen)]) == EXIT_MISMATCH
    assert "make down && make up" in capsys.readouterr().err
    main(["cluster-config", "write", "--dir", str(gen), "--speedup", "60"])
    assert main(["cluster-config", "check", "--dir", str(gen), "--speedup", "60"]) == 0
    assert main(["cluster-config", "check", "--dir", str(gen), "--speedup", "30"]) == EXIT_MISMATCH
    assert main(["cluster-config", "check", "--dir", str(gen), "--speedup", "60",
                 "--serialise"]) == EXIT_MISMATCH


def test_the_kind_template_and_the_generator_cannot_drift(tmp_path: Path) -> None:
    template = (ROOT / "cluster/kind.yaml").read_text()
    clusterconfig.render_kind_config(template, TimeScale(60.0))  # consistent today
    broken = template.replace('leader-elect: "false"', 'leader-elect: "true"')
    with pytest.raises(ValueError, match="leader-elect"):
        clusterconfig.render_kind_config(broken, TimeScale(60.0))
    unmounted = template.replace("containerPath: /etc/kubernetes/k8slab", "containerPath: /x")
    with pytest.raises(ValueError, match="does not mount"):
        clusterconfig.render_kind_config(unmounted, TimeScale(60.0))


def test_scheduler_config_command_prints_yaml(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scheduler-config", "--speedup", "60", "--serialise"]) == 0
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["kind"] == "KubeSchedulerConfiguration" and cfg["parallelism"] == 1


def test_state_round_trips_and_rejects_unknown_schemas(tmp_path: Path) -> None:
    state = clusterconfig.write(tmp_path, TimeScale(90.0), serialise=True)
    loaded = clusterconfig.load(tmp_path / "cluster-state.json")
    assert loaded == state and loaded is not None and loaded.serialise
    assert clusterconfig.load(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="schema"):
        clusterconfig.ClusterState.from_json({"schema": 99})


# ---- where results go, and what they claim about time scaling -------------------------


def test_default_results_directory_is_never_the_committed_phase_1_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fleet = str(ROOT / "fleets" / "default.yaml")
    assert main(["bench", "--reference-model", "--profile", "light", "--config", "D-fifo",
                 "--fleet", fleet]) == 0
    assert (tmp_path / "results-model" / "results.json").exists()
    assert not (tmp_path / "results").exists()


def test_bench_refuses_to_overwrite_results_of_another_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phase1 = tmp_path / "r"
    phase1.mkdir()
    legacy = '[{"config": "K0", "measured_on_cluster": true}]\n'
    (phase1 / "results.json").write_text(legacy)
    args = ["bench", "--reference-model", "--profile", "light", "--config", "D-fifo",
            "--results", str(phase1)]
    assert main(args) == 1
    assert "Phase 1 format" in capsys.readouterr().err
    assert (phase1 / "results.json").read_text() == legacy  # untouched
    # Cluster results in a directory are not overwritten by a model run.
    cluster = tmp_path / "c"
    cluster.mkdir()
    (cluster / "results.json").write_text(json.dumps({"schema": 3,
                                                      "harness": {"source": "cluster"}}))
    assert main([*args[:-1], str(cluster)]) == 1
    assert "holds cluster results" in capsys.readouterr().err
    # Re-running into one's own directory is the normal case; --force overrides.
    assert main(args + ["--force"]) == 0
    assert main(args) == 0


def test_model_results_say_which_time_scaling_applied(tmp_path: Path) -> None:
    _, doc = _bench(tmp_path)
    assert doc["timescale"]["status"] == "model"
    md = (tmp_path / "res" / "results.md").read_text()
    assert "in-process kube queue model" in md
    _, doc = _bench(tmp_path, "--queue-model", "none")
    ts = doc["timescale"]
    assert ts["status"] == "not applied" and "backoff_residual_error_sim_seconds" not in ts
    md = (tmp_path / "res" / "results.md").read_text()
    assert "- time scaling: not applied" in md and "kube-scheduler backoff" not in md


class _ModelRunner:
    """Stands in for the cluster Runner: no cluster exists here."""

    def __init__(self, fleet: Any, jobs: Any, cfg: Any) -> None:
        self.fleet, self.jobs, self.cfg = fleet, jobs, cfg

    def run(self) -> Any:
        import dataclasses

        from k8slab import sim

        obs = sim.run(self.fleet, self.jobs, self.cfg.config,
                      seed=self.cfg.run.harness_seed or 0, run_info=self.cfg.run)
        return dataclasses.replace(obs, measured_on_cluster=True)


def test_a_cluster_run_without_its_state_record_claims_no_scheduler_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from k8slab import runner

    monkeypatch.setattr(runner, "Runner", _ModelRunner)
    out = tmp_path / "r"
    code = main(["bench", "--profile", "light", "--config", "D-fifo",
                 "--cluster-state", str(tmp_path / "missing.json"), "--results", str(out)])
    assert code == 0 and "UNKNOWN" in capsys.readouterr().err
    doc = json.loads((out / "results.json").read_text())
    ts = doc["timescale"]
    assert doc["harness"]["cluster_state"].startswith("missing")
    assert ts["status"] == "unknown" and "configured_real_seconds" not in ts
    assert ts["intended"]["configured_real_seconds"]["podMaxBackoffSeconds"] == 1
    md = (out / "results.md").read_text()
    assert "configuration **UNKNOWN**" in md and "configured kube-scheduler" not in md
    # With the record, the configuration make up wrote is stated as configured
    # -- the record is written before the cluster exists and nothing reads the
    # running scheduler's flags back, so never as applied or experienced.
    gen = tmp_path / "gen"
    clusterconfig.write(gen, TimeScale(60.0), kind_template=ROOT / "cluster/kind.yaml")
    code = main(["bench", "--profile", "light", "--config", "D-fifo", "--force",
                 "--cluster-state", str(gen / "cluster-state.json"), "--results", str(out)])
    assert code == 0
    doc = json.loads((out / "results.json").read_text())
    assert doc["timescale"]["status"] == "recorded"
    md = (out / "results.md").read_text()
    assert "as `make up` configured kube-scheduler" in md and "not read back" in md
    assert "experienced by kube-scheduler" not in md


class _ClusterStandIn(_ModelRunner):
    """As _ModelRunner, but K0 (kube-scheduler, which the model does not have)
    is replayed with D-fifo's decisions and keeps its K0 label and run info."""

    def run(self) -> Any:
        import dataclasses

        from k8slab import sim

        policy = "D-fifo" if self.cfg.config == "K0" else self.cfg.config
        obs = sim.run(self.fleet, self.jobs, policy,
                      seed=self.cfg.run.harness_seed or 0, run_info=self.cfg.run)
        return dataclasses.replace(obs, config=self.cfg.config, measured_on_cluster=True)


def test_a_cluster_brought_up_serialised_gates_k0_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make up SERIALISE=1` then a plain `make bench`: the cluster's
    kube-scheduler runs at parallelism 1 whatever bench is told, so the state
    record -- not the flag -- decides that K0's rows are a gated diagnostic.
    Without it they would be ranked against the headline configurations."""
    from k8slab import runner

    monkeypatch.setattr(runner, "Runner", _ClusterStandIn)
    gen = tmp_path / "gen"
    clusterconfig.write(gen, TimeScale(60.0), serialise=True,
                        kind_template=ROOT / "cluster/kind.yaml")
    out = tmp_path / "r"
    code = main(["bench", "--profile", "light", "--config", "K0", "--config", "D-fifo",
                 "--config", "D-random", "--cluster-state", str(gen / "cluster-state.json"),
                 "--results", str(out)])
    assert code == 0
    doc = json.loads((out / "results.json").read_text())
    src = {c["config"]: c["src"] for c in doc["configs"]}
    assert src == {"K0": "cluster+gated", "D-fifo": "cluster", "D-random": "cluster"}
    assert doc["harness"]["serialise"] is True
    assert doc["comparisons"]  # D-fifo against D-random is still compared ...
    assert all("K0" not in (c["config"], c["baseline"]) for c in doc["comparisons"])  # ... K0 never


def test_the_terminal_table_uses_the_sigma_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, doc = _bench(tmp_path, "--repeat", "3", "--sigma-tolerance", "10")
    assert code == 0
    verdict = {c["config"]: c["verdict"] for c in doc["configs"]}["D-random"]
    row = next(r for r in capsys.readouterr().out.splitlines() if r.startswith("| D-random"))
    assert f"| {verdict} " in row and verdict == "stable"


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_make_clean_keeps_a_live_clusters_generated_state(tmp_path: Path) -> None:
    """`make clean` used to delete cluster/generated even with the kind cluster
    up: the next `make bench` then found no state record, warned instead of
    refusing a SPEEDUP/STARTUP_DELAY mismatch, and recorded the scheduler
    configuration as UNKNOWN. A fake `kind` stands in for the cluster; every
    path the target removes is redirected into tmp_path."""
    bin_dir, work = tmp_path / "bin", tmp_path / "work"
    bin_dir.mkdir()
    work.mkdir()
    gen = work / "gen"
    kind = bin_dir / "kind"

    def clean(clusters: str) -> str:
        kind.write_text(f"#!/bin/sh\nprintf '{clusters}'\n")
        kind.chmod(0o755)
        gen.mkdir(exist_ok=True)
        (gen / "cluster-state.json").write_text("{}")
        done = subprocess.run(
            ["make", "-s", "-f", str(ROOT / "Makefile"), "-C", str(work), "clean",
             f"VENV={work / 'venv'}", f"MODEL_RESULTS={work / 'rm'}", f"GENERATED={gen}"],
            env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
            capture_output=True, text=True, check=True,
        )
        return done.stdout

    assert "keeping" in clean("k8slab\\n") and (gen / "cluster-state.json").exists()
    clean("other\\n")  # another kind cluster is not this lab's
    assert not gen.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "--reference-model", "--speedup", "0"],
        ["bench", "--reference-model", "--speedup", "inf"],
        ["scheduler-config", "--speedup", "-1"],
        ["cluster-config", "check", "--speedup", "nan"],
        ["bench", "--reference-model", "--startup-delay", "nan"],
        ["bench", "--reference-model", "--startup-delay", "0:inf"],
        ["bench", "--reference-model", "--grace-seconds", "nan"],
        ["bench", "--reference-model", "--cycle-latency", "inf"],
        ["bench", "--reference-model", "--sigma-tolerance", "nan"],
    ],
)
def test_non_finite_or_non_positive_numbers_are_usage_errors(
    argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused by argparse (exit 2, a message) before anything runs or is
    written. `--speedup 0` used to raise a traceback from TimeScale; `nan`
    delays and grace periods ran to the 48 h cap and exited 0 with NaN
    GPU-hours; a NaN --sigma-tolerance would call every configuration stable
    (no CV is greater than NaN)."""
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--results", str(tmp_path / "r")] if argv[0] == "bench" else argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err
    assert not (tmp_path / "r").exists()


def test_cluster_config_refuses_an_infinite_delay_before_writing_anything(
    tmp_path: Path,
) -> None:
    """It used to write scheduler-config.yaml and kind.yaml, then raise
    OverflowError converting inf milliseconds to an integer."""
    gen = tmp_path / "gen"
    with pytest.raises(SystemExit) as exc:
        main(["cluster-config", "write", "--dir", str(gen), "--startup-delay", "0:inf"])
    assert exc.value.code == 2 and not gen.exists()
