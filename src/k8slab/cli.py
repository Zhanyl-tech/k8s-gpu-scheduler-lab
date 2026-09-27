"""Command line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import clusterconfig, sim, trace
from . import fleet as fleet_mod
from . import metrics as metrics_mod
from . import report as report_mod
from .baselines import DEGENERATE_IDS, SPECS
from .execution import StartupDelay, harness_seed
from .model import QUEUE_MODELS, TOPOLOGY_MODES, Job, RunInfo
from .stats import DEFAULT_CV_TOLERANCE, dataset_verdict
from .timescale import TimeScale, scheduler_config_yaml
from .topology import DEFAULT_FACTORS

DEFAULT_FLEET = "fleets/default.yaml"
DEFAULT_CLUSTER_STATE = "cluster/generated/cluster-state.json"
#: Where ``bench`` writes when ``--results`` is not given. Never ``results/``:
#: results/results.{md,json} is the committed Phase 1 cluster run the README
#: cites, and the CLI defaulting there overwrote it whenever the Makefile's
#: own default (results/phase2) was bypassed.
DEFAULT_CLUSTER_RESULTS = "results/phase2"
DEFAULT_MODEL_RESULTS = "results-model"
#: Exit codes beyond 0/1.
EXIT_MISMATCH = 2
EXIT_UNSTABLE = 3


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from None
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("trace seeds must be distinct")
    return seeds


def _delay(text: str) -> StartupDelay:
    try:
        return StartupDelay.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _speedup(text: str) -> float:
    """A finite compression factor above 0, refused at parse time.

    ``TimeScale`` is built before any other validation runs, so ``--speedup 0``
    used to end in a Python traceback rather than a usage error."""
    try:
        TimeScale(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return float(text)


def _finite_nonnegative(text: str) -> float:
    """A float that is finite and >= 0. ``float()`` accepts "nan" and "inf",
    and every range check compares -- NaN passes all of them."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"expected a finite number >= 0, got {text!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k8slab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_nodes = sub.add_parser("nodes", help="render kwok Node manifests for a fleet")
    p_nodes.add_argument("--fleet", default=DEFAULT_FLEET)

    p_trace = sub.add_parser("trace", help="generate a workload trace")
    p_trace.add_argument("--profile", default="default")
    p_trace.add_argument("--seed", type=int, default=0)
    p_trace.add_argument("--out")

    p_sched = sub.add_parser(
        "scheduler-config",
        help="print the compression-scaled KubeSchedulerConfiguration for a speedup",
    )
    p_sched.add_argument("--speedup", type=_speedup, default=60.0)
    p_sched.add_argument(
        "--serialise", action="store_true",
        help="DIAGNOSTIC: parallelism 1. Changes what is measured; rows are gated.",
    )

    p_cc = sub.add_parser(
        "cluster-config",
        help="write (make up) or check the generated cluster files and their state record",
    )
    p_cc.add_argument("action", choices=["write", "check"])
    p_cc.add_argument("--dir", default="cluster/generated")
    p_cc.add_argument("--speedup", type=_speedup, default=60.0)
    p_cc.add_argument("--serialise", action="store_true")
    p_cc.add_argument("--startup-delay", type=_delay, default=StartupDelay())
    p_cc.add_argument("--kind-template", default="cluster/kind.yaml")

    p_bench = sub.add_parser("bench", help="run configurations and write results")
    p_bench.add_argument(
        "--config", action="append",
        help=f"repeatable. K0, or one of {', '.join(DEGENERATE_IDS)}",
    )
    p_bench.add_argument("--fleet", default=DEFAULT_FLEET)
    p_bench.add_argument("--profile", default="default")
    p_bench.add_argument(
        "--seed", type=int, default=0,
        help="trace seed (fixed across repeats) and base of the per-repeat harness seeds",
    )
    p_bench.add_argument("--speedup", type=_speedup, default=60.0)
    p_bench.add_argument(
        "--results", default=None,
        help=f"output directory (default {DEFAULT_CLUSTER_RESULTS}, or "
             f"{DEFAULT_MODEL_RESULTS} with --reference-model). Refuses a directory "
             f"holding results of another schema or source unless --force.",
    )
    p_bench.add_argument(
        "--force", action="store_true",
        help="overwrite results of another schema or source in --results",
    )
    p_bench.add_argument(
        "--reference-model", action="store_true",
        help="score with the in-process model instead of a cluster. Never comparable "
             "to a cluster run; the output is labelled 'model' either way.",
    )
    p_bench.add_argument(
        "--repeat", type=int, default=1,
        help="runs per configuration, interleaved across configurations (make: REPEAT=5)",
    )
    p_bench.add_argument(
        "--trace-seeds", type=_seeds, default=[],
        help="comma-separated trace seeds for a SEPARATE across-trace-seed study "
             "(one run per config per seed)",
    )
    p_bench.add_argument(
        "--sigma-tolerance", type=_finite_nonnegative, default=DEFAULT_CV_TOLERANCE,
        help="stability tolerance, applied as a coefficient of variation (sd/|mean|) "
             "above each metric's display-resolution floor",
    )
    p_bench.add_argument(
        "--fail-on-unstable", action="store_true",
        help=f"exit {EXIT_UNSTABLE} if any configuration is unstable (n=1 always is)",
    )
    p_bench.add_argument(
        "--queue-model", choices=QUEUE_MODELS, default="kube",
        help="in-process binder's queue: kube-scheduler mechanics (default) or none "
             "(Phase 1: global visibility, no backoff)",
    )
    p_bench.add_argument(
        "--cycle-latency", type=_finite_nonnegative, default=0.0,
        help="simulated seconds per scheduling attempt (kube only). Default 0.",
    )
    p_bench.add_argument(
        "--topology-penalty", choices=TOPOLOGY_MODES, default="report",
        help="off, report (placement_penalty_mean only) or extend (SCENARIO: stretch "
             "runtimes by the ASSUMED factors)",
    )
    p_bench.add_argument(
        "--startup-delay", type=_delay, default=StartupDelay(),
        help="bind-to-Running delay MIN:MAX in REAL milliseconds (kwok Stage delay). "
             "Default 0:0 (Phase 1). 50:200 is an illustrative, uncalibrated setting "
             "(a judgment call, not a measured start-up latency).",
    )
    p_bench.add_argument("--grace-seconds", type=_finite_nonnegative, default=30.0,
                         help="simulated seconds an evicted pod's GPUs stay locked")
    p_bench.add_argument("--checkpoint-fraction", type=float, default=0.0,
                         help="share of an evicted pod's progress kept (default 0)")
    p_bench.add_argument(
        "--admission-gate", action="store_true",
        help="DIAGNOSTIC: submit each pod only after the previous binding was observed. "
             "Changes what is measured; rows are gated and never ranked.",
    )
    p_bench.add_argument(
        "--serialise", action="store_true",
        help="cluster: assert the cluster's kube-scheduler was brought up with "
             "parallelism 1 (make up SERIALISE=1). K0 rows are then gated.",
    )
    p_bench.add_argument("--cluster-state", default=DEFAULT_CLUSTER_STATE)

    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]

    if args.cmd == "nodes":
        print(fleet_mod.render_yaml(fleet_mod.load(args.fleet)))
        return 0

    if args.cmd == "trace":
        jobs = trace.generate(trace.profile_named(args.profile), seed=args.seed)
        if args.out:
            trace.write_csv(jobs, args.out)
            print(f"wrote {len(jobs)} jobs to {args.out}", file=sys.stderr)
        print(trace.summary(jobs))
        return 0

    if args.cmd == "scheduler-config":
        print(scheduler_config_yaml(TimeScale(args.speedup), serialise=args.serialise), end="")
        return 0

    if args.cmd == "cluster-config":
        return _cluster_config(args)

    if args.cmd == "bench":
        return _bench(args)

    return 1


def _cluster_config(args: argparse.Namespace) -> int:
    scale = TimeScale(args.speedup)
    if args.action == "write":
        clusterconfig.write(
            args.dir, scale, serialise=args.serialise, delay=args.startup_delay,
            kind_template=args.kind_template,
        )
        print(f"wrote {args.dir}/ for speedup {args.speedup:g}", file=sys.stderr)
        return 0
    state = clusterconfig.load(f"{args.dir}/{clusterconfig.STATE_FILE}")
    if state is None:
        print(
            f"the cluster exists but {args.dir}/{clusterconfig.STATE_FILE} does not: it was "
            f"not brought up by this Makefile, so its scheduler configuration is unknown. "
            f"Run `make down && make up`.",
            file=sys.stderr,
        )
        return EXIT_MISMATCH
    problems = clusterconfig.mismatches(
        state, speedup=args.speedup, delay=args.startup_delay, serialise=args.serialise
    )
    for problem in problems:
        print(f"REFUSING: {problem}", file=sys.stderr)
    return EXIT_MISMATCH if problems else 0


def _bench(args: argparse.Namespace) -> int:
    configs: list[str] = args.config or ["K0", *DEGENERATE_IDS]
    unknown = [c for c in configs if c != "K0" and c not in SPECS]
    if unknown:
        print(f"unknown configuration(s) {unknown}", file=sys.stderr)
        return 1
    duplicated = sorted({c for c in configs if configs.count(c) > 1})
    if duplicated:
        # A repeated --config ran that configuration again, back to back with
        # the same harness seed, and pooled the copies: `--config D-fifo
        # --config D-fifo` at --repeat 1 reported n=2, "deterministic", and five
        # copies would reach the n >= 5 needed to call a difference resolvable.
        print(f"--config given more than once for {duplicated}; use --repeat for "
              f"repeated runs", file=sys.stderr)
        return 1
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 1
    flt = fleet_mod.load(args.fleet)
    profile = trace.profile_named(args.profile)
    delay: StartupDelay = args.startup_delay
    scale = TimeScale(args.speedup)
    factors = DEFAULT_FACTORS
    serialise = args.serialise
    source = "reference model" if args.reference_model else "cluster"
    if args.results is None:
        args.results = DEFAULT_MODEL_RESULTS if args.reference_model else DEFAULT_CLUSTER_RESULTS
    problem = _results_problem(Path(args.results), source)
    if problem is not None and not args.force:
        print(f"REFUSING to write into {args.results}: {problem}. Pass another "
              f"--results, or --force to overwrite.", file=sys.stderr)
        return 1
    cluster_state = "not applicable (reference model)"

    if args.reference_model:
        if args.serialise:
            print("--serialise configures kube-scheduler; the reference model has none",
                  file=sys.stderr)
            return 1
        timescale_record = _model_timescale(scale, args.queue_model)
    else:
        state = clusterconfig.load(args.cluster_state)
        if state is None:
            banner = "!" * 78
            print(
                f"{banner}\n! WARNING: {args.cluster_state} not found. This cluster was not "
                f"brought up by\n! `make up`, so kube-scheduler's backoff may not be scaled "
                f"for --speedup {args.speedup:g}.\n! The results record the scheduler "
                f"configuration as UNKNOWN.\n{banner}",
                file=sys.stderr,
            )
            cluster_state = "missing: scheduler configuration unknown"
            # Nothing verified the kube-scheduler configuration, so the results
            # must not state one: the scaled values are kept, as INTENDED only.
            timescale_record = {
                "status": report_mod.TIMESCALE_UNKNOWN,
                "speedup": args.speedup,
                "note": (
                    f"{args.cluster_state} not found, so the cluster was not brought up "
                    f"by `make up` and K0's scheduler configuration is unverified"
                    + ("; the in-process binder's kube queue model applied the intended "
                       "values" if args.queue_model == "kube" else "")
                ),
                "intended": scale.summary(),
            }
        else:
            problems = clusterconfig.mismatches(
                state, speedup=args.speedup, delay=delay,
                serialise=True if args.serialise else None,
            )
            if problems:
                for problem in problems:
                    print(f"REFUSING: {problem}", file=sys.stderr)
                return EXIT_MISMATCH
            serialise = state.serialise
            # "recorded", not "applied": cluster-state.json is written by
            # `make up` BEFORE `kind create cluster`, and nothing here reads
            # the running kube-scheduler's flags back. It states what was
            # configured, which the results must not present as verified.
            timescale_record = {**state.timescale, "status": report_mod.TIMESCALE_RECORDED}
            cluster_state = args.cluster_state

    try:
        base = RunInfo(
            queue_model=args.queue_model,
            cycle_latency=args.cycle_latency,
            speedup=args.speedup,
            startup_delay_ms=(delay.min_ms, delay.max_ms),
            topology_penalty=args.topology_penalty,
            grace_seconds=args.grace_seconds,
            checkpoint_fraction=args.checkpoint_fraction,
            admission_gate=args.admission_gate,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    runnable = [c for c in configs if not (args.reference_model and c == "K0")]
    if args.reference_model and "K0" in configs:
        print("    K0 has no reference model (it IS kube-scheduler); skipped", file=sys.stderr)
    seeds = [harness_seed(args.seed, r) for r in range(args.repeat)]
    harness_record: dict[str, Any] = {
        "source": source,
        "argv": ["k8slab", *args.argv],
        "configs": runnable,
        "repeat": args.repeat,
        "trace_seed": args.seed,
        "harness_seeds": seeds,
        "interleaved": True,
        **base.harness(),
        "serialise": serialise,
        "cluster_state": cluster_state,
        "trace_seeds_study": args.trace_seeds,
    }
    jobs = trace.generate(profile, seed=args.seed)
    runs: list[metrics_mod.Metrics] = []
    trace_runs: list[metrics_mod.Metrics] = []

    def write() -> None:
        report_mod.write_results(
            runs, args.results, tolerance=args.sigma_tolerance, harness=harness_record,
            timescale=timescale_record, trace_seed_runs=trace_runs,
            trace_seeds=args.trace_seeds,
        )

    def one(cfg: str, trace_jobs: list[Job], hseed: int) -> metrics_mod.Metrics:
        info = dataclasses.replace(
            base, serialise=serialise and cfg == "K0", harness_seed=hseed
        )
        if args.reference_model:
            obs = sim.run(flt, trace_jobs, cfg, seed=hseed, run_info=info, factors=factors)
        else:
            from .runner import RunConfig, Runner

            obs = Runner(
                flt, trace_jobs,
                RunConfig(config=cfg, speedup=args.speedup, run=info, factors=factors),
            ).run()
        return metrics_mod.compute(obs, factors=factors)

    # Interleaved: every config's repeat 1, then every config's repeat 2, ...
    # The bootstrap treats repeats as independent draws; back-to-back repeats
    # of one config would let slow machine drift pose as a difference.
    for r, hseed in enumerate(seeds):
        for cfg in runnable:
            print(f"==> {cfg}  repeat {r + 1}/{args.repeat}  (harness seed {hseed})",
                  file=sys.stderr)
            runs.append(one(cfg, jobs, hseed))
            # Write after every run: a crash on the last one loses nothing.
            write()
    for ts in args.trace_seeds:
        seed_jobs = trace.generate(profile, seed=ts)
        for cfg in runnable:
            print(f"==> {cfg}  trace seed {ts}", file=sys.stderr)
            trace_runs.append(one(cfg, seed_jobs, seeds[0]))
            write()

    if not runs:
        print("no configurations ran", file=sys.stderr)
        return 1
    print(report_mod.text_table(runs, args.sigma_tolerance))
    print(f"\nwrote {args.results}/results.md and results.json", file=sys.stderr)
    if args.fail_on_unstable:
        stable, notes = dataset_verdict(report_mod.aggregate_runs(runs, args.sigma_tolerance))
        if not stable:
            print("UNSTABLE (--fail-on-unstable):", file=sys.stderr)
            for note in notes:
                print(f"  {note}", file=sys.stderr)
            return EXIT_UNSTABLE
    return 0


def _model_timescale(scale: TimeScale, queue_model: str) -> dict[str, Any]:
    """The reference model has no kube-scheduler. With the kube queue model its
    in-process queue experiences the scaled values; with none, nothing does."""
    if queue_model == "kube":
        return {**scale.summary(), "status": report_mod.TIMESCALE_MODEL}
    return {
        "status": report_mod.TIMESCALE_NOT_APPLIED,
        "speedup": scale.speedup,
        "note": (
            "reference model with --queue-model none: no kube-scheduler and no queue "
            "model ran, so no backoff, unschedulable-pool timeout or flush timer applied"
        ),
    }


def _results_problem(directory: Path, source: str) -> str | None:
    """Why writing into ``directory`` would destroy results of another kind.

    ``None`` when it is empty, or holds this lab's current schema from the same
    source (re-running a benchmark into its own directory is the normal case).
    """
    path = directory / "results.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{path} exists and is not readable JSON"
    if not isinstance(doc, dict):
        return f"{path} holds results in the Phase 1 format (a bare list), e.g. a committed run"
    if doc.get("schema") != report_mod.JSON_SCHEMA_VERSION:
        return f"{path} holds results of schema {doc.get('schema')!r}"
    existing = (doc.get("harness") or {}).get("source")
    if existing is not None and existing != source:
        return f"{path} holds {existing} results and this run is a {source} run"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
