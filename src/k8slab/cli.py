"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from . import fleet as fleet_mod
from . import metrics as metrics_mod
from . import report as report_mod
from . import sim, trace
from .baselines import DEGENERATE_IDS

DEFAULT_FLEET = "fleets/default.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k8slab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_nodes = sub.add_parser("nodes", help="render kwok Node manifests for a fleet")
    p_nodes.add_argument("--fleet", default=DEFAULT_FLEET)

    p_trace = sub.add_parser("trace", help="generate a workload trace")
    p_trace.add_argument("--profile", default="default")
    p_trace.add_argument("--seed", type=int, default=0)
    p_trace.add_argument("--out")

    p_bench = sub.add_parser("bench", help="run configurations and write results")
    p_bench.add_argument(
        "--config", action="append",
        help=f"repeatable. K0, or one of {', '.join(DEGENERATE_IDS)}",
    )
    p_bench.add_argument("--fleet", default=DEFAULT_FLEET)
    p_bench.add_argument("--profile", default="default")
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--speedup", type=float, default=60.0)
    p_bench.add_argument("--results", default="results")
    p_bench.add_argument(
        "--reference-model", action="store_true",
        help="score with the in-process model instead of a cluster. Never comparable "
             "to a cluster run; the output is labelled 'model' either way.",
    )

    args = parser.parse_args(argv)

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

    if args.cmd == "bench":
        configs = args.config or ["K0", *DEGENERATE_IDS]
        flt = fleet_mod.load(args.fleet)
        jobs = trace.generate(trace.profile_named(args.profile), seed=args.seed)
        results = []
        for cfg in configs:
            print(f"==> {cfg}", file=sys.stderr)
            if args.reference_model:
                if cfg == "K0":
                    print("    K0 has no reference model (it IS kube-scheduler); skipped",
                          file=sys.stderr)
                    continue
                obs = sim.run(flt, jobs, cfg, seed=args.seed)
            else:
                from .runner import RunConfig, Runner

                obs = Runner(
                    flt, jobs, RunConfig(config=cfg, speedup=args.speedup)
                ).run()
            results.append(metrics_mod.compute(obs))
            # Write after every configuration. A fifteen-minute replay that
            # dies on the last config should not discard the ones that worked.
            report_mod.write_results(results, args.results)
        if not results:
            print("no configurations ran", file=sys.stderr)
            return 1
        path = report_mod.write_results(results, args.results)
        print(report_mod.text_table(results))
        print(f"\nwrote {path}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
