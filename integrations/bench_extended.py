"""Replay the default trace against an in-cluster scheduler (Kueue = K1,
Volcano = K2) with a generous stall_ticks.

`k8slab bench` uses RunConfig's default stall_ticks of 15 real seconds, too
tight for a deep pending queue under a multi-hour replay. The scheduler-specific
wiring lives in src/k8slab/runner.py.

Usage: bench_extended.py [K1|K2] [fleet] [speedup] [jobs]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from k8slab import fleet as fleet_mod  # noqa: E402
from k8slab import metrics as metrics_mod
from k8slab import report as report_mod
from k8slab import trace
from k8slab.runner import RunConfig, Runner  # noqa: E402

#: Results directory prefix per config, so two schedulers' runs never collide.
LABELS = {"K1": "kueue", "K2": "volcano"}


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "K2"
    fleet_path = sys.argv[2] if len(sys.argv) > 2 else "fleets/default.yaml"
    speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    job_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    if config not in LABELS:
        print(f"unknown config {config!r}; have {sorted(LABELS)}", file=sys.stderr)
        return 2

    flt = fleet_mod.load(fleet_path)
    jobs = trace.generate(trace.profile_named("default"), seed=0)[:job_limit]
    print(f"running {len(jobs)} jobs against {config} on fleet {flt.name}", file=sys.stderr)

    obs = Runner(flt, jobs, RunConfig(config=config, speedup=speedup, stall_ticks=600)).run()
    # deadlock_threshold is simulated seconds, so scale by speedup.
    result = metrics_mod.compute(obs, deadlock_threshold=60.0 * speedup)
    out_dir = f"results-{LABELS[config]}-extended"
    report_mod.write_results([result], out_dir)
    print(report_mod.text_table([result]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
