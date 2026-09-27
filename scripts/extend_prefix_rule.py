"""Reproduce docs/metrics.md's pre-fix topology-extension figures.

``--topology-penalty extend`` once stretched a job only if every member was
STILL BOUND when its last member bound. A gang whose first member had already
finished -- a gang that never co-ran -- was not stretched at all, which
exempted exactly the worst co-scheduling. The fix stretches a job once every
member has a node (a finished member keeps its node), and stretches the
members still bound. That fix was made before the branch was committed, so no
committed tree contains the old rule; this script rebuilds it from the current
one.

It reads ``src/k8slab/sim.py``, swaps the one block that decides whether a job
is stretched back to the old condition (it refuses to run if that block is no
longer there verbatim), loads the result as a separate module, and replays the
default trace under both rules with the harness docs/metrics.md names:
reference model, default fleet and profile, trace seed 0, harness seed 0,
``--queue-model kube``, speedup 60, zero startup delay, one run per policy --
i.e. ``k8slab bench --reference-model --fleet fleets/default.yaml --profile
default --speedup 60 --repeat 1 --queue-model kube --topology-penalty extend
--startup-delay 0:0``. For each rule it prints topology extension (GPU-h),
makespan (h) and the gangs left entirely unstretched: fully placed gang jobs
whose placement factor is above 1 and whose every member ran no longer than
its trace duration (+1e-6 s: a member's ``end - start`` can fall short of its
duration by float rounding, so an exact comparison miscounts).

Run from the repository root (about half a minute):

    .venv/bin/python scripts/extend_prefix_rule.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from k8slab import fleet as fleet_mod
from k8slab import sim, trace
from k8slab.metrics import compute
from k8slab.model import Job, Observation, PodEvent, RunInfo
from k8slab.topology import JobPlacement, derive, placement_factor

SIM = Path(sim.__file__)

#: The current rule, verbatim from sim.run.
CURRENT = """\
                    nodes = [pods[(job_id, i)].node for i in range(job.gang_size)]
                    placed = tuple(n for n in nodes if n is not None)
                    if len(placed) != job.gang_size:
                        continue
                    f = placement_factor(JobPlacement(placed, job.gpus), topo, factors)
                    for i in range(job.gang_size):
                        member = current.get((job_id, i))
                        if member is not None:
                            member.set_factor(f, t)"""

#: The pre-fix rule: every member must still be bound (in ``current``).
PREFIX = """\
                    members = [current.get((job_id, i)) for i in range(job.gang_size)]
                    if any(m is None for m in members):
                        continue
                    placed = tuple(m.node for m in members if m is not None)
                    f = placement_factor(JobPlacement(placed, job.gpus), topo, factors)
                    for m in members:
                        if m is not None:
                            m.set_factor(f, t)"""

POLICIES = ("D-fifo", "D-random", "D-largest", "D-preempt")
TOLERANCE = 1e-6


def prefix_sim() -> ModuleType:
    """``k8slab.sim`` with the pre-fix stretch condition, as its own module."""
    source = SIM.read_text(encoding="utf-8")
    if source.count(CURRENT) != 1:
        raise SystemExit(
            f"{SIM} no longer contains the stretch condition this script reverts; "
            f"update CURRENT and PREFIX to match it before trusting any output"
        )
    name = "k8slab._sim_prefix_rule"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "k8slab"  # the relative imports resolve against k8slab
    sys.modules[name] = module  # dataclasses look the module up while decorating
    code = compile(source.replace(CURRENT, PREFIX), f"{SIM} (pre-fix rule)", "exec")
    exec(code, module.__dict__)
    return module


def unstretched_gangs(obs: Observation, jobs: list[Job]) -> int:
    topo = derive(obs.fleet)
    by_job: dict[int, list[PodEvent]] = {}
    for p in obs.pods:
        by_job.setdefault(p.job_id, []).append(p)
    count = 0
    for job in jobs:
        if not job.is_gang:
            continue
        pods = sorted(by_job[job.job_id], key=lambda p: p.pod_index)
        nodes = tuple(p.node for p in pods if p.node is not None)
        if len(nodes) != len(pods):
            continue
        if placement_factor(JobPlacement(nodes, job.gpus), topo) <= 1.0:
            continue
        if all(
            p.start_time is not None and p.end_time is not None
            and p.end_time - p.start_time <= job.duration + TOLERANCE
            for p in pods
        ):
            count += 1
    return count


def main() -> int:
    fleet = fleet_mod.load("fleets/default.yaml")
    jobs = trace.generate(trace.profile_named("default"), seed=0)
    info = RunInfo(queue_model="kube", speedup=60.0, topology_penalty="extend")
    rules = (("pre-fix rule", prefix_sim()), ("current rule", sim))
    print("rule          policy     extension GPU-h  makespan h  unstretched gangs")
    for label, module in rules:
        for policy in POLICIES:
            obs = module.run(fleet, jobs, policy, seed=0, run_info=info)
            m = compute(obs)
            print(f"{label:13} {policy:10} {m.topology_extension_gpu_hours:15.1f} "
                  f"{m.makespan_hours:11.2f} {unstretched_gangs(obs, jobs):18d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
