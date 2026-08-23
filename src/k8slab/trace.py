"""Workload traces: synthetic generation now, translated `sacct` in Phase 2.

The record schema is fixed here rather than in the generator, so a real trace
and a synthetic one are indistinguishable to everything downstream. Phase 2
adds a ``from_sacct`` loader that emits the same :class:`~k8slab.model.Job`
records; nothing else has to change.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .model import Job

#: Column order for the on-disk CSV form. Phase 2's sacct translator writes
#: this same header, which is the whole point of pinning it here.
TRACE_COLUMNS = (
    "job_id",
    "account",
    "submit_time",
    "duration",
    "gpus",
    "gang_size",
    "priority",
)


@dataclass(frozen=True)
class WorkloadProfile:
    """Shape of a synthetic GPU workload.

    Defaults approximate a mixed ML research fleet: a majority of single-GPU
    interactive and fine-tuning work, a minority of multi-node training runs
    that must gang-schedule, and a heavy tail in runtime.
    """

    job_count: int = 800
    #: Mean seconds between arrivals (Poisson).
    #:
    #: 12s is not arbitrary. At the first defaults tried (300 jobs, 45s) every
    #: degenerate policy completed every job and their utilizations sat within
    #: 0.7 percentage points of each other — the fleet was never contended, so
    #: the trace could not distinguish a good scheduler from a coin flip. The
    #: convention this repo inherits says the fix for that is a harder trace,
    #: never a softer metric. At 800/12s the fleet runs ~4x oversubscribed
    #: during the arrival window and the policies separate.
    arrival_interval: float = 12.0
    #: Log-normal runtime, in log-seconds. exp(7.0) ~ 18 min median.
    runtime_mu: float = 7.0
    runtime_sigma: float = 1.2
    max_runtime: float = 6 * 3600.0
    #: GPUs per pod, weighted toward small.
    gpu_choices: tuple[int, ...] = (1, 1, 1, 1, 2, 2, 4, 8)
    #: Fraction of jobs that are multi-pod gangs.
    gang_fraction: float = 0.18
    gang_choices: tuple[int, ...] = (2, 2, 4, 8)
    accounts: tuple[str, ...] = ("research", "trading", "infra")
    account_weights: tuple[float, ...] = (0.5, 0.35, 0.15)
    priority_choices: tuple[int, ...] = (0, 0, 0, 100, 500)


#: Named profiles. ``light`` exists for CI smoke tests, where a 800-job replay
#: is wasted wall-clock; it is deliberately NOT contended and must never be used
#: to compare schedulers.
PROFILES: dict[str, WorkloadProfile] = {}


def profile_named(name: str) -> WorkloadProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


def generate(profile: WorkloadProfile | None = None, seed: int = 0) -> list[Job]:
    """Seeded synthetic trace.

    Seeded because every configuration must face an identical workload — a
    comparison against a freshly-random trace measures noise.
    """
    profile = profile or WorkloadProfile()
    rng = random.Random(seed)

    jobs: list[Job] = []
    now = 0.0
    for job_id in range(1, profile.job_count + 1):
        now += rng.expovariate(1.0 / profile.arrival_interval)
        duration = min(
            rng.lognormvariate(profile.runtime_mu, profile.runtime_sigma),
            profile.max_runtime,
        )
        gang = rng.choice(profile.gang_choices) if rng.random() < profile.gang_fraction else 1
        gpus = rng.choice(profile.gpu_choices)
        # A gang of 8-GPU pods would need a whole DGX per pod; cap so gangs stay
        # schedulable in principle, otherwise the gang metric measures the
        # generator rather than the scheduler.
        if gang > 1:
            gpus = min(gpus, 4)
        jobs.append(
            Job(
                job_id=job_id,
                account=rng.choices(profile.accounts, weights=profile.account_weights)[0],
                submit_time=now,
                duration=duration,
                gpus=gpus,
                gang_size=gang,
                priority=rng.choice(profile.priority_choices),
            )
        )
    return jobs


def write_csv(jobs: list[Job], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TRACE_COLUMNS)
        for j in jobs:
            writer.writerow(
                [j.job_id, j.account, f"{j.submit_time:.3f}", f"{j.duration:.3f}",
                 j.gpus, j.gang_size, j.priority]
            )


def read_csv(path: str | Path) -> list[Job]:
    jobs: list[Job] = []
    with Path(path).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != TRACE_COLUMNS:
            raise ValueError(
                f"{path}: header must be exactly {','.join(TRACE_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            jobs.append(
                Job(
                    job_id=int(row["job_id"]),
                    account=row["account"],
                    submit_time=float(row["submit_time"]),
                    duration=float(row["duration"]),
                    gpus=int(row["gpus"]),
                    gang_size=int(row["gang_size"]),
                    priority=int(row["priority"]),
                )
            )
    if not jobs:
        raise ValueError(f"{path}: trace is empty")
    return jobs


def summary(jobs: list[Job]) -> str:
    gangs = [j for j in jobs if j.is_gang]
    gpu_demand = sum(j.total_gpus * j.duration for j in jobs) / 3600.0
    return json.dumps(
        {
            "jobs": len(jobs),
            "gang_jobs": len(gangs),
            "pods": sum(j.gang_size for j in jobs),
            "gpu_hours_demanded": round(gpu_demand, 1),
            "max_gpus_one_job": max(j.total_gpus for j in jobs),
            "span_hours": round(max(j.submit_time for j in jobs) / 3600.0, 2),
        },
        indent=2,
    )


PROFILES.update(
    {
        "default": WorkloadProfile(),
        "light": WorkloadProfile(job_count=300, arrival_interval=45.0),
    }
)
