from __future__ import annotations

from pathlib import Path

import pytest

from k8slab.trace import (
    PROFILES,
    TRACE_COLUMNS,
    WorkloadProfile,
    generate,
    profile_named,
    read_csv,
    write_csv,
)


def test_generation_is_seeded() -> None:
    """Configurations must face an identical workload or the comparison is noise."""
    a = generate(seed=7)
    b = generate(seed=7)
    assert [(j.job_id, j.gpus, j.duration) for j in a] == [
        (j.job_id, j.gpus, j.duration) for j in b
    ]


def test_different_seeds_differ() -> None:
    assert [j.duration for j in generate(seed=1)] != [j.duration for j in generate(seed=2)]


def test_trace_contains_gangs() -> None:
    """The gang scenario is the failure mode Kubernetes has and Slurm does not."""
    jobs = generate(seed=0)
    gangs = [j for j in jobs if j.is_gang]
    assert gangs, "a trace with no gang job cannot exercise gang scheduling"
    assert all(j.gang_size > 1 for j in gangs)


def test_gang_pods_stay_placeable() -> None:
    """A gang pod larger than the biggest node would measure the generator."""
    for job in generate(seed=0):
        if job.is_gang:
            assert job.gpus <= 4


def test_csv_roundtrip(tmp_path: Path) -> None:
    jobs = generate(WorkloadProfile(job_count=25), seed=3)
    path = tmp_path / "t.csv"
    write_csv(jobs, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(TRACE_COLUMNS)
    back = read_csv(path)
    assert [j.job_id for j in back] == [j.job_id for j in jobs]
    assert [j.gang_size for j in back] == [j.gang_size for j in jobs]


def test_rejects_foreign_header(tmp_path: Path) -> None:
    """Phase 2's sacct translator must emit this exact schema."""
    path = tmp_path / "bad.csv"
    path.write_text("job_id,account\n1,research\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(path)


def test_default_profile_is_contended() -> None:
    """A trace that every policy sails through cannot rank schedulers.

    Guards the finding that motivated the current defaults: at 300 jobs / 45 s
    the degenerate policies were within 0.7 points of each other.
    """
    default = profile_named("default")
    assert default.job_count >= 500
    assert default.arrival_interval <= 20


def test_light_profile_exists_and_is_not_contended() -> None:
    light = profile_named("light")
    assert light.job_count < profile_named("default").job_count
    assert set(PROFILES) >= {"default", "light"}


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        profile_named("nope")
