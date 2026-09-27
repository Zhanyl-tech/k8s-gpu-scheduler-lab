"""Controller-horizon scaling (k8slab.timescale) against the v1.32.2 schema."""

from __future__ import annotations

import pytest
import yaml

from k8slab.timescale import (
    CONFIG_PATH_IN_NODE,
    SCHEDULER_KUBECONFIG,
    TimeScale,
    _go_duration,
    scheduler_config,
    scheduler_config_yaml,
    scheduler_extra_args,
)

#: Top-level fields of kubescheduler.config.k8s.io/v1 KubeSchedulerConfiguration
#: at v1.32.2 (staging/src/k8s.io/kube-scheduler/config/v1/types.go), plus the
#: inlined DebuggingConfiguration fields and TypeMeta.
V1_FIELDS = {
    "apiVersion", "kind", "parallelism", "leaderElection", "clientConnection",
    "enableProfiling", "enableContentionProfiling", "percentageOfNodesToScore",
    "podInitialBackoffSeconds", "podMaxBackoffSeconds", "profiles", "extenders",
    "delayCacheUntilActive",
}


@pytest.mark.parametrize("speedup", [0.5, 1.0, 2.0, 3.0, 60.0, 400.0, 1e6])
def test_backoff_fields_always_pass_kube_scheduler_validation(speedup: float) -> None:
    """validation.go: podInitialBackoffSeconds must be > 0, and
    podMaxBackoffSeconds >= podInitialBackoffSeconds. An earlier draft emitted 0,
    which kube-scheduler refuses to start with."""
    s = TimeScale(speedup)
    assert s.pod_initial_backoff_seconds >= 1
    assert s.pod_max_backoff_seconds >= s.pod_initial_backoff_seconds
    cfg = scheduler_config(s)
    assert isinstance(cfg["podInitialBackoffSeconds"], int)
    assert cfg["podInitialBackoffSeconds"] > 0


def test_uncompressed_is_exact_and_carries_no_residual() -> None:
    s = TimeScale(1.0)
    assert (s.pod_initial_backoff_seconds, s.pod_max_backoff_seconds) == (1, 10)
    assert s.residual_quantisation_error() == 0.0 and s.initial_backoff_residual() == 0.0
    assert s.pod_max_in_unschedulable == "300s"  # the upstream 5m default


def test_speedup_60_residuals_are_what_the_docs_say() -> None:
    s = TimeScale(60.0)
    assert (s.initial_backoff_sim, s.max_backoff_sim) == (60.0, 60.0)
    assert s.residual_quantisation_error() == 50.0  # 60 s experienced vs 10 s wanted
    assert s.initial_backoff_residual() == 59.0  # 60 s vs 1 s
    summary = s.summary()
    assert summary["backoff_residual_error_sim_seconds"] == 50.0
    assert summary["unscaled_backoff_ceiling_sim_seconds"] == 600.0
    assert summary["experienced_sim_seconds"]["backoff_flush_interval"] == 60.0
    assert summary["experienced_sim_seconds"]["unschedulable_flush_interval"] == 1800.0


def test_rounding_to_the_nearest_whole_second() -> None:
    s = TimeScale(4.0)  # 1/4 -> 1 (floor at the minimum), 10/4 = 2.5 -> 3
    assert (s.pod_initial_backoff_seconds, s.pod_max_backoff_seconds) == (1, 3)


def test_unschedulable_timeout_scales_exactly_as_a_go_duration() -> None:
    assert TimeScale(60.0).pod_max_in_unschedulable == "5s"
    assert TimeScale(400.0).pod_max_in_unschedulable == "750ms"
    assert TimeScale(400.0).max_in_unschedulable_sim == pytest.approx(300.0)


def test_go_durations_never_collapse_to_zero() -> None:
    assert _go_duration(0.0) == "1ms"
    assert _go_duration(0.25) == "250ms"
    assert _go_duration(2.0) == "2s"


def test_config_uses_only_v1_fields_and_no_invented_ones() -> None:
    for serialise in (False, True):
        cfg = scheduler_config(TimeScale(60.0), serialise=serialise, leader_elect=True)
        assert set(cfg) <= V1_FIELDS
    text = scheduler_config_yaml(TimeScale(60.0))
    # The plan named MaxInFlightMovePods; it is not in the v1 schema.
    assert "MaxInFlight" not in text and "maxInFlight" not in text
    assert yaml.safe_load(text)["apiVersion"] == "kubescheduler.config.k8s.io/v1"


def test_config_carries_the_kubeconfig_the_flag_would_have() -> None:
    """--kubeconfig is ignored under --config; without this field the scheduler
    would have no way to reach the API server."""
    cfg = scheduler_config(TimeScale(60.0))
    assert cfg["clientConnection"] == {"kubeconfig": SCHEDULER_KUBECONFIG, "qps": 200,
                                       "burst": 400}
    assert cfg["leaderElection"] == {"leaderElect": False}
    assert cfg["percentageOfNodesToScore"] == 100


def test_serialise_is_parallelism_one_and_nothing_else() -> None:
    a = scheduler_config(TimeScale(60.0))
    b = scheduler_config(TimeScale(60.0), serialise=True)
    assert (a["parallelism"], b["parallelism"]) == (16, 1)
    assert {k: v for k, v in a.items() if k != "parallelism"} == {
        k: v for k, v in b.items() if k != "parallelism"
    }


def test_leader_election_timings_scale_when_kept() -> None:
    cfg = scheduler_config(TimeScale(60.0), leader_elect=True)
    assert cfg["leaderElection"]["leaseDuration"] == "250ms"
    assert cfg["leaderElection"]["retryPeriod"] == "33ms"


def test_extra_args_override_kubeadms_leader_elect_and_point_at_the_file() -> None:
    args = scheduler_extra_args(TimeScale(60.0))
    assert args == {
        "config": CONFIG_PATH_IN_NODE,
        "leader-elect": "false",
        "pod-max-in-unschedulable-pods-duration": "5s",
    }


def test_speedup_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TimeScale(0.0)


def test_flush_ticks_gate_the_configured_values() -> None:
    """scheduling_queue.go v1.32.2: backoffQ is flushed every 1 s and the
    unschedulable pool every 30 s (real), and the timeout is only checked in
    that 30 s flush. At speedup 60 a 1 s backoff is [60, 120) simulated s and
    the scaled 5 s timeout releases a pod after (300, 2100] simulated s."""
    s = TimeScale(60.0)
    assert s.experienced_ranges() == {
        "initial_backoff": (60.0, 120.0),
        "max_backoff": (60.0, 120.0),
        "max_in_unschedulable": (300.0, 2100.0),
    }
    assert s.worst_case_residuals() == {
        "initial_backoff": 119.0, "max_backoff": 110.0, "max_in_unschedulable": 1800.0,
    }
    summary = s.summary()
    assert summary["experienced_range_sim_seconds"]["max_in_unschedulable"] == [300.0, 2100.0]
    assert summary["uncompressed_range_sim_seconds"] == {
        "initial_backoff": [1.0, 2.0], "max_backoff": [10.0, 11.0],
        "max_in_unschedulable": [300.0, 330.0],
    }
    assert summary["worst_case_residual_sim_seconds"]["max_in_unschedulable"] == 1800.0
    # Uncompressed, only the ticks' own lag remains.
    assert TimeScale(1.0).worst_case_residuals() == {
        "initial_backoff": 1.0, "max_backoff": 1.0, "max_in_unschedulable": 30.0,
    }
