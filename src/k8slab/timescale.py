"""Controller-horizon synchronisation under time compression.

The problem this module exists to solve
---------------------------------------

The replay divides every trace timestamp by ``--speedup`` (default 60), so one
real second of wall-clock is 60 simulated seconds. Scheduling *decisions* are
not time-dependent at that resolution, but several control-plane mechanisms are,
and they are all specified in **real** seconds:

* kube-scheduler's unschedulable-pod backoff (1 s initial, 10 s ceiling),
* the time a pod may sit in the unschedulable pool before it is retried anyway
  (5 min),
* leader-election lease duration, renew deadline and retry period,
* the scheduling queue's own flush timers (1 s and 30 s),
* client-go rate-limiter windows and controller resync periods.

At ``--speedup 60`` a 10 s real backoff ceiling is **10 minutes of simulated
trace time**. A pod that fails a few fit attempts is then excluded from the
queue for ten simulated minutes, during which the trace has moved on. That is
not the behaviour of a real cluster at any speed -- it is an artefact of
compression, and it inflates pending queues and makespan for exactly the
configurations that use a real scheduler (K0, and later K1-K4).

The fix is to divide every real-time control-plane constant by the same factor
the trace was divided by, so that *simulated* durations match what an
uncompressed cluster would experience -- as far as the configuration schema
lets us.

What was verified, and where (read 2026-09-26, Kubernetes v1.32.2 source)
------------------------------------------------------------------------

* ``podInitialBackoffSeconds`` / ``podMaxBackoffSeconds`` are ``*int64`` whole
  seconds in ``kubescheduler.config.k8s.io/v1``, defaulting to 1 and 10, and
  **validation rejects** ``podInitialBackoffSeconds <= 0`` ("must be greater
  than 0") and ``podMaxBackoffSeconds < podInitialBackoffSeconds``.
  https://github.com/kubernetes/kubernetes/blob/v1.32.2/staging/src/k8s.io/kube-scheduler/config/v1/types.go
  https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/apis/config/v1/defaults.go
  https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/apis/config/validation/validation.go
  https://kubernetes.io/docs/reference/config-api/kube-scheduler-config.v1/
  An earlier draft of this module floored both to 0 at any speedup above 1.
  kube-scheduler would have refused to start on that file. The smallest valid
  value is 1 s real, i.e. ``speedup`` simulated seconds, and that is what is
  emitted; the residual is reported.
* The unschedulable-pool timeout is **not** a configuration field. It is the
  deprecated flag ``--pod-max-in-unschedulable-pods-duration`` (default 5m),
  still honoured in 1.32 even when ``--config`` is given
  (cmd/kube-scheduler/app/options/deprecated.go and options.go). It is a Go
  duration, so the flag *value* scales exactly (to the millisecond). The
  *behaviour* does not: see the next point.
* The queue flushes backoffQ every 1 s and the unschedulable pool every 30 s
  (``PriorityQueue.Run``: ``wait.Until(... flushBackoffQCompleted ...,
  1.0*time.Second ...)`` and ``wait.Until(... flushUnschedulablePodsLeftover
  ..., 30*time.Second ...)``, pkg/scheduler/backend/queue/scheduling_queue.go
  lines 358-364 at v1.32.2). These are hard-coded: **they cannot be scaled**,
  so at speedup S they run every S and 30*S simulated seconds. And they gate
  the two scaled values. The timeout is only checked inside the 30 s flush
  (``currentTime.Sub(lastScheduleTime) > p.podMaxInUnschedulablePodsDuration``,
  lines 834-848), so a pod no event releases leaves the pool 5-35 s real after
  it was parked under the scaled 5 s flag: (300, 2100] simulated seconds at
  speedup 60, against (300, 330] uncompressed. A backed-off pod leaves backoffQ
  only on a 1 s flush tick at or after its expiry (lines 804-830), so a 1 s
  real backoff is [60, 120) simulated seconds at speedup 60, against [1, 2)
  for the uncompressed initial backoff and [10, 11) for its ceiling.
  :meth:`TimeScale.summary` records these ranges; the reference model's kube
  queue uses the same simulated intervals.
  https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/backend/queue/scheduling_queue.go
* With ``--config``, kube-scheduler ignores ``--kubeconfig``,
  ``--kube-api-qps`` and ``--kube-api-burst`` ("This parameter is ignored if a
  config file is specified in --config"), but still applies leader-election
  flags on top of the file (``ApplyLeaderElectionTo``). kubeadm always passes
  ``--leader-elect=true`` and ``--kubeconfig=/etc/kubernetes/scheduler.conf``
  (cmd/kubeadm/app/phases/controlplane/manifests.go). So the file must carry
  ``clientConnection.kubeconfig`` itself, and disabling leader election needs
  the ``leader-elect: "false"`` flag, not only the file field. See
  :func:`scheduler_extra_args`.
* ``MaxInFlightMovePods``: the optimisation plan names such a field. It does
  **not** exist in the v1 ``KubeSchedulerConfiguration`` (not in types.go at
  v1.32.2, not on the reference page). It is not emitted.

Integer quantisation, stated rather than hidden
----------------------------------------------

The backoff fields cannot express 1/60 s. :meth:`TimeScale.summary` records,
in simulated seconds, what the trace *should* experience (the uncompressed
values), what it *will* experience, and the difference. At speedup 60 the
initial backoff is 60 simulated seconds instead of 1 and the ceiling 60 instead
of 10: a residual of 59 s and 50 s -- against 600 s of ceiling if nothing were
scaled at all. Those are the *configured* values; the unscalable flush ticks
then add up to one tick on top (``experienced_range_sim_seconds`` and
``worst_case_residual_sim_seconds`` in the summary): up to +119 s on the
initial backoff, +110 s on the ceiling and +1800 s on the unschedulable-pool
timeout at speedup 60. That is the difference between "we scaled the
controller horizons" and "we scaled them to within N simulated seconds, and
here is N".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import yaml

#: Upstream defaults, in **real** seconds, for the mechanisms compression
#: distorts. Verified against the v1.32.2 sources listed in the module docstring.
DEFAULT_POD_INITIAL_BACKOFF = 1.0
DEFAULT_POD_MAX_BACKOFF = 10.0
DEFAULT_POD_MAX_IN_UNSCHEDULABLE = 300.0  # --pod-max-in-unschedulable-pods-duration=5m
DEFAULT_LEASE_DURATION = 15.0
DEFAULT_RENEW_DEADLINE = 10.0
DEFAULT_RETRY_PERIOD = 2.0
#: Hard-coded queue timers (``wait.Until`` periods). Not configurable.
BACKOFF_FLUSH_INTERVAL = 1.0
UNSCHEDULABLE_FLUSH_INTERVAL = 30.0

#: kube-scheduler's default ``parallelism`` (v1 defaults.go).
DEFAULT_PARALLELISM = 16

#: Where the generated file is mounted inside the kind control-plane node and
#: the kube-scheduler static pod (cluster/kind.yaml).
CONFIG_DIR_IN_NODE = "/etc/kubernetes/k8slab"
CONFIG_PATH_IN_NODE = f"{CONFIG_DIR_IN_NODE}/scheduler-config.yaml"
#: kubeadm's scheduler kubeconfig (``--kubeconfig`` in manifests.go), which
#: kube-scheduler ignores once ``--config`` is set.
SCHEDULER_KUBECONFIG = "/etc/kubernetes/scheduler.conf"


def _whole_seconds(value: float) -> int:
    """Nearest whole second, rounding halves up, never below 1 (the schema's
    smallest valid backoff)."""
    return max(1, math.floor(value + 0.5))


@dataclass(frozen=True)
class TimeScale:
    """A compression factor and the scaled control-plane horizons it implies.

    ``speedup`` is the same number passed to ``--speedup``. Values named
    ``*_seconds`` without ``_sim`` are what gets configured on the cluster
    (real seconds); ``*_sim`` values are what the trace therefore experiences.
    """

    speedup: float

    def __post_init__(self) -> None:
        # `not > 0` already refuses NaN; infinity would scale every horizon to 0.
        if not (self.speedup > 0 and math.isfinite(self.speedup)):
            raise ValueError(f"speedup must be finite and > 0, got {self.speedup}")

    def scale(self, real_seconds: float) -> float:
        """Divide a real-time constant by the compression factor."""
        return real_seconds / self.speedup

    def to_sim(self, real_seconds: float) -> float:
        """Real seconds on the cluster -> simulated seconds in the trace."""
        return real_seconds * self.speedup

    # -- exact, sub-second-capable (Go duration) values ------------------------

    @property
    def lease_duration(self) -> float:
        return self.scale(DEFAULT_LEASE_DURATION)

    @property
    def renew_deadline(self) -> float:
        return self.scale(DEFAULT_RENEW_DEADLINE)

    @property
    def retry_period(self) -> float:
        return self.scale(DEFAULT_RETRY_PERIOD)

    @property
    def pod_max_in_unschedulable(self) -> str:
        """The scaled ``--pod-max-in-unschedulable-pods-duration`` flag value."""
        return _go_duration(self.scale(DEFAULT_POD_MAX_IN_UNSCHEDULABLE))

    # -- integer-quantised fields -------------------------------------------

    @property
    def pod_initial_backoff_seconds(self) -> int:
        """Scaled initial backoff, rounded to the integer schema, at least 1.

        At any ``speedup >= 2`` this is 1: the scaled target is below the
        smallest value validation accepts.
        """
        return _whole_seconds(self.scale(DEFAULT_POD_INITIAL_BACKOFF))

    @property
    def pod_max_backoff_seconds(self) -> int:
        """Scaled backoff ceiling, rounded, never below the initial backoff
        (validation rejects a ceiling below the initial value)."""
        return max(
            self.pod_initial_backoff_seconds,
            _whole_seconds(self.scale(DEFAULT_POD_MAX_BACKOFF)),
        )

    # -- what the trace experiences, in simulated seconds --------------------

    @property
    def initial_backoff_sim(self) -> float:
        return self.to_sim(self.pod_initial_backoff_seconds)

    @property
    def max_backoff_sim(self) -> float:
        return self.to_sim(self.pod_max_backoff_seconds)

    @property
    def max_in_unschedulable_sim(self) -> float:
        return self.to_sim(_parse_go_duration(self.pod_max_in_unschedulable))

    @property
    def backoff_flush_interval_sim(self) -> float:
        """Unscalable: the backoffQ flush runs every 1 s of REAL time."""
        return self.to_sim(BACKOFF_FLUSH_INTERVAL)

    @property
    def unschedulable_flush_interval_sim(self) -> float:
        """Unscalable: the unschedulable-pool flush runs every 30 s REAL."""
        return self.to_sim(UNSCHEDULABLE_FLUSH_INTERVAL)

    def residual_quantisation_error(self) -> float:
        """Worst-case backoff error this configuration still carries.

        ``|backoff ceiling the trace experiences - uncompressed ceiling|`` in
        **simulated** seconds, directly comparable to trace durations and waits.
        0.0 at speedup 1.
        """
        return abs(self.max_backoff_sim - DEFAULT_POD_MAX_BACKOFF)

    def initial_backoff_residual(self) -> float:
        """Same as :meth:`residual_quantisation_error`, for the initial backoff."""
        return abs(self.initial_backoff_sim - DEFAULT_POD_INITIAL_BACKOFF)

    def experienced_ranges(self) -> dict[str, tuple[float, float]]:
        """What the trace experiences once the unscalable flush ticks gate the
        configured values, in simulated seconds, as ``(low, high)``.

        A pod released from the unschedulable pool by an event while still
        backing off waits in backoffQ for the first 1 s flush tick at or after
        its expiry: ``[backoff, backoff + tick)``. A pod no event releases
        leaves the pool on the first 30 s tick at which it has been parked
        strictly longer than the timeout: ``(timeout, timeout + tick]``. The
        tick phase is arbitrary, so any value in the range can occur.
        """
        bf, uf = self.backoff_flush_interval_sim, self.unschedulable_flush_interval_sim
        return {
            "initial_backoff": (self.initial_backoff_sim, self.initial_backoff_sim + bf),
            "max_backoff": (self.max_backoff_sim, self.max_backoff_sim + bf),
            "max_in_unschedulable": (
                round(self.max_in_unschedulable_sim, 6),
                round(self.max_in_unschedulable_sim + uf, 6),
            ),
        }

    def worst_case_residuals(self) -> dict[str, float]:
        """Upper end of each experienced range minus the uncompressed
        configured value, in simulated seconds: how much later than an
        uncompressed kube-scheduler's configured value a pod can be released.
        (An uncompressed cluster has flush lag too -- up to 1 s and 30 s --
        which ``uncompressed_range_sim_seconds`` records.)"""
        r = self.experienced_ranges()
        return {
            "initial_backoff": round(r["initial_backoff"][1] - DEFAULT_POD_INITIAL_BACKOFF, 6),
            "max_backoff": round(r["max_backoff"][1] - DEFAULT_POD_MAX_BACKOFF, 6),
            "max_in_unschedulable": round(
                r["max_in_unschedulable"][1] - DEFAULT_POD_MAX_IN_UNSCHEDULABLE, 6
            ),
        }

    def summary(self) -> dict[str, Any]:
        """Machine-readable record of what was scaled, for results.json.

        ``experienced_sim_seconds`` are the configured values in simulated
        seconds; ``experienced_range_sim_seconds`` adds the flush-tick gating
        (half-open as in :meth:`experienced_ranges`: ``[low, high)`` for the
        backoffs, ``(low, high]`` for the timeout).
        """
        ranges = self.experienced_ranges()
        return {
            "speedup": self.speedup,
            "configured_real_seconds": {
                "podInitialBackoffSeconds": self.pod_initial_backoff_seconds,
                "podMaxBackoffSeconds": self.pod_max_backoff_seconds,
                "pod-max-in-unschedulable-pods-duration": self.pod_max_in_unschedulable,
                "leaderElection": "disabled",
            },
            "experienced_sim_seconds": {
                "initial_backoff": self.initial_backoff_sim,
                "max_backoff": self.max_backoff_sim,
                "max_in_unschedulable": round(self.max_in_unschedulable_sim, 6),
                "backoff_flush_interval": self.backoff_flush_interval_sim,
                "unschedulable_flush_interval": self.unschedulable_flush_interval_sim,
            },
            "uncompressed_sim_seconds": {
                "initial_backoff": DEFAULT_POD_INITIAL_BACKOFF,
                "max_backoff": DEFAULT_POD_MAX_BACKOFF,
                "max_in_unschedulable": DEFAULT_POD_MAX_IN_UNSCHEDULABLE,
                "backoff_flush_interval": BACKOFF_FLUSH_INTERVAL,
                "unschedulable_flush_interval": UNSCHEDULABLE_FLUSH_INTERVAL,
            },
            "experienced_range_sim_seconds": {k: list(v) for k, v in ranges.items()},
            "uncompressed_range_sim_seconds": {
                "initial_backoff": [
                    DEFAULT_POD_INITIAL_BACKOFF,
                    DEFAULT_POD_INITIAL_BACKOFF + BACKOFF_FLUSH_INTERVAL,
                ],
                "max_backoff": [
                    DEFAULT_POD_MAX_BACKOFF, DEFAULT_POD_MAX_BACKOFF + BACKOFF_FLUSH_INTERVAL
                ],
                "max_in_unschedulable": [
                    DEFAULT_POD_MAX_IN_UNSCHEDULABLE,
                    DEFAULT_POD_MAX_IN_UNSCHEDULABLE + UNSCHEDULABLE_FLUSH_INTERVAL,
                ],
            },
            # Quantisation residuals of the integer backoff fields alone.
            "backoff_residual_error_sim_seconds": round(self.residual_quantisation_error(), 6),
            "initial_backoff_residual_sim_seconds": round(self.initial_backoff_residual(), 6),
            # Integer quantisation plus flush-tick gating, worst case.
            "worst_case_residual_sim_seconds": self.worst_case_residuals(),
            # What the ceiling would have been with no scaling at all.
            "unscaled_backoff_ceiling_sim_seconds": self.to_sim(DEFAULT_POD_MAX_BACKOFF),
            "unscalable": [
                "backoffQ flush period (1 s real, hard-coded)",
                "unschedulable-pool flush period (30 s real, hard-coded)",
                "client-go rate limiter windows",
            ],
        }


def _go_duration(seconds: float) -> str:
    """Render seconds as a Go duration string kube-scheduler will parse.

    Sub-second values become milliseconds; Go's ``time.ParseDuration`` accepts
    ``250ms`` where a bare ``0.25`` is not a duration. Values are rounded to
    whole milliseconds and never rendered as ``0s``, which would disable the
    mechanism rather than shorten it.
    """
    millis = max(1, int(round(seconds * 1000)))
    if millis % 1000 == 0:
        return f"{millis // 1000}s"
    return f"{millis}ms"


def _parse_go_duration(text: str) -> float:
    """Inverse of :func:`_go_duration` for the two forms it emits."""
    if text.endswith("ms"):
        return int(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(int(text[:-1]))
    raise ValueError(f"unsupported duration {text!r}")


def scheduler_config(
    scale: TimeScale,
    *,
    serialise: bool = False,
    leader_elect: bool = False,
) -> dict[str, Any]:
    """Build a ``KubeSchedulerConfiguration`` with compression-scaled horizons.

    Parameters
    ----------
    scale:
        The compression factor in force for this run.
    serialise:
        Set ``parallelism: 1``. **Off by default, deliberately, and a
        diagnostic only** (``--serialise``). ``parallelism`` is the number of
        workers kube-scheduler uses *inside* one scheduling cycle (filtering
        and scoring nodes for one pod); pods are already popped one per cycle,
        and binding stays asynchronous. So it does not serialise pods, and it
        does not make kube-scheduler deterministic -- ties between equally
        scored nodes are broken at random (``selectHost`` in schedule_one.go,
        v1.32.2). What it does is change scheduler throughput, i.e. change the
        thing being measured. Rows produced with it are marked gated and kept
        out of headline comparisons; repetition with a reported spread
        (:mod:`k8slab.stats`) is how this lab handles non-determinism.
    leader_elect:
        Keep leader election, with scaled lease timings. A single-replica lab
        scheduler does not need it; turning it off removes one wall-clock
        mechanism entirely rather than approximating it. NOTE: kubeadm passes
        ``--leader-elect=true``, which overrides this file, so the flag must be
        overridden too (:func:`scheduler_extra_args`).

    Only fields that exist in ``kubescheduler.config.k8s.io/v1`` are emitted.
    """
    cfg: dict[str, Any] = {
        "apiVersion": "kubescheduler.config.k8s.io/v1",
        "kind": "KubeSchedulerConfiguration",
        # Compression-scaled, and at least 1: validation rejects 0. The
        # residual is recorded (TimeScale.summary), not ignored.
        "podInitialBackoffSeconds": scale.pod_initial_backoff_seconds,
        "podMaxBackoffSeconds": scale.pod_max_backoff_seconds,
        "parallelism": 1 if serialise else DEFAULT_PARALLELISM,
        # Score every node. Valid range [0, 100]. The fleet (26 nodes) is below
        # kube-scheduler's minimum of 100 feasible nodes to find anyway, so this
        # states the behaviour rather than changing it.
        "percentageOfNodesToScore": 100,
        "clientConnection": {
            # Required here: --kubeconfig is ignored once --config is given.
            "kubeconfig": SCHEDULER_KUBECONFIG,
            # Moved from the kube-api-qps/burst flags, ignored under --config.
            "qps": 200,
            "burst": 400,
        },
        "leaderElection": {
            "leaderElect": leader_elect,
        },
    }

    if leader_elect:
        cfg["leaderElection"].update(
            {
                "leaseDuration": _go_duration(scale.lease_duration),
                "renewDeadline": _go_duration(scale.renew_deadline),
                "retryPeriod": _go_duration(scale.retry_period),
            }
        )

    return cfg


def scheduler_extra_args(scale: TimeScale, *, leader_elect: bool = False) -> dict[str, str]:
    """Flags kube-scheduler needs next to the file (kubeadm ``extraArgs``)."""
    return {
        "config": CONFIG_PATH_IN_NODE,
        # kubeadm's --leader-elect=true would override the file's leaderElect.
        "leader-elect": "true" if leader_elect else "false",
        # Not a config field; the deprecated flag is the only knob in 1.32.
        "pod-max-in-unschedulable-pods-duration": scale.pod_max_in_unschedulable,
    }


def scheduler_config_yaml(scale: TimeScale, **kwargs: Any) -> str:
    return yaml.safe_dump(scheduler_config(scale, **kwargs), sort_keys=False)
