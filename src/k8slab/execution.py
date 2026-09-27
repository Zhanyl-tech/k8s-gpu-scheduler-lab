"""Execution-layer bookkeeping shared by the reference model and the runner.

Three pieces, each small, each used by both paths so that the model and the
cluster runner account for time identically:

* :func:`harness_seed` -- the per-repeat seed of everything random in the
  harness (D-random's choices, the startup-delay draws). The *trace* seed is
  separate and fixed across repeats.
* :class:`StartupDelay` -- the bind-to-Running gap (kubelet and device-plugin
  startup, which kwok has none of). Specified in **real** milliseconds because
  that is what a kwok Stage delay is; converted to simulated seconds by the
  speedup.
* :class:`Attempt` -- one pod attempt on one node: when it was bound, when it
  started, how much trace work it still has, and the ASSUMED topology factor
  stretching it. Trace runtime counts from Running, never from bind.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass


def harness_seed(seed: int, repeat: int) -> int:
    """Seed for repeat ``repeat`` (0-based) of a benchmark started with ``seed``.

    Repeat 0 uses ``seed`` itself. So a one-repeat reference-model run with
    ``--queue-model none`` and a zero startup delay reproduces the Phase 1
    numbers exactly; under the default ``--queue-model kube`` it does not
    (the queue changes every policy's decisions), and with a non-zero delay
    the delay draws differ from Phase 1's (which had none). Later repeats
    take 32 bits of
    ``sha256("k8slab-harness|<seed>|<repeat>")``: deterministic, independent of
    ``PYTHONHASHSEED``, and unrelated to the trace seed.
    """
    if repeat < 0:
        raise ValueError("repeat must be >= 0")
    if repeat == 0:
        return seed
    digest = hashlib.sha256(f"k8slab-harness|{seed}|{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class StartupDelay:
    """Uniform bind-to-Running delay on ``[min_ms, max_ms)`` of REAL time.

    Mirrors kwok v0.8.0's Stage delay: ``durationMilliseconds`` plus
    ``Int63n(jitterDurationMilliseconds - durationMilliseconds)``, and exactly
    ``jitterDurationMilliseconds`` when it is not larger
    (pkg/utils/lifecycle/lifecycle.go, ``Stage.Delay``, read 2026-09-26).
    """

    min_ms: float = 0.0
    max_ms: float = 0.0

    def __post_init__(self) -> None:
        # Finite first: every comparison with NaN is False, so "nan" passed the
        # range check below and ran a 48 h replay whose delivered GPU-hours
        # were NaN; "0:inf" delivered nothing and exited 0, and the kwok patch
        # overflowed converting inf to an integer.
        if not (math.isfinite(self.min_ms) and math.isfinite(self.max_ms)):
            raise ValueError(
                f"startup delay must be finite milliseconds, got {self.min_ms}:{self.max_ms}"
            )
        if self.min_ms < 0 or self.max_ms < self.min_ms:
            raise ValueError(
                f"startup delay needs 0 <= MIN <= MAX, got {self.min_ms}:{self.max_ms}"
            )

    @classmethod
    def parse(cls, text: str) -> StartupDelay:
        """``"MIN:MAX"`` in milliseconds, or a single ``"N"`` meaning ``N:N``."""
        parts = text.split(":")
        if len(parts) == 1:
            value = float(parts[0])
            return cls(value, value)
        if len(parts) != 2:
            raise ValueError(f"expected MIN:MAX milliseconds, got {text!r}")
        return cls(float(parts[0]), float(parts[1]))

    @property
    def zero(self) -> bool:
        return self.max_ms == 0.0

    def text(self) -> str:
        return f"{self.min_ms:g}:{self.max_ms:g}"

    def sample_ms(self, rng: random.Random) -> float:
        """One delay in real milliseconds. Draws nothing when constant."""
        if self.max_ms <= self.min_ms:
            return self.max_ms
        return self.min_ms + rng.random() * (self.max_ms - self.min_ms)

    def sample_sim(self, rng: random.Random, speedup: float) -> float:
        """One delay in simulated seconds (0.0 exactly when zero)."""
        if self.zero:
            return 0.0
        return self.sample_ms(rng) / 1000.0 * speedup


@dataclass
class Attempt:
    """One pod attempt, and the work it has left.

    ``work`` is in trace seconds (the job's duration, less any checkpointed
    progress). While Running the attempt progresses at ``1 / factor`` trace
    seconds per simulated second, so it ends at
    ``seg_start + seg_work * factor``. With factor 1.0 that is exactly
    ``start + work`` -- the Phase 1 arithmetic, bit for bit.
    """

    key: tuple[int, int]
    node: str
    gpus: int
    priority: int
    bind_time: float
    work: float
    factor: float = 1.0
    start_time: float | None = None
    #: Beginning of the current constant-factor segment, and work left then.
    seg_start: float | None = None
    seg_work: float = 0.0
    stretched: bool = False

    @property
    def started(self) -> bool:
        return self.start_time is not None

    def start(self, t: float) -> None:
        self.start_time = t
        self.seg_start = t
        self.seg_work = self.work

    @property
    def end(self) -> float | None:
        """When the attempt will finish; ``None`` until it has started."""
        if self.seg_start is None:
            return None
        return self.seg_start + self.seg_work * self.factor

    def work_done(self, now: float) -> float:
        """Trace seconds of work completed by ``now``."""
        if self.seg_start is None or self.start_time is None or now <= self.start_time:
            return 0.0
        in_segment = min(self.seg_work, max(0.0, now - self.seg_start) / self.factor)
        return (self.work - self.seg_work) + in_segment

    def finished_by(self, now: float) -> bool:
        """The attempt's trace work was complete at or before ``now``.

        The runner notices completion only when it polls, so an attempt can be
        finished and still bound (its pod not yet deleted). Such an attempt
        has nothing left to lose or stretch: evicting it would charge Running
        time beyond its duration as lost and rerun it, and stretching it would
        move its end to ``now``.
        """
        end = self.end
        return end is not None and end <= now

    def set_factor(self, factor: float, now: float) -> None:
        """Stretch the REMAINING work by ``factor`` from ``now`` on.

        A no-op once the work is done (:meth:`finished_by`): there is no
        remaining work, and applying the factor would move ``end`` to ``now``
        -- phantom Running time booked as topology extension.
        """
        if factor == self.factor or self.finished_by(now):
            return
        self.stretched = True
        if self.seg_start is not None and now > self.seg_start:
            consumed = (now - self.seg_start) / self.factor
            self.seg_work = max(0.0, self.seg_work - consumed)
            self.seg_start = now
        self.factor = factor

    def stretch_seconds(self, until: float) -> float:
        """Wall seconds Running up to ``until`` beyond the work they did:
        the part of delivered time that exists only because of the factor."""
        if not self.stretched or self.start_time is None or until <= self.start_time:
            return 0.0
        return (until - self.start_time) - self.work_done(until)
