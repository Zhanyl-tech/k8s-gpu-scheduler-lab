"""Fragmentation Definition C: structural, queue-independent, topology-leveled.

This module is deliberately self-contained -- standard library only, no import
from the rest of the lab, no :class:`~k8slab.model.Observation`, no queue. It is
a pure function of three things every scheduler lab has:

* per-node GPU capacity,
* each node's domain at each topology level above the node, and
* a free-GPU sample series.

That is so ``slurm-scheduler-lab`` can implement the *identical* function
against its own simulator, and the two labs can be shown to agree on the same
golden vector (``tests/test_fragmentation.py``) rather than merely claimed to.

The definition
--------------

At level ``L`` (node, rack, switch), a level-``L`` domain is **carved** at time
``t`` if any node inside it has any GPU allocated (``free < capacity``). The
carved free GPUs are

    S_C^L(t) = sum of free GPUs on nodes whose level-L domain is carved

and

    frag_C^L = integral(S_C^L dt) / integral(total free GPUs dt)

-- 0 when the denominator is 0. At the node level the domain is the node itself,
so a fully allocated node contributes 0 free GPUs and an idle node contributes
nothing carved.

The samples are a step function: each sample's value holds until the next
sample's timestamp (the left-point convention definitions A and B already use in
:mod:`k8slab.metrics`). The last sample only closes the final interval; its own
values are never integrated.

Why it is the Slurm-comparable one: ``sbatch --exclusive`` jobs "can not share
nodes ... with other running jobs" (https://slurm.schedmd.com/sbatch.html, read
2026-09-26), so they need whole idle nodes, and ``frag_C^node`` is exactly the
share of free GPU-time that is *not* on a whole idle node. ``frag_C^rack`` is
the share not inside a whole idle rack -- what a job needing an entire idle
leaf switch could not use, reading the rack as a topology/tree leaf switch.
That is NOT what ``--switches`` jobs cannot use: ``--switches`` only caps how
many leaf switches an allocation spans, and nothing in the Slurm docs says such
a job needs an idle switch. Which Slurm option, if any, asks for a whole idle
leaf switch is unverified (``--exclusive=topo`` refers to "topology segments",
which the topology guide defines for block, ring and torus3d topologies). How
Slurm represents node state internally was not checked and is not claimed.
Definitions A and B answer "does some pod fit" questions that depend on the
queue or on one reference request size, which Slurm's accounting has no
equivalent of. docs/metrics.md has the full argument and the headline choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NODE_LEVEL = "node"


@dataclass(frozen=True)
class StructuralFragmentation:
    """Numerators per level, and the shared denominator, in GPU-seconds."""

    #: Levels in the order given, finest first; always starts with ``"node"``.
    levels: tuple[str, ...]
    carved_gpu_seconds: dict[str, float]
    free_gpu_seconds: float

    def rate(self, level: str) -> float:
        """``frag_C^level``; 0.0 when there was no free GPU-time at all."""
        if level not in self.carved_gpu_seconds:
            raise KeyError(f"no level {level!r}; have {list(self.levels)}")
        if self.free_gpu_seconds <= 0:
            return 0.0
        return self.carved_gpu_seconds[level] / self.free_gpu_seconds

    def rates(self) -> dict[str, float]:
        return {level: self.rate(level) for level in self.levels}


def structural_fragmentation(
    capacity: Mapping[str, int],
    domains: Mapping[str, Mapping[str, str]],
    samples: Sequence[tuple[float, Mapping[str, int]]],
) -> StructuralFragmentation:
    """Definition C at the node level and at every level in ``domains``.

    Parameters
    ----------
    capacity:
        ``{node: GPU capacity}``. Defines the node set.
    domains:
        ``{level: {node: domain id}}`` for levels above the node, **finest
        first** (e.g. ``{"rack": ..., "switch": ...}``). Every node must have a
        domain at every level, and each level must nest inside the next: all
        nodes of one rack in one switch. Nesting is what makes the result
        monotone -- ``frag_C^node <= frag_C^rack <= frag_C^switch`` -- so a
        non-nesting mapping is rejected rather than silently producing a
        ranking that can invert.
    samples:
        ``[(t, {node: free GPUs})]`` in non-decreasing time order. Every sample
        -- including the last one and any followed by a zero-length interval,
        which are never integrated -- must give every node in ``capacity``,
        and no other node, a free count in ``[0, capacity]``. Zero-length
        intervals contribute nothing.

    Raises ``ValueError`` on any violation: this function is the reference two
    labs are checked against, so it refuses malformed input instead of guessing.
    Every sample is validated before anything is integrated; an earlier version
    checked only samples that opened a positive-length interval, so a malformed
    final sample passed silently where a stricter implementation would refuse.
    """
    if NODE_LEVEL in domains:
        raise ValueError("the node level is implicit; do not pass it in domains")
    nodes = sorted(capacity)
    caps = [int(capacity[n]) for n in nodes]
    if any(c < 0 for c in caps):
        raise ValueError("capacity must be >= 0 on every node")

    # Domain ids -> dense ints per level, and the nesting check.
    level_names = tuple(domains)
    dom_idx: list[list[int]] = []
    for level in level_names:
        mapping = domains[level]
        missing = [n for n in nodes if n not in mapping]
        if missing:
            raise ValueError(f"level {level!r}: no domain for node(s) {missing[:5]}")
        extra = sorted(set(mapping) - set(capacity))
        if extra:
            raise ValueError(f"level {level!r}: domain given for unknown node(s) {extra[:5]}")
        ids: dict[str, int] = {}
        dom_idx.append([ids.setdefault(mapping[n], len(ids)) for n in nodes])
    for k in range(len(level_names) - 1):
        parent: dict[int, int] = {}
        for fine, coarse in zip(dom_idx[k], dom_idx[k + 1], strict=True):
            if parent.setdefault(fine, coarse) != coarse:
                raise ValueError(
                    f"level {level_names[k]!r} does not nest in {level_names[k + 1]!r}: "
                    f"one {level_names[k]} domain spans several {level_names[k + 1]} domains"
                )

    # Validate every sample before integrating anything (see the docstring).
    # Key checks run per sample; the range check per distinct state.
    states = [_state(t, free, nodes) for t, free in samples]
    checked: set[tuple[int, ...]] = set()
    for (t, _), vals in zip(samples, states, strict=True):
        if vals not in checked:
            _check_range(t, vals, nodes, caps)
            checked.add(vals)
    for i in range(len(samples) - 1):
        t0, t1 = samples[i][0], samples[i + 1][0]
        if t1 < t0:
            raise ValueError(f"samples out of time order at index {i}: {t0} then {t1}")

    levels = (NODE_LEVEL, *level_names)
    carved = dict.fromkeys(levels, 0.0)
    free_total = 0.0
    # Identical consecutive states are common (a sample per poll, most polls
    # change nothing), so per-state sums are memoised. Pure speed; the result
    # is the same arithmetic in the same order.
    memo: dict[tuple[int, ...], tuple[int, ...]] = {}

    for i in range(len(samples) - 1):
        dt = samples[i + 1][0] - samples[i][0]
        if dt == 0:
            continue
        vals = states[i]
        sums = memo.get(vals)
        if sums is None:
            sums = _sums(vals, caps, dom_idx)
            memo[vals] = sums
        free_total += sums[0] * dt
        for level, s in zip(levels, sums[1:], strict=True):
            carved[level] += s * dt

    return StructuralFragmentation(
        levels=levels, carved_gpu_seconds=carved, free_gpu_seconds=free_total
    )


def _state(t: float, free: Mapping[str, int], nodes: list[str]) -> tuple[int, ...]:
    """One sample's free counts in ``nodes`` order, or ``ValueError`` if the
    sample misses a node or names a node outside the fleet."""
    try:
        vals = tuple(free[n] for n in nodes)
    except KeyError as exc:
        raise ValueError(f"sample at t={t} has no free count for node {exc}") from None
    if len(free) != len(nodes):
        extra = sorted(set(free) - set(nodes))
        raise ValueError(f"sample at t={t} has a free count for unknown node(s) {extra[:5]}")
    return vals


def _check_range(t: float, vals: tuple[int, ...], nodes: list[str], caps: list[int]) -> None:
    """``ValueError`` unless every free count is in ``[0, capacity]``."""
    for n, v, c in zip(nodes, vals, caps, strict=True):
        if v < 0 or v > c:
            raise ValueError(f"sample at t={t}: node {n} free {v} outside [0, {c}]")


def _sums(
    vals: tuple[int, ...], caps: list[int], dom_idx: list[list[int]]
) -> tuple[int, ...]:
    """(total free, S_C^node, S_C^<each level>) for one validated fleet state."""
    node_carved = [v < c for v, c in zip(vals, caps, strict=True)]
    out = [sum(vals), sum(v for v, carved in zip(vals, node_carved, strict=True) if carved)]
    for idx in dom_idx:
        carved_domains = {d for d, carved in zip(idx, node_carved, strict=True) if carved}
        out.append(sum(v for v, d in zip(vals, idx, strict=True) if d in carved_domains))
    return tuple(out)
