"""Fleet topology, and a measurement of how schedulers place jobs on it.

What this module measures, and what it only assumes
---------------------------------------------------

Two different things live here and they must not be confused.

**Measured:** *where* a scheduler put a job's pods. Every node has a rack and a
switch (derived deterministically from the fleet file by :func:`derive`), so the
widest domain a job's pods span -- one node, one rack, one switch, or several
switches -- is a fact about the binding decisions, read from the same
observation every other metric reads. A topology-aware scheduler should keep
multi-pod jobs in narrow domains; whether it does is a legitimate measurement of
its placement, and :func:`placement_quality` reports it for every run.

**Assumed:** what that placement would *cost*. :class:`PenaltyFactors` maps a
placement tier to a slowdown factor (default 1.0 single NVLink node, 1.2 single
non-NVLink node, 1.4 intra-rack, 1.8 cross-rack same switch, 2.2 cross-switch).
**These numbers are scenario parameters, not measurements.** The lab has no
fabric and sends no NCCL traffic -- see docs/limitations.md -- so nothing here
can tell whether a cross-switch job actually ran 2.2x slower, or slower at all.
The factors exist so that an execution layer can make placement *matter* inside
a replay under a stated assumption (:func:`placement_factor` is the hook), and
so that one run can report a single GPU-weighted placement cost under those
same stated assumptions. Any number derived from them inherits the assumption
and must be quoted with it.

Node labels
-----------

:func:`node_labels` emits ``topology.k8slab.io/{switch,rack,nvlink}``. The prefix
is the lab's own, deliberately: this is a synthetic topology on kwok nodes, and
borrowing a vendor's or ``kubernetes.io``'s label keys would assert a provenance
the values do not have. Label-defined topology levels are exactly the shape
Kueue Topology-Aware Scheduling consumes: a ``Topology`` object
(``kueue.x-k8s.io/v1beta2`` at the lab's planned Kueue pin, v0.19.2) lists
``spec.levels[].nodeLabel`` "from the widest (block) to the narrowest
(hostname)", and TAS computes free capacity per domain from those labels.
Verified against
https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/ ,
https://kueue.sigs.k8s.io/docs/tasks/manage/setup_topology_aware_scheduling/
and the v0.19.2 example
https://raw.githubusercontent.com/kubernetes-sigs/kueue/v0.19.2/site/static/examples/tas/sample-queues.yaml
(read 2026-09-26). A future K1 can therefore point a ``Topology`` at
``[topology.k8slab.io/switch, topology.k8slab.io/rack, kubernetes.io/hostname]``
without the fleet changing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields

from .model import Fleet, NodeClass

LABEL_PREFIX = "topology.k8slab.io"
LABEL_SWITCH = f"{LABEL_PREFIX}/switch"
LABEL_RACK = f"{LABEL_PREFIX}/rack"
LABEL_NVLINK = f"{LABEL_PREFIX}/nvlink"

#: Placement tiers, narrowest first. A job's tier is the widest domain its pods
#: span.
TIER_NODE = "node"
TIER_RACK = "rack"
TIER_SWITCH = "switch"
TIER_CROSS_SWITCH = "cross-switch"
TIERS: tuple[str, ...] = (TIER_NODE, TIER_RACK, TIER_SWITCH, TIER_CROSS_SWITCH)


@dataclass(frozen=True)
class NodeTopology:
    """Where one node sits. Every field is derived or declared, never probed."""

    name: str
    node_class: str
    gpus: int
    nvlink: bool
    rack: str
    switch: str


@dataclass(frozen=True)
class Topology:
    """Every node's domains, keyed by node name."""

    nodes: dict[str, NodeTopology]
    #: Copied from :attr:`Fleet.topology_declared`; False means defaults.
    declared: bool = False

    def node(self, name: str) -> NodeTopology:
        try:
            return self.nodes[name]
        except KeyError:
            raise KeyError(
                f"node {name!r} is not in the fleet topology; a pod bound to a node "
                f"outside the fleet means the run replayed against the wrong cluster"
            ) from None

    def capacity(self) -> dict[str, int]:
        """GPU capacity per node -- the first argument of Definition C."""
        return {n: t.gpus for n, t in self.nodes.items()}

    def domains(self) -> dict[str, dict[str, str]]:
        """``{level: {node: domain}}`` for the levels above the node, finest
        first -- the second argument of
        :func:`k8slab.fragmentation.structural_fragmentation`."""
        return {
            "rack": {n: t.rack for n, t in self.nodes.items()},
            "switch": {n: t.switch for n, t in self.nodes.items()},
        }

    @property
    def racks(self) -> list[str]:
        return sorted({t.rack for t in self.nodes.values()}, key=_ordinal)

    @property
    def switches(self) -> list[str]:
        return sorted({t.switch for t in self.nodes.values()}, key=_ordinal)


def _ordinal(domain: str) -> tuple[int, int, str]:
    """Sort ``rack-2`` before ``rack-10``; anything else after, by name."""
    head, _, tail = domain.rpartition("-")
    if head and tail.isdigit():
        return (0, int(tail), domain)
    return (1, 0, domain)


def derive(fleet: Fleet) -> Topology:
    """Assign every node a rack and a switch, deterministically.

    The rule, in full:

    1. Node classes are walked in fleet-file order; nodes within a class in
       ordinal order (``<class>-0``, ``<class>-1``, ...), the same order as
       :meth:`Fleet.node_names`.
    2. Each class fills its own racks, ``nodesPerRack`` nodes at a time (the
       last rack of a class may be short). **Racks never mix classes.** Racks
       are numbered globally across the fleet: ``rack-0``, ``rack-1``, ...
    3. Consecutive racks are grouped ``racksPerSwitch`` at a time into
       ``switch-0``, ``switch-1``, ... (the last switch may be short). A switch
       *may* mix classes, because the grouping runs across class boundaries.

    Defaults when the fleet file declares nothing: one rack per class, and
    every rack under one switch. Those are the flattest assumptions available
    -- they never invent a cross-switch hop -- and the fleet records that they
    were defaults (:attr:`Topology.declared` is False).

    Racks nest in switches by construction, which is what makes Definition C
    monotone across levels (docs/metrics.md).
    """
    nodes: dict[str, NodeTopology] = {}
    rack_of: list[tuple[str, NodeClass, int]] = []
    next_rack = 0
    for cls in fleet.classes:
        per_rack = cls.nodes_per_rack or cls.count
        for i in range(cls.count):
            rack_of.append((f"{cls.name}-{i}", cls, next_rack + i // per_rack))
        next_rack += -(-cls.count // per_rack)  # ceil division

    total_racks = next_rack
    per_switch = fleet.racks_per_switch or max(1, total_racks)
    for name, cls, rack in rack_of:
        nodes[name] = NodeTopology(
            name=name,
            node_class=cls.name,
            gpus=cls.gpus,
            nvlink=cls.nvlink,
            rack=f"rack-{rack}",
            switch=f"switch-{rack // per_switch}",
        )
    return Topology(nodes=nodes, declared=fleet.topology_declared)


#: The node-class label ``fleet.render_nodes`` already sets.
LABEL_NODE_CLASS = "k8slab.io/node-class"


def from_labels(
    labels: Mapping[str, Mapping[str, str]],
    capacity: Mapping[str, int],
) -> Topology:
    """Rebuild a :class:`Topology` from labels read back off Node objects.

    For an execution layer that wants placement scored against the topology
    the scheduler actually saw (set it as ``Observation.topology``). ``labels``
    is ``{node: metadata.labels}``, ``capacity`` is ``{node: GPUs}``; every
    node in ``capacity`` must carry the rack and switch labels, or this raises
    rather than guessing. The result is ``declared=True``: the labels exist.
    """
    nodes: dict[str, NodeTopology] = {}
    for name, gpus in capacity.items():
        node_labels_ = labels.get(name)
        if node_labels_ is None:
            raise ValueError(f"no labels for node {name!r}")
        missing = [k for k in (LABEL_RACK, LABEL_SWITCH) if k not in node_labels_]
        if missing:
            raise ValueError(f"node {name!r} lacks topology label(s) {missing}")
        nodes[name] = NodeTopology(
            name=name,
            node_class=node_labels_.get(LABEL_NODE_CLASS, ""),
            gpus=int(gpus),
            nvlink=node_labels_.get(LABEL_NVLINK) == "true",
            rack=node_labels_[LABEL_RACK],
            switch=node_labels_[LABEL_SWITCH],
        )
    return Topology(nodes=nodes, declared=True)


def node_labels(node: NodeTopology) -> dict[str, str]:
    """Kubernetes labels carrying one node's topology, under the lab prefix."""
    return {
        LABEL_SWITCH: node.switch,
        LABEL_RACK: node.rack,
        LABEL_NVLINK: "true" if node.nvlink else "false",
    }


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobPlacement:
    """The nodes one job's pods were bound to: one entry per pod."""

    nodes: tuple[str, ...]
    gpus_per_pod: int

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a placement needs at least one pod")

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_pod * len(self.nodes)

    @property
    def multi_pod(self) -> bool:
        return len(self.nodes) > 1


def placement_tier(placement: JobPlacement, topo: Topology) -> str:
    """The widest domain the job's pods span: node, rack, switch or cross-switch."""
    placed = [topo.node(n) for n in placement.nodes]
    if len({t.name for t in placed}) == 1:
        return TIER_NODE
    if len({t.rack for t in placed}) == 1:
        return TIER_RACK
    if len({t.switch for t in placed}) == 1:
        return TIER_SWITCH
    return TIER_CROSS_SWITCH


@dataclass(frozen=True)
class PenaltyFactors:
    """ASSUMED slowdown per placement tier. Scenario parameters, not measurements.

    Nothing in this lab measures any of these numbers: kwok nodes have no GPUs
    and no fabric, and no collective traffic is ever sent. They encode one
    stated assumption -- communication-bound work slows as its pods spread
    across wider network domains -- so that a replay can be *run under* that
    assumption, and so that placement quality can be summarised as one number
    whose premise is on the page.

    The ordering is the only part with physical grounding (a hop through more
    switches is never cheaper); the magnitudes are illustrative. In particular
    ``node_no_nvlink`` (1.2) sits between a single NVLink node and an
    intra-rack placement on the reasoning that GPU-to-GPU traffic inside one
    host over PCIe avoids the NIC and switch but lacks NVLink bandwidth. That
    placement relative to ``rack`` is itself an assumption; the validator
    deliberately does not enforce it.
    """

    #: Single node with NVLink, and more than one GPU.
    node_nvlink: float = 1.0
    #: Single node without NVLink, and more than one GPU.
    node_no_nvlink: float = 1.2
    #: Several nodes, one rack.
    rack: float = 1.4
    #: Several racks, one switch.
    switch: float = 1.8
    #: Several switches.
    cross_switch: float = 2.2

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"penalty factor {f.name} must be a positive number")
        if not self.node_nvlink <= self.rack <= self.switch <= self.cross_switch:
            raise ValueError(
                "penalty factors must not decrease as placement widens: "
                "node_nvlink <= rack <= switch <= cross_switch"
            )
        if self.node_no_nvlink < self.node_nvlink:
            raise ValueError("node_no_nvlink must be >= node_nvlink")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, float]) -> PenaltyFactors:
        """Build from a scenario mapping whose keys are this class's field names."""
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown penalty factor(s) {sorted(unknown)}; have {sorted(known)}")
        return cls(**{k: float(v) for k, v in raw.items()})


DEFAULT_FACTORS = PenaltyFactors()


def placement_factor(
    placement: JobPlacement,
    topo: Topology,
    factors: PenaltyFactors = DEFAULT_FACTORS,
) -> float:
    """ASSUMED slowdown of one job given where its pods were bound.

    Pure: no clock, no queue, no observation. This is the hook an execution
    layer calls when a job's placement is complete (every pod bound) to decide
    how much longer it would run under the scenario's assumptions. A job with
    at most one GPU in total has no inter-GPU traffic and always returns 1.0.
    """
    if placement.total_gpus <= 1:
        return 1.0
    tier = placement_tier(placement, topo)
    if tier == TIER_NODE:
        nvlink = topo.node(placement.nodes[0]).nvlink
        return factors.node_nvlink if nvlink else factors.node_no_nvlink
    if tier == TIER_RACK:
        return factors.rack
    if tier == TIER_SWITCH:
        return factors.switch
    return factors.cross_switch


@dataclass(frozen=True)
class PlacementQuality:
    """How narrowly a run's jobs were placed. Always computed, always reported."""

    #: Multi-pod jobs whose every pod was bound -- the denominator of the shares.
    multi_pod_jobs: int
    #: Share of those jobs at each tier. Every tier key is always present; with
    #: no multi-pod job every share is ``None`` -- 0/0 is undefined, and 0.0
    #: would both read as "never placed at this tier" and be averaged as a
    #: real zero across repeats (the Metrics "None, never zero" contract).
    tier_share: dict[str, float | None]
    #: Jobs with more than one GPU in total whose every pod was bound.
    communicating_jobs: int
    #: GPU-weighted mean of :func:`placement_factor` over communicating jobs:
    #: ``sum(G_j * f_j) / sum(G_j)``. ``None`` when there are none. Inherits the
    #: ASSUMED factors; quote it with them.
    penalty_mean: float | None


def placement_quality(
    placements: Iterable[JobPlacement],
    topo: Topology,
    factors: PenaltyFactors = DEFAULT_FACTORS,
) -> PlacementQuality:
    """Tier shares over multi-pod jobs, and the GPU-weighted mean factor.

    Single-GPU jobs are excluded from the mean because they cannot
    communicate: including them would pull every configuration towards 1.0 by
    the share of single-GPU work in the trace, which is a property of the trace
    and not of the scheduler.
    """
    counts = dict.fromkeys(TIERS, 0)
    multi = 0
    weighted = 0.0
    weight = 0
    communicating = 0
    for p in placements:
        if p.multi_pod:
            counts[placement_tier(p, topo)] += 1
            multi += 1
        if p.total_gpus > 1:
            communicating += 1
            weighted += p.total_gpus * placement_factor(p, topo, factors)
            weight += p.total_gpus
    return PlacementQuality(
        multi_pod_jobs=multi,
        tier_share={t: (counts[t] / multi if multi else None) for t in TIERS},
        communicating_jobs=communicating,
        penalty_mean=(weighted / weight) if weight else None,
    )
