from __future__ import annotations

import re
from pathlib import Path

import pytest

from k8slab.fleet import load, render_nodes
from k8slab.model import Fleet, NodeClass
from k8slab.topology import (
    DEFAULT_FACTORS,
    LABEL_NVLINK,
    LABEL_RACK,
    LABEL_SWITCH,
    TIER_CROSS_SWITCH,
    TIER_NODE,
    TIER_RACK,
    TIER_SWITCH,
    TIERS,
    JobPlacement,
    PenaltyFactors,
    derive,
    from_labels,
    placement_factor,
    placement_quality,
    placement_tier,
)

FLEETS = Path(__file__).resolve().parent.parent / "fleets"
DEFAULT = derive(load(FLEETS / "default.yaml"))


def test_default_fleet_derivation_is_the_documented_one() -> None:
    """fleets/default.yaml's comment spells this layout out; keep them in step."""
    layout: dict[str, list[str]] = {}
    for node in DEFAULT.nodes.values():
        layout.setdefault(node.rack, []).append(node.name)
    assert DEFAULT.racks == [f"rack-{i}" for i in range(6)]
    assert [len(layout[r]) for r in DEFAULT.racks] == [4, 4, 4, 4, 4, 6]
    assert layout["rack-2"] == ["dgx8-8", "dgx8-9", "dgx8-10", "dgx8-11"]
    switch_of = {n.rack: n.switch for n in DEFAULT.nodes.values()}
    assert switch_of == {
        "rack-0": "switch-0", "rack-1": "switch-0",
        "rack-2": "switch-1", "rack-3": "switch-1",
        "rack-4": "switch-2", "rack-5": "switch-2",
    }
    assert DEFAULT.declared


def test_racks_never_mix_classes_and_nest_in_one_switch() -> None:
    for path in sorted(FLEETS.glob("*.yaml")):
        topo = derive(load(path))
        classes: dict[str, set[str]] = {}
        switches: dict[str, set[str]] = {}
        for node in topo.nodes.values():
            classes.setdefault(node.rack, set()).add(node.node_class)
            switches.setdefault(node.rack, set()).add(node.switch)
        assert all(len(c) == 1 for c in classes.values()), path.name
        assert all(len(s) == 1 for s in switches.values()), path.name


def test_derivation_is_deterministic_and_covers_every_node() -> None:
    fleet = load(FLEETS / "default.yaml")
    assert derive(fleet) == derive(fleet)
    assert list(derive(fleet).nodes) == fleet.node_names()


def test_homogeneous_control_keeps_one_node_shape_with_topology() -> None:
    fleet = load(FLEETS / "homogeneous.yaml")
    topo = derive(fleet)
    assert {n.gpus for n in topo.nodes.values()} == {8}
    assert len(topo.racks) == 5 and len(topo.switches) == 3


def test_undeclared_topology_uses_the_flat_defaults() -> None:
    fleet = Fleet("flat", (NodeClass("a", 3, 8), NodeClass("b", 2, 4)))
    topo = derive(fleet)
    assert not topo.declared
    assert topo.racks == ["rack-0", "rack-1"]  # one rack per class
    assert topo.switches == ["switch-0"]  # one switch: no invented cross-switch hop
    assert not any(n.nvlink for n in topo.nodes.values())


def test_short_last_rack_and_switch() -> None:
    fleet = Fleet("x", (NodeClass("a", 5, 8, nodes_per_rack=2),), racks_per_switch=2)
    topo = derive(fleet)
    assert [topo.node(f"a-{i}").rack for i in range(5)] == [
        "rack-0", "rack-0", "rack-1", "rack-1", "rack-2"
    ]
    assert topo.node("a-4").switch == "switch-1"


# ---- tiers and factors ------------------------------------------------------


@pytest.mark.parametrize(
    ("nodes", "tier"),
    [
        (("dgx8-0", "dgx8-0"), TIER_NODE),
        (("dgx8-0", "dgx8-3"), TIER_RACK),
        (("dgx8-0", "dgx8-4"), TIER_SWITCH),  # rack-0 and rack-1, both switch-0
        (("dgx8-8", "mid4-0"), TIER_SWITCH),  # rack-2 and rack-3, both switch-1
        (("dgx8-0", "dgx8-8"), TIER_CROSS_SWITCH),
        (("dgx8-0", "dgx8-1", "edge2-5"), TIER_CROSS_SWITCH),  # widest span wins
    ],
)
def test_placement_tier_is_the_widest_domain_spanned(nodes: tuple[str, ...], tier: str) -> None:
    assert placement_tier(JobPlacement(nodes, gpus_per_pod=2), DEFAULT) == tier


def test_default_factors_are_the_documented_assumptions() -> None:
    f = DEFAULT_FACTORS
    assert (f.node_nvlink, f.node_no_nvlink, f.rack, f.switch, f.cross_switch) == (
        1.0, 1.2, 1.4, 1.8, 2.2
    )
    cases = {
        ("dgx8-0",): 1.0,  # one NVLink node, 4 GPUs
        ("mid4-0",): 1.2,  # one PCIe node, 4 GPUs
        ("dgx8-0", "dgx8-1"): 1.4,
        ("dgx8-0", "dgx8-4"): 1.8,
        ("dgx8-0", "mid4-7"): 2.2,
    }
    for nodes, factor in cases.items():
        assert placement_factor(JobPlacement(nodes, gpus_per_pod=4), DEFAULT) == factor


def test_single_gpu_jobs_have_no_communication_penalty() -> None:
    custom = PenaltyFactors(node_nvlink=1.1, node_no_nvlink=1.3)
    assert placement_factor(JobPlacement(("mid4-0",), 1), DEFAULT, custom) == 1.0


def test_factors_are_configurable_and_validated() -> None:
    f = PenaltyFactors.from_mapping({"rack": 1.5, "switch": 2.0, "cross_switch": 3.0})
    assert placement_factor(JobPlacement(("dgx8-0", "dgx8-8"), 1), DEFAULT, f) == 3.0
    with pytest.raises(ValueError, match="unknown"):
        PenaltyFactors.from_mapping({"crossSwitch": 3.0})
    with pytest.raises(ValueError, match="must not decrease"):
        PenaltyFactors(rack=2.0, switch=1.5)
    with pytest.raises(ValueError, match="positive"):
        PenaltyFactors(node_nvlink=0.0)
    with pytest.raises(ValueError, match="node_no_nvlink"):
        PenaltyFactors(node_no_nvlink=0.9)


def test_placement_quality_shares_and_gpu_weighted_penalty() -> None:
    placements = [
        JobPlacement(("dgx8-0", "dgx8-0"), 4),  # node, NVLink: 8 GPUs x 1.0
        JobPlacement(("dgx8-0", "dgx8-1"), 2),  # rack: 4 GPUs x 1.4
        JobPlacement(("dgx8-0", "dgx8-8"), 1),  # cross-switch: 2 GPUs x 2.2
        JobPlacement(("mid4-0",), 4),  # single pod, PCIe: 4 GPUs x 1.2; not multi-pod
        JobPlacement(("edge2-0",), 1),  # one GPU: excluded from the mean
    ]
    q = placement_quality(placements, DEFAULT)
    assert q.multi_pod_jobs == 3
    assert q.tier_share == pytest.approx(
        {TIER_NODE: 1 / 3, TIER_RACK: 1 / 3, TIER_SWITCH: 0.0, TIER_CROSS_SWITCH: 1 / 3}
    )
    assert q.communicating_jobs == 4
    assert q.penalty_mean == pytest.approx((8 * 1.0 + 4 * 1.4 + 2 * 2.2 + 4 * 1.2) / 18)


def test_placement_quality_of_nothing_is_undefined_not_one() -> None:
    q = placement_quality([JobPlacement(("dgx8-0",), 1)], DEFAULT)
    assert q.multi_pod_jobs == 0 and q.penalty_mean is None
    # 0/0 is undefined, not 0%: every tier present, every share None, so an
    # aggregate over repeats skips this run instead of averaging in zeros.
    assert set(q.tier_share) == set(TIERS)
    assert all(v is None for v in q.tier_share.values())
    assert placement_quality([], DEFAULT).tier_share == dict.fromkeys(TIERS)


def test_bad_placements_are_refused() -> None:
    with pytest.raises(ValueError):
        JobPlacement((), 1)
    with pytest.raises(KeyError, match="outside the fleet"):
        placement_tier(JobPlacement(("ghost-0", "dgx8-0"), 1), DEFAULT)


# ---- node labels ------------------------------------------------------------

#: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set
_NAME = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
_DNS_SUBDOMAIN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$")


@pytest.mark.parametrize("path", sorted(FLEETS.glob("*.yaml")), ids=lambda p: p.name)
def test_rendered_nodes_carry_lab_prefixed_topology_labels(path: Path) -> None:
    fleet = load(path)
    topo = derive(fleet)
    for node in render_nodes(fleet):
        labels = node["metadata"]["labels"]
        name = node["metadata"]["name"]
        assert labels[LABEL_RACK] == topo.node(name).rack
        assert labels[LABEL_SWITCH] == topo.node(name).switch
        assert labels[LABEL_NVLINK] in {"true", "false"}
        for key, value in labels.items():
            prefix, _, short = key.rpartition("/")
            assert len(prefix) <= 253 and (not prefix or _DNS_SUBDOMAIN.match(prefix)), key
            assert _NAME.match(short), key
            assert value == "" or _NAME.match(value), (key, value)


def test_no_vendor_or_kubernetes_labels_were_invented() -> None:
    """Topology uses the lab's own prefix. The only nvidia.com / kubernetes.io
    keys are the two Phase 1 already emitted, both of which exist upstream."""
    for node in render_nodes(load(FLEETS / "default.yaml")):
        foreign = {
            k for k in node["metadata"]["labels"]
            if "nvidia.com" in k.split("/")[0] or "kubernetes.io" in k.split("/")[0]
        }
        assert foreign == {"nvidia.com/gpu.count", "kubernetes.io/hostname"}
        assert {k for k in node["metadata"]["labels"] if "topology" in k} == {
            LABEL_RACK, LABEL_SWITCH, LABEL_NVLINK
        }
    assert LABEL_RACK.startswith("topology.k8slab.io/")


def test_topology_round_trips_through_rendered_node_labels() -> None:
    """What an execution layer reads back off the Node objects rebuilds exactly
    the topology the fleet file declared."""
    for path in sorted(FLEETS.glob("*.yaml")):
        fleet = load(path)
        nodes = render_nodes(fleet)
        labels = {n["metadata"]["name"]: n["metadata"]["labels"] for n in nodes}
        capacity = {
            n["metadata"]["name"]: int(n["status"]["capacity"]["nvidia.com/gpu"]) for n in nodes
        }
        assert from_labels(labels, capacity) == derive(fleet), path.name


def test_from_labels_refuses_unlabelled_nodes() -> None:
    with pytest.raises(ValueError, match="no labels"):
        from_labels({}, {"a-0": 8})
    with pytest.raises(ValueError, match="lacks topology"):
        from_labels({"a-0": {LABEL_RACK: "r"}}, {"a-0": 8})


def test_domains_sort_numerically_and_tolerate_foreign_names() -> None:
    labels = {
        "x": {LABEL_RACK: "rack-10", LABEL_SWITCH: "spine"},
        "y": {LABEL_RACK: "rack-2", LABEL_SWITCH: "spine"},
        "z": {LABEL_RACK: "odd", LABEL_SWITCH: "spine"},
    }
    topo = from_labels(labels, {"x": 1, "y": 1, "z": 1})
    assert topo.racks == ["rack-2", "rack-10", "odd"]
    assert topo.switches == ["spine"]
