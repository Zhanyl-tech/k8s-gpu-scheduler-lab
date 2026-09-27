from __future__ import annotations

from pathlib import Path

import pytest

from k8slab.fleet import load, render_nodes
from k8slab.model import Fleet, NodeClass

FLEETS = Path(__file__).resolve().parent.parent / "fleets"


def test_default_fleet_is_heterogeneous() -> None:
    """The default fleet has more than one node shape, so request shapes and
    node shapes mismatch (fleets/default.yaml). Not because identical nodes
    cannot fragment: under a mixed trace they do (docs/metrics.md, "The
    control")."""
    fleet = load(FLEETS / "default.yaml")
    assert len({c.gpus for c in fleet.classes}) > 1


def test_default_fleet_totals() -> None:
    fleet = load(FLEETS / "default.yaml")
    assert fleet.total_nodes == 26
    assert fleet.total_gpus == 140


def test_node_names_are_unique_and_resolvable() -> None:
    fleet = load(FLEETS / "default.yaml")
    names = fleet.node_names()
    assert len(names) == len(set(names)) == fleet.total_nodes
    assert sum(fleet.gpus_of(n) for n in names) == fleet.total_gpus


def test_rendered_nodes_advertise_gpus_and_are_kwok_managed() -> None:
    fleet = load(FLEETS / "default.yaml")
    nodes = render_nodes(fleet)
    assert len(nodes) == fleet.total_nodes
    for node in nodes:
        assert node["metadata"]["annotations"]["kwok.x-k8s.io/node"] == "fake"
        assert int(node["status"]["allocatable"]["nvidia.com/gpu"]) >= 0
    total = sum(int(n["status"]["capacity"]["nvidia.com/gpu"]) for n in nodes)
    assert total == fleet.total_gpus


def test_homogeneous_control_fleet_has_one_shape() -> None:
    fleet = load(FLEETS / "homogeneous.yaml")
    assert len({c.gpus for c in fleet.classes}) == 1


def test_rejects_empty_and_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nnodeClasses: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)

    bad.write_text("name: x\nnodeClasses:\n  - name: a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)


def test_node_class_validates() -> None:
    with pytest.raises(ValueError):
        NodeClass(name="a", count=0, gpus=8)
    with pytest.raises(ValueError):
        NodeClass(name="a", count=1, gpus=-1)


def test_unknown_node_raises() -> None:
    fleet = Fleet("t", (NodeClass("a", 1, 8),))
    with pytest.raises(KeyError):
        fleet.gpus_of("nope-0")


@pytest.mark.parametrize("path", sorted(FLEETS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_fleet_renders_to_valid_nodes(path: Path) -> None:
    """A fleet that renders badly breaks `make up` with an opaque kubectl error."""
    import yaml

    from k8slab.fleet import render_yaml

    docs = [d for d in yaml.safe_load_all(render_yaml(load(path))) if d]
    assert docs
    for doc in docs:
        assert doc["kind"] == "Node"
        assert "nvidia.com/gpu" in doc["status"]["capacity"]
        assert doc["metadata"]["annotations"]["kwok.x-k8s.io/node"] == "fake"


# ---- topology keys ------------------------------------------------------------


def test_shipped_fleets_declare_topology() -> None:
    for name in ("default", "homogeneous"):
        fleet = load(FLEETS / f"{name}.yaml")
        assert fleet.topology_declared
        assert fleet.racks_per_switch == 2
        assert all(c.nodes_per_rack for c in fleet.classes)
    default = load(FLEETS / "default.yaml")
    assert {c.name: c.nvlink for c in default.classes} == {
        "dgx8": True, "mid4": False, "edge2": False
    }


def test_fleet_without_topology_still_loads_with_defaults(tmp_path: Path) -> None:
    f = tmp_path / "plain.yaml"
    f.write_text("name: p\nnodeClasses:\n  - {name: a, count: 2, gpus: 8}\n", encoding="utf-8")
    fleet = load(f)
    assert not fleet.topology_declared
    assert fleet.racks_per_switch is None
    assert fleet.classes[0].nvlink is False and fleet.classes[0].nodes_per_rack is None


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("nodeClasses:\n  - {name: a, count: 2, gpus: 8, nodesPerRak: 2}\n", "unknown key"),
        ("nodeClasses:\n  - {name: a, count: 2, gpus: 8, nvlink: 'yes'}\n", "nvlink"),
        ("nodeClasses:\n  - {name: a, count: 2, gpus: 8, nodesPerRack: 0}\n", "nodesPerRack"),
        ("topology: {racksPerSwitch: 0}\nnodeClasses:\n  - {name: a, count: 2, gpus: 8}\n",
         "racksPerSwitch"),
        ("topology: {racks: 2}\nnodeClasses:\n  - {name: a, count: 2, gpus: 8}\n",
         "unknown key"),
        ("topology: [1]\nnodeClasses:\n  - {name: a, count: 2, gpus: 8}\n", "mapping"),
        # A misspelt top-level block used to load as "no topology declared".
        ("topolgy: {racksPerSwitch: 1}\nnodeClasses:\n  - {name: a, count: 2, gpus: 8}\n",
         "unknown top-level key"),
    ],
)
def test_rejects_malformed_topology(tmp_path: Path, body: str, match: str) -> None:
    """A misspelt topology key must fail loudly, not fall back to a default
    layout that every placement number would then silently describe."""
    f = tmp_path / "bad.yaml"
    f.write_text("name: x\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load(f)


def test_max_node_gpus() -> None:
    assert load(FLEETS / "default.yaml").max_node_gpus == 8
    assert Fleet("t", (NodeClass("a", 1, 2), NodeClass("b", 1, 4))).max_node_gpus == 4
