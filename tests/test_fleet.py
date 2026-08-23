from __future__ import annotations

from pathlib import Path

import pytest

from k8slab.fleet import load, render_nodes
from k8slab.model import Fleet, NodeClass

FLEETS = Path(__file__).resolve().parent.parent / "fleets"


def test_default_fleet_is_heterogeneous() -> None:
    """Fragmentation is unmeasurable on a fleet of identical nodes."""
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
