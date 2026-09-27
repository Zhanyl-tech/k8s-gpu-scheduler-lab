"""Load a fleet description from YAML, and render it as kwok Node objects.

Topology keys (all optional; see docs/metrics.md, "Topology and placement"):

* per nodeClass: ``nvlink`` (bool), ``nodesPerRack`` (int >= 1);
* fleet level: ``topology: {racksPerSwitch: int >= 1}``.

Every rack and switch is derived from these by :func:`k8slab.topology.derive`,
deterministically. They describe a *scenario*: kwok nodes have no fabric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import topology as topology_mod
from .model import Fleet, NodeClass

_CLASS_KEYS = {"name", "count", "gpus", "cpus", "memoryGi", "nvlink", "nodesPerRack"}
_CLASS_TOPOLOGY_KEYS = {"nvlink", "nodesPerRack"}
_TOPOLOGY_KEYS = {"racksPerSwitch"}
_TOP_KEYS = {"name", "nodeClasses", "topology"}

#: kwok only manages nodes carrying this annotation, so a real node in the same
#: cluster (kind's control plane) is left alone.
KWOK_MANAGED = {"kwok.x-k8s.io/node": "fake"}


def load(path: str | Path) -> Fleet:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    # Checked like the nodeClass and topology keys below. A misspelt
    # `topolgy:` block used to load silently as "no topology declared", and
    # every rack and switch number then described the flat default layout.
    unknown_top = set(raw) - _TOP_KEYS
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown_top)}")

    name = str(raw.get("name") or Path(path).stem)
    classes_raw = raw.get("nodeClasses")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError(f"{path}: 'nodeClasses' must be a non-empty list")

    declared = False
    classes: list[NodeClass] = []
    for entry in classes_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each nodeClass must be a mapping")
        missing = {"name", "count", "gpus"} - set(entry)
        if missing:
            raise ValueError(f"{path}: nodeClass missing {sorted(missing)}")
        # Unknown keys are rejected rather than ignored: a misspelt
        # `nodesPerRak` would otherwise silently fall back to a default
        # topology and every placement number would describe the wrong fleet.
        unknown = set(entry) - _CLASS_KEYS
        if unknown:
            raise ValueError(f"{path}: nodeClass has unknown key(s) {sorted(unknown)}")
        nvlink = entry.get("nvlink", False)
        if not isinstance(nvlink, bool):
            raise ValueError(f"{path}: nodeClass {entry['name']!r}: nvlink must be true/false")
        per_rack = entry.get("nodesPerRack")
        declared = declared or bool(_CLASS_TOPOLOGY_KEYS & set(entry))
        classes.append(
            NodeClass(
                name=str(entry["name"]),
                count=int(entry["count"]),
                gpus=int(entry["gpus"]),
                cpus=int(entry.get("cpus", 32)),
                memory_gi=int(entry.get("memoryGi", 256)),
                nvlink=nvlink,
                nodes_per_rack=None if per_rack is None else int(per_rack),
            )
        )

    seen = {c.name for c in classes}
    if len(seen) != len(classes):
        raise ValueError(f"{path}: duplicate nodeClass names")

    racks_per_switch: int | None = None
    topo_raw = raw.get("topology")
    if topo_raw is not None:
        if not isinstance(topo_raw, dict):
            raise ValueError(f"{path}: 'topology' must be a mapping")
        unknown = set(topo_raw) - _TOPOLOGY_KEYS
        if unknown:
            raise ValueError(f"{path}: topology has unknown key(s) {sorted(unknown)}")
        if "racksPerSwitch" in topo_raw:
            racks_per_switch = int(topo_raw["racksPerSwitch"])
        declared = True

    return Fleet(
        name=name,
        classes=tuple(classes),
        racks_per_switch=racks_per_switch,
        topology_declared=declared,
    )


def render_nodes(fleet: Fleet) -> list[dict[str, Any]]:
    """Produce kwok-managed Node manifests advertising ``nvidia.com/gpu``.

    The GPUs are advertised, not present. Nothing in this repo runs CUDA; see
    docs/limitations.md.

    Topology travels as labels under the lab's own ``topology.k8slab.io/``
    prefix (rack, switch, nvlink). No NVIDIA or ``kubernetes.io`` label is
    invented for it: these values are a declared scenario, and a borrowed key
    would imply they were discovered. Label-defined levels are the shape Kueue
    Topology-Aware Scheduling consumes -- a ``Topology`` lists node-label keys
    from widest to narrowest -- so K1 can use them as they are; see the
    citations in :mod:`k8slab.topology`.
    """
    topo = topology_mod.derive(fleet)
    nodes: list[dict[str, Any]] = []
    for cls in fleet.classes:
        for i in range(cls.count):
            capacity = {
                "cpu": str(cls.cpus),
                "memory": f"{cls.memory_gi}Gi",
                "pods": "256",
                "nvidia.com/gpu": str(cls.gpus),
            }
            nodes.append(
                {
                    "apiVersion": "v1",
                    "kind": "Node",
                    "metadata": {
                        "name": f"{cls.name}-{i}",
                        "annotations": {
                            **KWOK_MANAGED,
                            "node.alpha.kubernetes.io/ttl": "0",
                        },
                        "labels": {
                            "type": "kwok",
                            "k8slab.io/fleet": fleet.name,
                            topology_mod.LABEL_NODE_CLASS: cls.name,
                            "nvidia.com/gpu.count": str(cls.gpus),
                            "kubernetes.io/hostname": f"{cls.name}-{i}",
                            **topology_mod.node_labels(topo.node(f"{cls.name}-{i}")),
                        },
                    },
                    "spec": {
                        # kwok nodes are tainted so nothing real lands on them
                        # by accident; the harness tolerates this taint.
                        "taints": [
                            {
                                "key": "kwok.x-k8s.io/node",
                                "value": "fake",
                                "effect": "NoSchedule",
                            }
                        ]
                    },
                    "status": {
                        "allocatable": dict(capacity),
                        "capacity": dict(capacity),
                        "nodeInfo": {"kubeletVersion": "fake"},
                        "phase": "Running",
                    },
                }
            )
    return nodes


def render_yaml(fleet: Fleet) -> str:
    docs = render_nodes(fleet)
    return "\n---\n".join(yaml.safe_dump(d, sort_keys=False) for d in docs)
