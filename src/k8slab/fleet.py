"""Load a fleet description from YAML, and render it as kwok Node objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import Fleet, NodeClass

#: kwok only manages nodes carrying this annotation, so a real node in the same
#: cluster (kind's control plane) is left alone.
KWOK_MANAGED = {"kwok.x-k8s.io/node": "fake"}


def load(path: str | Path) -> Fleet:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    name = str(raw.get("name") or Path(path).stem)
    classes_raw = raw.get("nodeClasses")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError(f"{path}: 'nodeClasses' must be a non-empty list")

    classes: list[NodeClass] = []
    for entry in classes_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each nodeClass must be a mapping")
        missing = {"name", "count", "gpus"} - set(entry)
        if missing:
            raise ValueError(f"{path}: nodeClass missing {sorted(missing)}")
        classes.append(
            NodeClass(
                name=str(entry["name"]),
                count=int(entry["count"]),
                gpus=int(entry["gpus"]),
                cpus=int(entry.get("cpus", 32)),
                memory_gi=int(entry.get("memoryGi", 256)),
            )
        )

    seen = {c.name for c in classes}
    if len(seen) != len(classes):
        raise ValueError(f"{path}: duplicate nodeClass names")

    return Fleet(name=name, classes=tuple(classes))


def render_nodes(fleet: Fleet) -> list[dict[str, Any]]:
    """Produce kwok-managed Node manifests advertising ``nvidia.com/gpu``.

    The GPUs are advertised, not present. Nothing in this repo runs CUDA; see
    docs/limitations.md.
    """
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
                            "k8slab.io/node-class": cls.name,
                            "nvidia.com/gpu.count": str(cls.gpus),
                            "kubernetes.io/hostname": f"{cls.name}-{i}",
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
