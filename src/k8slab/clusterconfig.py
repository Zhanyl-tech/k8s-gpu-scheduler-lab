"""Files ``make up`` generates for one cluster, and the record ``make bench`` checks.

Everything time-dependent that the control plane reads at start-up is fixed
when the cluster is created: kube-scheduler reads its configuration file and
flags once, and kwok's pod-ready Stage delay is a cluster object. So the
generated files and the settings they encode belong to the *cluster*, not to a
benchmark invocation. ``make up`` writes them into ``cluster/generated/``
(gitignored) together with ``cluster-state.json``; ``k8slab bench`` against a
cluster reads that record and refuses to run if its own ``--speedup`` or
``--startup-delay`` differs, because the scheduler would then be scaled for a
different compression than the trace is replayed at.

Generated files:

* ``scheduler-config.yaml`` -- :func:`k8slab.timescale.scheduler_config`.
* ``kind.yaml`` -- ``cluster/kind.yaml`` plus the one speedup-dependent flag,
  ``pod-max-in-unschedulable-pods-duration``, which is not a configuration
  field and so cannot live in the mounted file.
* ``pod-ready-delay.patch.yaml`` -- only when a startup delay is set: a JSON
  merge patch adding ``spec.delay`` to kwok's ``pod-ready`` Stage. kwok v0.8.0
  Stage delay fields (``durationMilliseconds``, ``jitterDurationMilliseconds``;
  the delay is ``duration + Int63n(jitter - duration)``) verified against
  https://github.com/kubernetes-sigs/kwok/blob/v0.8.0/pkg/apis/v1alpha1/stage_types.go
  and pkg/utils/lifecycle/lifecycle.go, read 2026-09-26. A patch rather than a
  copied Stage, so the pinned upstream ``pod-ready`` template stays upstream's.

NONE of this has been exercised on a running cluster: no kind or Docker daemon
was available when it was written. ``cluster/upstream.lock`` records the
combination as unverified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .execution import StartupDelay
from .timescale import (
    CONFIG_DIR_IN_NODE,
    TimeScale,
    scheduler_config_yaml,
    scheduler_extra_args,
)

STATE_FILE = "cluster-state.json"
SCHEDULER_FILE = "scheduler-config.yaml"
KIND_FILE = "kind.yaml"
DELAY_PATCH_FILE = "pod-ready-delay.patch.yaml"
STATE_SCHEMA = 1


@dataclass(frozen=True)
class ClusterState:
    """What the cluster was brought up with."""

    speedup: float
    serialise: bool
    startup_delay_ms: tuple[float, float]
    timescale: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "speedup": self.speedup,
            "serialise": self.serialise,
            "startup_delay_ms": list(self.startup_delay_ms),
            "timescale": self.timescale,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ClusterState:
        if raw.get("schema") != STATE_SCHEMA:
            raise ValueError(f"unknown cluster-state schema {raw.get('schema')!r}")
        low, high = raw["startup_delay_ms"]
        return cls(
            speedup=float(raw["speedup"]),
            serialise=bool(raw["serialise"]),
            startup_delay_ms=(float(low), float(high)),
            timescale=dict(raw.get("timescale") or {}),
        )


class _BlockDumper(yaml.SafeDumper):
    """Multi-line strings (the embedded kubeadm patch) as ``|`` blocks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _represent_str)


def render_kind_config(template: str, scale: TimeScale) -> str:
    """``cluster/kind.yaml`` with the speedup-dependent scheduler flag added.

    Checks that the template already mounts the generated directory and points
    kube-scheduler at the generated file, so the two cannot drift apart.
    """
    doc = yaml.safe_load(template)
    node = doc["nodes"][0]
    mounts = node.get("extraMounts") or []
    if not any(m.get("containerPath") == CONFIG_DIR_IN_NODE for m in mounts):
        raise ValueError(f"kind template does not mount {CONFIG_DIR_IN_NODE}")
    patches = node.get("kubeadmConfigPatches") or []
    for i, patch in enumerate(patches):
        inner = yaml.safe_load(patch)
        if not isinstance(inner, dict) or inner.get("kind") != "ClusterConfiguration":
            continue
        scheduler = inner.setdefault("scheduler", {})
        args = scheduler.setdefault("extraArgs", {})
        wanted = scheduler_extra_args(scale)
        for flag in ("config", "leader-elect"):
            if args.get(flag) != wanted[flag]:
                raise ValueError(
                    f"kind template scheduler extraArgs[{flag!r}] is {args.get(flag)!r}, "
                    f"expected {wanted[flag]!r}"
                )
        args.update(wanted)
        volumes = scheduler.get("extraVolumes") or []
        if not any(v.get("mountPath") == CONFIG_DIR_IN_NODE for v in volumes):
            raise ValueError("kind template does not mount the config into kube-scheduler")
        patches[i] = yaml.safe_dump(inner, sort_keys=False)
        break
    else:
        raise ValueError("kind template has no ClusterConfiguration patch")
    header = (
        "# GENERATED by `k8slab cluster-config write` from cluster/kind.yaml for\n"
        f"# --speedup {scale.speedup:g}. Do not edit; edit cluster/kind.yaml.\n"
    )
    return header + yaml.dump(doc, Dumper=_BlockDumper, sort_keys=False)


def delay_patch(delay: StartupDelay) -> dict[str, Any]:
    """JSON merge patch for kwok's ``pod-ready`` Stage."""
    spec: dict[str, Any] = {"durationMilliseconds": int(round(delay.min_ms))}
    if delay.max_ms > delay.min_ms:
        spec["jitterDurationMilliseconds"] = int(round(delay.max_ms))
    return {"spec": {"delay": spec}}


def write(
    directory: str | Path,
    scale: TimeScale,
    *,
    serialise: bool = False,
    delay: StartupDelay | None = None,
    kind_template: str | Path = "cluster/kind.yaml",
) -> ClusterState:
    """Generate every file for a new cluster and record its state."""
    delay = delay or StartupDelay()
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / SCHEDULER_FILE).write_text(
        scheduler_config_yaml(scale, serialise=serialise), encoding="utf-8"
    )
    (d / KIND_FILE).write_text(
        render_kind_config(Path(kind_template).read_text(encoding="utf-8"), scale),
        encoding="utf-8",
    )
    patch = d / DELAY_PATCH_FILE
    if delay.zero:
        patch.unlink(missing_ok=True)  # never leave a stale delay behind
    else:
        patch.write_text(yaml.safe_dump(delay_patch(delay), sort_keys=False), encoding="utf-8")
    summary = scale.summary()
    summary["scheduler_config_sha256"] = hashlib.sha256(
        (d / SCHEDULER_FILE).read_bytes()
    ).hexdigest()[:12]
    state = ClusterState(
        speedup=scale.speedup,
        serialise=serialise,
        startup_delay_ms=(delay.min_ms, delay.max_ms),
        timescale=summary,
    )
    (d / STATE_FILE).write_text(json.dumps(state.to_json(), indent=2) + "\n", encoding="utf-8")
    return state


def load(path: str | Path) -> ClusterState | None:
    """The recorded state, or ``None`` when no file exists."""
    p = Path(path)
    if not p.exists():
        return None
    return ClusterState.from_json(json.loads(p.read_text(encoding="utf-8")))


def mismatches(
    state: ClusterState, *, speedup: float, delay: StartupDelay, serialise: bool | None = None
) -> list[str]:
    """Every way a benchmark invocation disagrees with the running cluster."""
    problems: list[str] = []
    if state.speedup != speedup:
        problems.append(
            f"SPEEDUP {speedup:g} differs from the {state.speedup:g} the cluster was "
            f"brought up with: kube-scheduler's backoff is scaled for "
            f"{state.speedup:g}. Run `make down && make up SPEEDUP={speedup:g}`, or "
            f"bench with SPEEDUP={state.speedup:g}."
        )
    if state.startup_delay_ms != (delay.min_ms, delay.max_ms):
        problems.append(
            f"STARTUP_DELAY {delay.text()} differs from the "
            f"{state.startup_delay_ms[0]:g}:{state.startup_delay_ms[1]:g} ms kwok's "
            f"pod-ready Stage was configured with at `make up`."
        )
    if serialise is not None and state.serialise != serialise:
        problems.append(
            f"--serialise={serialise} differs from the cluster's scheduler "
            f"(parallelism {'1' if state.serialise else 'default'})."
        )
    return problems
