"""Render results.

Two rules, both inherited from the sibling repos:

* Degenerate baselines appear in every table. A configuration that does not
  clearly beat ``D-random`` has not been shown to schedule.
* Every row states whether it was measured on a cluster or produced by the
  in-process reference model. A table that mixes the two without saying so is
  the specific dishonesty this module exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .metrics import Metrics

COLUMNS: tuple[tuple[str, str], ...] = (
    ("config", "{m.config}"),
    ("src", "{src}"),
    ("makespan h", "{m.makespan_hours:.1f}"),
    ("util %", "{m.utilization:.1%}"),
    ("GPU-h used", "{m.gpu_hours_used:.1f}"),
    ("mean wait", "{mean_wait_min:.0f}m"),
    ("p95 wait", "{p95_wait_min:.0f}m"),
    ("frag A", "{m.fragmentation_rate:.1%}"),
    ("frag B", "{m.fragmentation_ref:.1%}"),
    ("gang DL", "{m.gang_deadlock_rate:.1%}"),
    ("fair", "{m.fairness_ratio:.2f}"),
)


def _row(m: Metrics) -> list[str]:
    ctx = {
        "m": m,
        "src": "cluster" if m.measured_on_cluster else "model",
        "mean_wait_min": m.mean_wait / 60,
        "p95_wait_min": m.p95_wait / 60,
    }
    return [fmt.format(**ctx) for _, fmt in COLUMNS]


def markdown_table(results: list[Metrics]) -> str:
    headers = [h for h, _ in COLUMNS]
    rows = [_row(m) for m in results]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |")
    return "\n".join(out)


def text_table(results: list[Metrics]) -> str:
    return markdown_table(results)


def write_results(results: list[Metrics], directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.md").write_text(markdown_table(results) + "\n", encoding="utf-8")
    (d / "results.json").write_text(
        json.dumps([asdict(m) for m in results], indent=2) + "\n", encoding="utf-8"
    )
    return d / "results.md"
