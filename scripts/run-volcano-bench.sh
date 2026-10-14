#!/usr/bin/env bash
# Bring up this lab's kind+kwok cluster, install Volcano on it, and run K2.
# Run `make down` first if Kueue is installed: its pod webhook gates every pod
# in the lab namespace, so the two must not share a cluster.
#
# Usage: ./run-volcano-bench.sh [fleet] [jobs] [speedup]
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"

FLEET=${1:-fleets/default.yaml}
JOBS=${2:-800}
SPEEDUP=${3:-60}

KCTX=kind-k8slab
VOLCANO_VERSION=v1.15.2

make up FLEET="$FLEET" SPEEDUP="$SPEEDUP"

kubectl --context "$KCTX" apply --server-side \
	-f "https://raw.githubusercontent.com/volcano-sh/volcano/$VOLCANO_VERSION/installer/volcano-development.yaml"
for d in volcano-admission volcano-controllers volcano-scheduler; do
	kubectl --context "$KCTX" -n volcano-system rollout status "deploy/$d" --timeout=300s
done

# Every PodGroup points at the installer's `default` Queue, which a job creates
# and can race the first pods.
kubectl --context "$KCTX" wait --for=create queue.scheduling.volcano.sh/default --timeout=180s

# Set explicitly so the run cannot follow kubectl's ambient current-context.
KUBE_CONTEXT="$KCTX" .venv/bin/python3 integrations/bench_extended.py K2 "$FLEET" "$SPEEDUP" "$JOBS"
