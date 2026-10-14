#!/usr/bin/env bash
# Bring up this lab's kind+kwok cluster, install Kueue on it, and run K1.
# Kueue's pod webhook gates every pod in the lab namespace, so run `make down`
# before run-volcano-bench.sh; the two must not share a cluster.
#
# Usage: ./run-kueue-bench.sh [fleet] [jobs] [speedup]
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"

FLEET=${1:-fleets/default.yaml}
JOBS=${2:-800}
SPEEDUP=${3:-60}

KCTX=kind-k8slab
KUEUE_VERSION=v0.19.5

make up FLEET="$FLEET" SPEEDUP="$SPEEDUP"

kubectl --context "$KCTX" apply --server-side \
	-f "https://github.com/kubernetes-sigs/kueue/releases/download/$KUEUE_VERSION/manifests.yaml"
kubectl --context "$KCTX" -n kueue-system rollout status deploy/kueue-controller-manager --timeout=180s

# The runner creates the namespace lazily; the LocalQueue needs it first.
kubectl --context "$KCTX" create namespace k8slab --dry-run=client -o yaml | kubectl --context "$KCTX" apply -f -
.venv/bin/k8slab kueue-queue --fleet "$FLEET" | kubectl --context "$KCTX" apply -f -

# Set explicitly so the run cannot follow kubectl's ambient current-context.
KUBE_CONTEXT="$KCTX" .venv/bin/python3 integrations/bench_extended.py K1 "$FLEET" "$SPEEDUP" "$JOBS"
