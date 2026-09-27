PY      ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
CLUSTER := k8slab
FLEET   ?= fleets/default.yaml
PROFILE ?= default
SPEEDUP ?= 60
# Cluster results go to a subdirectory: results/results.{md,json} is the
# committed Phase 1 cluster run the README cites, and must not be overwritten.
# results/phase2 is where Phase 2 cluster runs are meant to be committed, so
# `make clean` never deletes it (the CLI uses the same defaults and refuses to
# overwrite results of another schema or source).
RESULTS ?= results/phase2
MODEL_RESULTS ?= results-model
KWOK    := v0.8.0
# Generated per cluster by `make up`: the speedup-scaled scheduler config, the
# kind config that mounts it, the kwok startup-delay patch and their record.
GENERATED := cluster/generated

# ---- Phase 2 harness (docs/limitations.md, README "Phase 2 methodology") ----
# Runs per configuration, interleaved across configurations.
REPEAT        ?= 5
# The in-process binder's queue: kube (kube-scheduler mechanics) or none (Phase 1).
QUEUE_MODEL   ?= kube
# off | report | extend (extend is a SCENARIO: rows are labelled +topo).
TOPOLOGY      ?= report
# Bind-to-Running delay, MIN:MAX real milliseconds. 0:0 keeps Phase 1
# comparability; 50:200 is an illustrative, uncalibrated setting (a judgment
# call, not a measured start-up latency). A cluster setting: it is fixed at
# `make up` (kwok Stage) and `make bench` must use the same value.
STARTUP_DELAY ?= 0:0
# Optional comma-separated trace seeds for the separate across-seed study.
TRACE_SEEDS   ?=
# Set to 1 for the kube-scheduler parallelism-1 DIAGNOSTIC (gated rows).
SERIALISE     ?=
# Set to 1 to exit non-zero when any configuration is unstable.
FAIL_ON_UNSTABLE ?=

BENCH_FLAGS = --fleet $(FLEET) --profile $(PROFILE) --speedup $(SPEEDUP) \
	--repeat $(REPEAT) --queue-model $(QUEUE_MODEL) --topology-penalty $(TOPOLOGY) \
	--startup-delay $(STARTUP_DELAY) \
	$(if $(TRACE_SEEDS),--trace-seeds $(TRACE_SEEDS)) \
	$(if $(FAIL_ON_UNSTABLE),--fail-on-unstable)
CLUSTER_FLAGS = --speedup $(SPEEDUP) --startup-delay $(STARTUP_DELAY) \
	$(if $(SERIALISE),--serialise) --dir $(GENERATED)

.PHONY: help install up down bench bench-model scheduler-config nodes trace test lint typecheck check clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/k8slab: pyproject.toml
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(BIN)/python -m pip install -q --upgrade pip
	@$(BIN)/python -m pip install -q -e ".[dev,cluster]"
	@touch $(BIN)/k8slab

install: $(BIN)/k8slab ## Create the venv and install

up: install ## Bring up the simulated fleet (kind + kwok + nodes) for SPEEDUP/STARTUP_DELAY
	@# The scheduler config, the kind config that mounts it and the kwok delay
	@# are fixed when the cluster is created. An existing cluster is checked
	@# against the requested settings instead of being silently reused.
	@if kind get clusters 2>/dev/null | grep -qx $(CLUSTER); then \
		$(BIN)/k8slab cluster-config check $(CLUSTER_FLAGS); \
	else \
		$(BIN)/k8slab cluster-config write $(CLUSTER_FLAGS) \
			&& kind create cluster --config $(GENERATED)/kind.yaml; \
	fi
	@kubectl --context kind-$(CLUSTER) apply -f \
		https://github.com/kubernetes-sigs/kwok/releases/download/$(KWOK)/kwok.yaml
	@kubectl --context kind-$(CLUSTER) apply -f \
		https://github.com/kubernetes-sigs/kwok/releases/download/$(KWOK)/stage-fast.yaml
	@# stage-fast retires pods on kwok's own schedule: `pod-complete` flips a
	@# running pod to Succeeded and `pod-delete` removes it. Both are wrong
	@# here. A pod must hold its GPUs for exactly the duration the trace says,
	@# and job lifetime is the runner's to control -- otherwise the scheduler
	@# sees capacity free early and every utilization number is inflated.
	@kubectl --context kind-$(CLUSTER) delete stages.kwok.x-k8s.io pod-complete pod-delete --ignore-not-found
	@# Startup delay (STARTUP_DELAY): a merge patch adding spec.delay to kwok's
	@# pod-ready Stage. Only written when the delay is non-zero.
	@if [ -f $(GENERATED)/pod-ready-delay.patch.yaml ]; then \
		kubectl --context kind-$(CLUSTER) patch stages.kwok.x-k8s.io pod-ready \
			--type merge --patch-file $(GENERATED)/pod-ready-delay.patch.yaml; \
	fi
	@kubectl --context kind-$(CLUSTER) -n kube-system rollout status deploy/kwok-controller --timeout=120s
	@$(BIN)/k8slab nodes --fleet $(FLEET) | kubectl --context kind-$(CLUSTER) apply -f -
	@kubectl --context kind-$(CLUSTER) get nodes -l type=kwok --no-headers | wc -l | xargs echo "kwok nodes ready:"

bench: install ## Replay the trace against every configuration, REPEAT times each
	@$(BIN)/k8slab bench $(BENCH_FLAGS) $(if $(SERIALISE),--serialise) \
		--cluster-state $(GENERATED)/cluster-state.json --results $(RESULTS)

bench-model: install ## Score the degenerate policies with the reference model (no cluster)
	@$(BIN)/k8slab bench --reference-model $(BENCH_FLAGS) --results $(MODEL_RESULTS)

scheduler-config: install ## Print the KubeSchedulerConfiguration for SPEEDUP
	@$(BIN)/k8slab scheduler-config --speedup $(SPEEDUP) $(if $(SERIALISE),--serialise)

nodes: install ## Print the kwok Node manifests for the fleet
	@$(BIN)/k8slab nodes --fleet $(FLEET)

trace: install ## Print a summary of the generated trace
	@$(BIN)/k8slab trace --profile $(PROFILE)

down: ## Tear the cluster down
	@kind delete cluster --name $(CLUSTER) 2>/dev/null || true
	@rm -rf $(GENERATED)
	@echo "cluster $(CLUSTER) deleted"

test: install ## Run the test suite (no cluster required)
	@$(BIN)/python -m pytest -q

lint: install ## ruff
	@$(BIN)/ruff check .

typecheck: install ## mypy --strict
	@$(BIN)/mypy

check: lint typecheck test ## lint, typecheck, test (CI also runs bench-model)

clean: ## Remove venv, caches and reference-model results; cluster/generated only with no cluster up
	@rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache $(MODEL_RESULTS)
	@# cluster/generated is bind-mounted into a live cluster's kube-scheduler and
	@# is the record `make bench` checks SPEEDUP and STARTUP_DELAY against.
	@# Deleting it under a running cluster turned that refusal into a warning
	@# and an UNKNOWN configuration. `make down` removes it with the cluster.
	@if kind get clusters 2>/dev/null | grep -qx $(CLUSTER); then \
		echo "cluster $(CLUSTER) is up: keeping $(GENERATED) (make down removes both)"; \
	else \
		rm -rf $(GENERATED); \
	fi
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
