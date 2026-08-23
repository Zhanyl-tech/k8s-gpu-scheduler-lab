PY      ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
CLUSTER := k8slab
FLEET   ?= fleets/default.yaml
PROFILE ?= default
SPEEDUP ?= 60
RESULTS ?= results
KWOK    := v0.8.0

.PHONY: help install up down bench bench-model nodes trace test lint typecheck check clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/k8slab: pyproject.toml
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(BIN)/python -m pip install -q --upgrade pip
	@$(BIN)/python -m pip install -q -e ".[dev,cluster]"
	@touch $(BIN)/k8slab

install: $(BIN)/k8slab ## Create the venv and install

up: install ## Bring up the simulated fleet (kind + kwok + nodes)
	@kind get clusters 2>/dev/null | grep -qx $(CLUSTER) \
		|| kind create cluster --config cluster/kind.yaml
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
	@kubectl --context kind-$(CLUSTER) -n kube-system rollout status deploy/kwok-controller --timeout=120s
	@$(BIN)/k8slab nodes --fleet $(FLEET) | kubectl --context kind-$(CLUSTER) apply -f -
	@kubectl --context kind-$(CLUSTER) get nodes -l type=kwok --no-headers | wc -l | xargs echo "kwok nodes ready:"

bench: install ## Replay the trace against every configuration and write results
	@$(BIN)/k8slab bench --fleet $(FLEET) --profile $(PROFILE) \
		--speedup $(SPEEDUP) --results $(RESULTS)

bench-model: install ## Score the degenerate policies with the reference model (no cluster)
	@$(BIN)/k8slab bench --reference-model --fleet $(FLEET) --profile $(PROFILE) \
		--results $(RESULTS)-model

nodes: install ## Print the kwok Node manifests for the fleet
	@$(BIN)/k8slab nodes --fleet $(FLEET)

trace: install ## Print a summary of the generated trace
	@$(BIN)/k8slab trace --profile $(PROFILE)

down: ## Tear the cluster down
	@kind delete cluster --name $(CLUSTER) 2>/dev/null || true
	@echo "cluster $(CLUSTER) deleted"

test: install ## Run the test suite (no cluster required)
	@$(BIN)/python -m pytest -q

lint: install ## ruff
	@$(BIN)/ruff check .

typecheck: install ## mypy --strict
	@$(BIN)/mypy

check: lint typecheck test ## Everything CI runs

clean: ## Remove venv, caches and results
	@rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache results results-model
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
