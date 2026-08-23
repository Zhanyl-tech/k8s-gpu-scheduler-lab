# Limitations

What this lab cannot tell you. Stated in full rather than as a footnote,
because these bound every number the repo will ever produce.

## The hardware is not real

Every GPU node is a [kwok](https://kwok.sigs.k8s.io/) node: an API object with
no kubelet behind it, advertising `nvidia.com/gpu` resources that do not exist.
Pods are `registry.k8s.io/pause` containers that never run on the node they are
bound to.

**Real:** the API server, the scheduler under test, Pod and Node objects, the
resource model, quota and priority admission, and every binding decision.

**Not real, and unmeasurable here:**

- NCCL and collective-communication performance. No traffic is sent.
- GPU contention, MPS and MIG sharing, memory pressure, ECC behaviour.
- Thermal throttling and power.
- Network topology. A topology-aware scheduler's placement can be *observed*,
  but whether that placement was actually faster cannot be — the lab has no
  fabric.
- Kubelet-side reality: image pulls, container start latency, device-plugin
  faults, node pressure eviction.

A configuration that wins here has been shown to make better **decisions** on
this trace. It has not been shown to make jobs finish sooner. Those are
different claims and only the first is supportable from this repo.

## Time is compressed

Timestamps and durations are divided by `--speedup` (default 60). Scheduling
decisions are not time-dependent at this resolution, but several things are, and
the lab is blind to all of them: leases and leader election, scheduler backoff
ceilings, controller resync periods, and any rate limiting with a fixed window.

Two consequences that affect the numbers directly:

- **Wait times are quantized** to one poll interval — `speedup × TICK`
  simulated seconds, 200 s at `--speedup 200`. A difference between two
  configurations smaller than that is not a measurement.
- **Capacity returns up to one poll late.** The runner deletes a pod when its
  duration elapses, and the scheduler sees the GPUs return when that delete
  lands. Job runtime accounting corrects for this (see
  [metrics.md](metrics.md)); the scheduler's *view* still lags.

## kwok's default stages are wrong for this

`stage-fast.yaml` installs `pod-complete` and `pod-delete`, which retire a
running pod on kwok's own schedule. Left in place they free GPUs early and
delete pods between polls — silently inflating utilization and losing pods from
the accounting. `make up` deletes both stages. Job lifetime must be the trace's,
not the simulator's.

## Preemption is disabled in Phase 1

The first cluster run lost **66 of 476 pods** to kube-scheduler's priority
preemption. Bare pods have no owning controller, so preempted work is destroyed
rather than requeued — and a real training job would have been resubmitted.

Phase 1 creates PriorityClasses with `preemptionPolicy: Never`. Priority
therefore affects queue order only. This means:

- the lab currently says nothing about preemption behaviour, and
- K0's numbers are **not** what a default-configured cluster with preemptive
  priority classes would produce.

Preemption deserves to be its own configuration, measured deliberately. It
should not be an uncontrolled variable inside every other one.

## The workload is synthetic

Phase 1 traces are generated, not replayed from a real cluster. The generator's
shape — log-normal runtimes, power-of-two GPU counts, 18% gangs — is a
plausible ML research fleet, not a measured one. Phase 2 replaces it with
translated `sacct` traces; the record schema is already fixed for that, so
nothing downstream changes.

The default profile is deliberately contended. At the first defaults tried
(300 jobs, 45 s arrivals) every degenerate policy completed every job and their
utilizations sat within 0.7 percentage points of each other — the fleet was
never pressured, so the trace could not distinguish a scheduler from a coin
flip. The `light` profile preserves those settings for CI smoke tests and must
never be used to compare schedulers.

## Run-to-run variance is large, and it is not yet controlled

**This is the most important limitation on the page.** Two runs of the *same*
configuration, on the *same* trace and seed, on the same machine, produced:

| run | K0 utilization | K0 makespan | D-fifo utilization |
|---|---|---|---|
| first  | 33.7% | 29.8 h | 37.9% |
| second | 54.2% | 18.5 h | 65.6% |

The trace is deterministic and seeded. The variance comes from the harness, not
the workload: the replay is driven by real wall-clock, so API-server latency,
scheduler responsiveness under queue pressure, and the machine's load all feed
back into the simulated clock. A slow poll means simulated time advances
further between scheduling passes.

Consequences, stated plainly:

- **A single run's absolute numbers mean very little.** Do not quote a
  utilization or makespan figure from this repo as though it were a property of
  the scheduler.
- **Only large, repeated differences should be read as signal.** The one
  direction that held across both runs is that K0 finished the trace *later*
  than every degenerate policy — see the README's reading note for why part of
  that is a harness artefact rather than a scheduler property.
- **Phase 2 must add repeated runs and report a spread**, not a point estimate.
  Until it does, every table here is a smoke test.

## Single versions, one fleet, one seed

Everything is pinned in [../cluster/upstream.lock](../cluster/upstream.lock).
Scheduler behaviour moves across minor versions. Results are valid for the
pinned combination and nothing else.

Results are also from a single fleet shape and a single trace seed. Nothing here
reports variance across seeds yet, so small differences between configurations
should be treated as noise until a multi-seed run exists.

## The comparison is not yet a comparison

Only K0 and the degenerate baselines are built. Kueue, Volcano, NVIDIA's
bin-packing and KAI/DRA are unimplemented, and the cross-substrate Slurm control
(S0) is unimplemented. **The central claim this repo was built to test — the 34%
fragmentation reduction — has not been tested.** Its row in the results table is
empty and stays empty until it has been run.
