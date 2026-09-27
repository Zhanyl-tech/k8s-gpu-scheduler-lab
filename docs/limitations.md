# Limitations

What this lab cannot tell you. Stated in full rather than as a footnote,
because these bound every number the repo will ever produce.

> **Phase 2 status, first.** The execution layer described below — repeated
> runs, kube-scheduler queue parity, controller-horizon scaling, startup delay,
> preemption with grace locks, topology scenarios — exists in code and is
> tested against the reference model and a *fake* API server. **No cluster run
> has been made with it.** Every cluster number in this repository is still the
> Phase 1 harness's. `cluster/upstream.lock` records the new scheduler
> configuration as `verified = false`.

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
  fabric. The racks, switches and NVLink flags in `fleets/*.yaml` are a
  **declared scenario**, carried as `topology.k8slab.io/*` node labels; nothing
  probes them. The per-tier penalty factors (1.0 single NVLink node … 2.2
  cross-switch) are **assumed scenario parameters, not measurements**, and the
  lab still sends no NCCL traffic. Placement *tiers* are a measurement of
  decisions; any number that applies the factors inherits the assumption. See
  [metrics.md](metrics.md#topology-and-placement) and
  [the extension scenario](#topology-extension-is-a-scenario-not-a-measurement).
- Kubelet-side reality: image pulls, container start latency, device-plugin
  faults, node pressure eviction. A startup delay can now be *modelled*
  ([below](#startup-latency-is-modelled-not-measured)); it is still not
  measured.

A configuration that wins here has been shown to make better **decisions** on
this trace. It has not been shown to make jobs finish sooner. Those are
different claims and only the first is supportable from this repo.

## Time is compressed

Timestamps and durations are divided by `--speedup` (default 60). Scheduling
decisions are not time-dependent at this resolution, but several control-plane
mechanisms are, and they are specified in real seconds.

**Now controlled (Phase 2), within stated residuals.** `make up` generates a
`KubeSchedulerConfiguration` for the cluster's `SPEEDUP`
(`src/k8slab/timescale.py`, `k8slab scheduler-config --speedup S` prints it),
mounts it into the control-plane node and points kube-scheduler at it
(`cluster/kind.yaml`). Every field and flag was checked against the Kubernetes
v1.32.2 source; the citations are in `timescale.py`. What it scales, and what
is left over, at the default speedup 60:

| mechanism | uncompressed (sim s) | configured | trace experiences (sim s) | residual, worst case |
|---|---|---|---|---|
| initial backoff | 1 (1 to <2 with the 1 s flush) | `podInitialBackoffSeconds: 1` | 60 to <120 | +59 s configured; up to +119 s experienced |
| backoff ceiling | 10 (10 to <11) | `podMaxBackoffSeconds: 1` | 60 to <120 | +50 s configured; up to +110 s experienced |
| unschedulable-pool timeout | 300 (>300 to 330 with the 30 s flush) | `--pod-max-in-unschedulable-pods-duration=5s` | >300 to 2100 | flag value exact (ms rounding); release up to +1800 s late |
| backoffQ flush | 1 | *hard-coded 1 s* | 60 | unscalable; gates both backoff rows |
| unschedulable-pool flush | 30 | *hard-coded 30 s* | 1800 | unscalable; gates the timeout row |
| leader election | 15/10/2 | disabled (`leader-elect: "false"`) | — | removed, not approximated |

Without scaling the ceiling would be 600 simulated seconds. The backoff fields
are whole seconds and **validation rejects 0** (an earlier draft of
`timescale.py` emitted 0 and kube-scheduler would have refused to start), so
1 s real — 60 simulated s — is the floor. The flush timers are fixed
`wait.Until` periods in kube-scheduler's queue and cannot be configured at all
(`PriorityQueue.Run`, pkg/scheduler/backend/queue/scheduling_queue.go lines
358-364 at v1.32.2).

**The flush timers gate the scaled values.** This table used to report the
timeout as experienced at exactly 300 s with no residual. That is the *flag*,
not the behaviour. kube-scheduler checks the timeout only inside the 30 s
flush, and moves a pod when `currentTime.Sub(lastScheduleTime) >
p.podMaxInUnschedulablePodsDuration` (`flushUnschedulablePodsLeftover`, lines
834-848). So a pod no event releases leaves the pool 5 to 35 s real after it
was parked — (300, 2100] simulated s at speedup 60, against (300, 330] on an
uncompressed cluster. Likewise a pod that an event releases while it is still
backing off waits in backoffQ for the next 1 s flush tick at or after its
expiry (`flushBackoffQCompleted`, lines 804-830): a 1 s backoff is [60, 120)
simulated s. The reference model's kube queue (`queueing.py`) models both
ticks the same way. Every results.json records the configured values, these
ranges and the worst-case residuals under `timescale`
(`experienced_range_sim_seconds`, `worst_case_residual_sim_seconds`), and
results.md prints them.
[Source, v1.32.2.](https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/backend/queue/scheduling_queue.go)

Two gotchas the configuration handles, both verified in source: with `--config`
set, kube-scheduler **ignores** `--kubeconfig` and `--kube-api-qps/burst` (so
the file carries `clientConnection.kubeconfig`, `qps` and `burst`), but still
applies `--leader-elect`, which kubeadm sets to `true` (so the flag is
overridden in `cluster/kind.yaml`).

**The plan's `MaxInFlightMovePods` does not exist.** The optimisation plan
names a kube-scheduler field of that name. There is no such field in the v1
`KubeSchedulerConfiguration` (not in `types.go` at v1.32.2, not on the
[reference page](https://kubernetes.io/docs/reference/config-api/kube-scheduler-config.v1/)),
and nothing here sets it.

**The cluster and the benchmark must agree.** The scheduler reads its file
once, at start-up, so the speedup belongs to the cluster. `make up` records it
in `cluster/generated/cluster-state.json`; `make bench` **refuses** (exit 2) if
its `SPEEDUP` or `STARTUP_DELAY` differs, and prints a banner warning if the
record is missing (a cluster not brought up by this Makefile), in which case
results.json records the scheduler configuration as unknown: `timescale`
carries `status: unknown` and keeps the scaled values only as `intended`, and
results.md says the configuration is UNKNOWN instead of stating it. (Until this
revision the scaled values were written as if applied.) With the record, the
configuration is stated only as recorded (`status: recorded`; results.md: "as
`make up` configured kube-scheduler ... the running scheduler's flags were not
read back"), never as applied: `make up` writes `cluster-state.json` before
`kind create cluster` runs, `make bench` compares it only with its own flags,
and nothing reads the running kube-scheduler's command line (`--config`,
`--pod-max-in-unschedulable-pods-duration`, `--leader-elect`) back from the
cluster. The status used to be `applied`. Reading the flags back from the
scheduler's static pod would verify them, and is not built. Reference-model
results say which queue experienced the values (`status: model`) or, with
`--queue-model none`, that none did (`status: not applied`).

**The cluster must carry exactly the fleet.** The runner reads the kwok nodes
back before a replay and refuses a cluster that lacks a fleet node, whose GPU
capacity differs, or — since this revision — that has kwok nodes *outside* the
fleet. `make up` on an existing cluster applies the requested fleet's nodes
and deletes none, so `make up FLEET=a` followed by `make up FLEET=b` leaves a's
extra nodes behind, and K0's pods (`nodeSelector: type=kwok`) could bind there;
the replay would then fail in scoring, after it had run. Switch fleets with
`make down && make up FLEET=...`. `make clean` leaves `cluster/generated/` in
place while the cluster is up (it is what `make bench` checks the speedup
against); `make down` removes both.

**Not verified by a run.** kind's handling of a relative `hostPath` (resolved
with `filepath.Abs` against the directory `kind create cluster` runs in, i.e.
the repo root under `make up`), the kubeadm v1beta3 `extraArgs`/`extraVolumes`
shape kind v0.27.0 generates for 1.32, and kwok's Stage `delay` fields were all
read from the pinned source, not observed working. The mechanism is in place;
whether a cluster comes up with it is unverified.

**Still uncontrolled:**

- **Wait times are quantized** to one runner poll — `speedup × TICK` simulated
  seconds (TICK = 1 s real; 60 s at the default speedup). A difference between
  two configurations smaller than that is not a measurement. The reference
  model passes every 5 simulated seconds: finer than the runner, kept at Phase
  1's value so Phase 1 results reproduce.
- **Capacity returns up to one poll late.** The runner deletes a pod when its
  duration elapses, and the scheduler sees the GPUs return when that delete
  lands. Job runtime accounting corrects for this (see
  [metrics.md](metrics.md)); the scheduler's *view* still lags.
- Client-go rate-limiter windows and any controller resync period of a future
  K1–K4 component, until that component is built and scaled (table below).

### Wall-clock knobs K1 and K2 will need scaled

Kueue (K1) and Volcano (K2) are not built. When they are, these of their
settings are wall-clock-driven and must be scaled like kube-scheduler's. Only
names found in the pinned sources are listed (read 2026-09-26); anything not
found there is marked unverified.

| component | setting | default | source |
|---|---|---|---|
| Kueue v0.19.2 | `waitForPodsReady.timeout` | 30m | `apis/config/v1beta2` types + defaults.go |
| Kueue v0.19.2 | `waitForPodsReady.recoveryTimeout` | = timeout | same |
| Kueue v0.19.2 | `waitForPodsReady.requeuingStrategy.backoffBaseSeconds` | 60 (int32 seconds: quantised like kube-scheduler's) | same |
| Kueue v0.19.2 | `waitForPodsReady.requeuingStrategy.backoffMaxSeconds` | 3600 (int32 seconds) | same |
| Kueue v0.19.2 | `admissionFairSharing.usageHalfLifeTime`, `usageSamplingInterval` | —, 5m | same |
| Kueue v0.19.2 | `multiKueue.gcInterval`, `multiKueue.workerLostTimeout` | 1m, 15m (MultiKueue is out of scope here) | same |
| Kueue v0.19.2 | leader election lease/renew/retry | 15s / 10s / 2s | defaults.go |
| Volcano v1.15.1 | `--schedule-period` ("The period between each scheduling cycle") | 1s | `cmd/scheduler/app/options/options.go` |
| Volcano v1.15.1 | `--resync-period` (informer resync) | 0 | same |
| Volcano v1.15.1 | leader-election lease flags | **unverified** (only `--leader-elect-resource-namespace` was seen) | same |
| Volcano v1.15.1 | job/PodGroup-level timeouts | **unverified** | not checked |

Volcano's `--schedule-period` matters most: it is a batch cycle, so at speedup
60 an unscaled 1 s period is a scheduling pass every simulated minute.

## kwok's default stages are wrong for this

`stage-fast.yaml` installs `pod-complete` and `pod-delete`, which retire a
running pod on kwok's own schedule. Left in place they free GPUs early and
delete pods between polls — silently inflating utilization and losing pods from
the accounting. `make up` deletes both stages. Job lifetime must be the trace's,
not the simulator's.

## Queue mechanics: parity, not a tuned penalty

Phase 1's caveat 2: kube-scheduler (K0) runs its own queue — one pod per
scheduling cycle, unschedulable pods parked, exponential backoff — while the
in-process binder that drives the degenerate policies saw every pending pod on
every pass and never backed off. That flattered the degenerate baselines.

The optimisation plan proposed "an artificial latency penalty matching API
transit" for the in-process binder. **That was not built.** A latency penalty
would have to be tuned until the baselines "look fair", and a tuned constant is
exactly the kind of unfalsifiable adjustment this lab exists to avoid: nothing
measures what API transit costs kube-scheduler per cycle, so any value chosen
would encode the answer. What was built instead is **mechanism parity**:
`src/k8slab/queueing.py` reproduces kube-scheduler's queue — activeQ, backoffQ,
the unschedulable pool, per-pod attempt counts, `initial × 2^(attempts−1)`
backoff capped at the ceiling, requeue on a capacity-freed event, the
unschedulable-pool timeout, the two flush timers, one pod per cycle — each
behaviour checked against the v1.32.2 source (citations in the module). Its
durations come from the same `TimeScale` as the generated scheduler
configuration, so the degenerate binder and the real kube-scheduler experience
the same simulated backoff, including the same quantisation residual. A
per-attempt cycle latency exists as a flag (`--cycle-latency`), **default 0**.

`--queue-model kube` is the default for new results; `--queue-model none`
reproduces Phase 1 exactly (`tests/test_sim_equivalence.py`). What parity does
not cover: kube-scheduler orders activeQ by priority then timestamp, while the
degenerate binder keeps each policy's own order (the order *is* the policy);
scheduling gates, PreEnqueue plugins and events arriving mid-cycle are not
modelled.

The mechanism matters. In the reference model (not a cluster), switching the
same command from `QUEUE_MODEL=none` to `kube` moved D-fifo's makespan from
14.0 h to 15.8 h and D-largest's definition-A fragmentation from 0.0% to 23.1%
([README](../README.md#reference-model--not-a-cluster-run)). How much of
Phase 1's K0-versus-baseline gap it closes on a real cluster is unmeasured.

## Startup latency is modelled, not measured

`--startup-delay MIN:MAX` (real milliseconds; `STARTUP_DELAY` in the Makefile)
adds a bind-to-Running gap: on a cluster, a merge patch adds `spec.delay` to
kwok's `pod-ready` Stage (`durationMilliseconds`, `jitterDurationMilliseconds`;
kwok v0.8.0 draws uniformly in between); in the reference model the same
distribution is drawn from a seeded RNG and multiplied by the speedup. Default
`0:0` keeps Phase 1 comparability. 50:200 ms (3–12 simulated seconds at
speedup 60) is an **illustrative, uncalibrated setting — a judgment call**: no
measurement of kubelet or device-plugin start-up backs it, and earlier text
calling it "the recommended realistic setting" had no source. It is a stand-in
for start-up latency kwok does not have — a modelling choice, not a measurement
of any real node.

Trace runtime now counts from Running, not from bind, and bind-to-Running is
held-but-idle GPU time (`startup_overhead_gpu_hours`; stranded time for a
gang). What the delay is **not**: a fix for double allocation. kube-scheduler's
cache assumes a pod onto its node at bind time, before anything runs, so a
Running delay does not change what the scheduler believes is free. No claim
about races is made.

**Poll quantisation.** The runner sees both transitions only when it polls
(every `speedup` simulated seconds). A 50–200 ms real delay is shorter than a
1 s poll, so on a cluster the gap is observed as 0 or one poll. Worse, it is
observed *asymmetrically*: the in-process binder binds right after a poll, so
bind and Running usually appear together on the next poll (gap 0), while
kube-scheduler binds at arbitrary moments. The model's continuous gap is the
better estimate of the overhead itself; cluster values are coarse.

## Serialisation and the admission gate are diagnostics

Two switches remove concurrency, and both change **what** is measured:

- `--serialise` / `SERIALISE=1` sets kube-scheduler's `parallelism: 1`. That
  parallelism is the number of workers filtering and scoring nodes *inside one
  scheduling cycle*; pods are already popped one per cycle and binding stays
  asynchronous. It changes throughput and does **not** make kube-scheduler
  deterministic — ties between equally scored nodes are broken at random
  (`selectHost`, v1.32.2).
- `--admission-gate` submits each pod only after the previous pod's binding
  was observed. That removes queue pressure, which scheduler throughput is a
  property of.

Rows produced with either are labelled `+gated` in the `src` column, shown, and
excluded from every comparison. Determinism for headline numbers comes from
repetition with reported spreads, not from serialising the scheduler.

## Preemption

The first cluster run lost **66 of 476 pods** to kube-scheduler's priority
preemption. Bare pods have no owning controller, so preempted work is destroyed
rather than requeued — and a real training job would have been resubmitted.

**Unchanged:** the PriorityClasses keep `preemptionPolicy: Never`, so for K0
and the non-preemptive baselines priority affects queue order only, and K0's
numbers are **not** what a default-configured cluster with preemptive priority
classes would produce.

**New (Phase 2):** preemption is its own configuration. `D-preempt` is a
degenerate baseline — strict priority order, and a pod that fits nowhere evicts
the lowest-priority running pods on the node that needs the fewest evictions,
whole gangs at a time. Evicted GPUs stay **locked for the grace period**
(`--grace-seconds`, default 30 simulated seconds, the Kubernetes default
`terminationGracePeriodSeconds`), and the victims are requeued with their work
lost (`--checkpoint-fraction`, default 0, keeps part of it). It is the floor
K1/K2 preemption will be compared against, not a model of kube-scheduler's
DefaultPreemption. That one picks among candidate nodes by fewest
PodDisruptionBudget violations, then the lowest highest-victim priority, then
the smallest sum of victim priorities, then the fewest victims
(`pickOneNodeForPreemption`, v1.32.2), and respects PDBs only on a **best-effort**
basis — "if no such victims are found, preemption will still happen, and lower
priority Pods will be removed despite their PDBs being violated"
([docs](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/));
citations in `src/k8slab/preemption.py`. Known simplifications:
GPUs already under a grace lock are not counted as about to be free, so
D-preempt can over-preempt; on a cluster the same holds for a pod whose work
ended since the last poll (the runner deletes it only after the bind pass, the
lag noted under *Time is compressed*): it is never evicted, but its GPUs are
not yet free either, so D-preempt may evict a running pod elsewhere where the
reference model, which releases finished pods before its pass, would simply
bind; and a nominated preemptor's GPUs are reserved
against every other pod, so a higher-priority pod never takes a lower one's
nomination — where kube-scheduler counts a nominated pod only against pods of
equal or lower priority and may give the node to a higher-priority pod
(`addNominatedPods`; `src/k8slab/binder.py`). The one nomination behaviour
D-preempt shares with upstream is trying the nominated node first.

On a cluster, a pod the control plane deletes is still recorded as lost (a bare
pod is never recreated), and now also keeps its GPUs held in the free-GPU
samples for the grace period instead of freeing them instantly; a Terminating
pod is force-deleted when the grace period ends (kwok has no kubelet to do it).
A lock is released on the first poll at or after its end, before that poll's
sample and bind pass (it used to be one poll later). Such a pod's Running time
is lost work, and none of it counts as topology extension.
If the control plane deleted the pod outright, the real scheduler may reuse
those GPUs before the modelled lock ends; the runner then clamps the sample at
0 and records the conflict in results.json. D-preempt on a cluster performs its
evictions itself through the API and honours its own locks.

## Topology extension is a scenario, not a measurement

`--topology-penalty extend` stretches every job's remaining runtime by its
ASSUMED placement factor when its last pod binds. The factor is computed over
every member's node, including members that already finished before the last
one bound; those have no remaining runtime and are not stretched, the members
still bound are. A gang that never co-ran is therefore penalised for the part
of its work that ran after its placement was complete, and not for the rest —
the scenario prices communication, and its finished members did none. (Such
gangs used to be exempted entirely, which favoured the worst co-schedulers;
[metrics.md](metrics.md#gpu-hours-used-gpu-hours-idle-utilization) has the
numbers.) That makes placement *matter*
inside a replay under a stated premise; it does not measure anything. Such rows
are labelled `+topo`, are never pooled with or ranked against measured rows
(`stats.aggregate` refuses), and change the delivered-GPU-hours invariant to
"demand plus the stretch" ([metrics.md](metrics.md#gpu-hours-used-gpu-hours-idle-utilization)).
`report` (default) applies the factors only to `placement_penalty_mean`; `off`
applies them nowhere.

## The workload is synthetic

Phase 1 traces are generated, not replayed from a real cluster. The generator's
shape — log-normal runtimes, power-of-two GPU counts, 18% gangs — is a
plausible ML research fleet, not a measured one. Translated `sacct` traces are
**planned, and not part of the Phase 2 work so far**; the record schema is
already fixed for them, so nothing downstream would change.

The default profile is deliberately contended. At the first defaults tried
(300 jobs, 45 s arrivals) every degenerate policy completed every job and their
utilizations sat within 0.7 percentage points of each other — the fleet was
never pressured, so the trace could not distinguish a scheduler from a coin
flip. The `light` profile preserves those settings for CI smoke tests and must
never be used to compare schedulers.

## Run-to-run variance is large, and it is not yet controlled on a cluster

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
- **Only large, repeated differences should be read as signal.**
- **Repeated runs must be reported as a spread**, not a point estimate. Both
  halves of that now exist. Reporting: every configuration is aggregated into
  `mean±sd` with `n` and a stability verdict, n=1 is never stable, results.md
  opens with an UNSTABLE banner when a gated metric varies beyond tolerance,
  and differences are called resolvable only when a seeded bootstrap interval
  excludes zero ([metrics.md](metrics.md#repeated-runs-mean--sd-and-the-stability-verdict)).
  Execution: `make bench` runs `REPEAT` (default 5) repeats of every
  configuration, **interleaved** (every config's repeat 1, then repeat 2, …) so
  machine drift is not mistaken for a difference, with the trace fixed and a
  per-repeat harness seed; results are rewritten after every run.
  **No repeated cluster run has been made yet**, so every cluster table here is
  still a smoke test, and whether five repeats are enough to tame the spread
  above is unknown.
- **"Resolvable" is a screen, not a 5% test.** With the five repeats the plan
  calls for, the percentile bootstrap declared a difference between two
  *identical* distributions resolvable in 13% of simulated trials (20% at three
  repeats). The numbers and the command are in
  [metrics.md](metrics.md#pairwise-differences-the-bootstrap).
- **`deterministic` on the reference model says nothing about a cluster.**
  At the default **zero** startup delay, repeats of D-fifo, D-largest and
  D-preempt in the model are bit-identical (sd = 0); only D-random's own
  randomness varies. With a non-zero `--startup-delay` the delay draws are
  seeded per repeat, so every policy's repeats vary (all three were `unstable`
  at 50:200 in [metrics.md](metrics.md#repeated-runs-mean--sd-and-the-stability-verdict)'s
  check). Either way that is a property of the model.

## Single versions, one fleet, one trace seed

Everything is pinned in [../cluster/upstream.lock](../cluster/upstream.lock).
Scheduler behaviour moves across minor versions. Results are valid for the
pinned combination and nothing else.

Results are also from a single fleet shape and, by default, a single trace
seed. `--trace-seeds a,b,c` (`TRACE_SEEDS`) now runs a separate across-seed
study, reported in its own section and never pooled with the repeats. With one
run per configuration per trace its spread mixes workload variance, harness
variance and (D-random) policy randomness, and cannot separate them — this page
used to call it workload variance alone — so it carries no stability verdict;
configurations are compared paired by trace
([metrics.md](metrics.md#repeated-runs-mean--sd-and-the-stability-verdict)).
No such study has been published yet.

## The comparison is not yet a comparison

Only K0 and the degenerate baselines are built. Kueue, Volcano, NVIDIA's
bin-packing and KAI/DRA are unimplemented, and the cross-substrate Slurm control
(S0) is unimplemented. **The central claim this repo was built to test —
NVIDIA's enhanced Volcano bin-packing — has not been tested.** Its row does not
exist in any results table, and will not until K3 has been built and run.

What there is to test is narrower than the figure the repo started with. The
primary source, NVIDIA's [*Practical Tips for Preventing GPU Fragmentation for
Volcano Scheduler*][nv-volcano] (2025-03-31, read 2026-09-26), reports
before-and-after snapshots on one DGX Cloud cluster of four-GPU L40S nodes,
against Volcano's default placement rather than kube-scheduler: nodes with all
four GPUs free went from 18 to 214, and average GPU utilization reached roughly
90%. It describes the cluster only as "thousands of GPUs" and the workload only
as four classes (distributed training, batch inference, data processing,
notebooks); it gives no node count, job mix or trace, time window or
fragmentation formula, and no percentage reduction. The README used to state as fact that this
bin-packing "was reported at KubeCon EU 2026 to cut GPU node fragmentation by
34% versus the default scheduler". That is **unverified**: I found it only in
a third-party blog post that names no talk or speaker, and in neither NVIDIA
source I checked (the post above, and NVIDIA's KubeCon 2026 post); details in
[the README](../README.md#the-thesis). A K3 row can at most test NVIDIA's
operational quantity — nodes by free-GPU count — on this trace and fleet. It
cannot confirm or refute a percentage whose definition, baseline and workload
are unknown.

[nv-volcano]: https://developer.nvidia.com/blog/practical-tips-for-preventing-gpu-fragmentation-for-volcano-scheduler/
