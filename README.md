# k8s-gpu-scheduler-lab

**A controlled comparison of Kubernetes GPU schedulers on the same workload
traces.** Slurm and Kubernetes solve the same problem — allocating scarce
accelerators across competing jobs — with different mechanisms. I could not
find a published controlled comparison of the Kubernetes options on an
identical trace. That comparison is the product.

> **Phase 2 of 3, in progress.** Phase 1 built the substrate, the workload
> generator, the metrics, the degenerate baselines and K0. Phase 2 so far adds
> the measurement and execution layers the Phase 1 results showed were missing:
> repeated runs with spreads and a bootstrap screen for differences, kube-scheduler queue parity for
> the baselines, time-scaled controller horizons, a startup delay, preemption
> with grace locks (D-preempt) and topology scenarios — see
> [Phase 2 methodology](#phase-2-methodology). **No cluster run has been made
> with the Phase 2 harness yet**: every cluster number below is Phase 1's.
> Kueue, Volcano, NVIDIA's bin-packing and KAI/DRA are not built, and their rows
> do not exist. Published as it is built. Changes: [CHANGELOG.md](CHANGELOG.md).

<p align="center">
  <img src="./docs/architecture.svg" width="100%"
       alt="Pipeline: a seeded workload trace and a heterogeneous fleet definition feed a kind cluster. Inside it, the control plane, the scheduler under test and every binding decision are real; the GPU nodes are kwok objects with no kubelet and no hardware. Pod bindings and sampled free-GPU counts become an observation, scored into makespan, waits, three fragmentation definitions (definition C at the node level is the headline), stranded gang GPU-hours and fairness. The degenerate baselines are D-fifo, D-random, D-largest and D-preempt. The runner, not kwok, deletes each pod when its trace duration elapses.">
</p>

---

## The thesis

Slurm splits nothing: multifactor priority plus EASY backfill, one controller,
one policy. Kubernetes splits the same job across projects — [Kueue][kueue] does
quota and admission, [Volcano][volcano] does gang scheduling and DRF fairness,
NVIDIA's [KAI][kai] does topology-aware gang scheduling with DRA. Each is
benchmarked, when at all, against the default scheduler on its own chosen
workload.

Two claims motivated this repo, and neither is currently checkable:

- **NVIDIA reported that adding bin-packing to the Volcano scheduler cut GPU
  fragmentation on one production cluster.** The primary source is NVIDIA's
  technical blog post [*Practical Tips for Preventing GPU Fragmentation for
  Volcano Scheduler*][nv-volcano] (Ameya Parab, 2025-03-31). What it reports,
  as before-and-after snapshots on a DGX Cloud Kubernetes cluster of NVIDIA
  L40S nodes with four GPUs each, against **Volcano's own default placement**
  (gang scheduling with the workload "placed randomly"), not against
  kube-scheduler:
  - nodes with all four GPUs free: **18 → 214** (nodes with three free GPUs:
    about 115 → about 9);
  - average GPU utilization: **roughly 90%**, against an 80% contractual
    target.

  It describes the cluster's scale only as "thousands of GPUs", and its
  workload only as four classes: multi-node, multi-GPU distributed training,
  batch inference, GPU-backed data processing, and interactive notebooks. It
  gives no node count, no job mix or trace, no time window for the node counts
  and no formula for fragmentation, and it contains no percentage reduction in
  fragmentation. The number this repo was
  started with — "cut GPU node fragmentation by **34%** versus the default
  scheduler", attributed to benchmarks shared at **KubeCon EU 2026** — is
  **unverified**, in both its value and its attribution. The only place I found
  it is a third-party blog post ([Gheware, 2026-05-13][gheware]) that names no
  talk, speaker, slides or recording. NVIDIA's post above does not mention
  KubeCon; NVIDIA's own KubeCon 2026 post ([2026-03-24][nv-kubecon]) mentions
  neither Volcano nor fragmentation; and a search of the
  [KubeCon EU 2026 schedule][kccnc] for "fragmentation" lists no NVIDIA
  session. (All four pages read 2026-09-26.) I found no independent
  reproduction of NVIDIA's result either. The quantity NVIDIA did report — how many
  nodes have every GPU free, and more generally how nodes are distributed by
  free-GPU count — is computable from the per-node free-GPU samples this lab
  records, but no metric for it is implemented yet. As
  [docs/metrics.md](docs/metrics.md) shows, the choice of fragmentation
  definition moves the number by tens of points, so a percentage without one
  cannot be reproduced.
- **Gang scheduling is assumed to be strictly better for multi-node training.**
  It has a cost — held-but-idle GPUs while a gang assembles — that is rarely
  measured alongside the benefit.

This lab is the apparatus for checking both. **If NVIDIA's result does not
reproduce on this trace, that is the finding and it gets published as the
finding.**

## What kwok can and cannot tell you

Read this before any number below.

Every "GPU node" here is a [kwok][kwok] node: an API object with no kubelet
behind it, advertising `nvidia.com/gpu` it does not have. **The control plane is
real, the scheduler is real, and every binding decision is real. Nothing else
is.**

**This lab can measure:** placement decisions, queue wait, admission order,
packing and fragmentation, gang-admission behaviour, quota and preemption
effects, and scheduler throughput under queue pressure.

**This lab cannot measure — at all:**

- NCCL or collective-communication performance. No traffic is sent.
- Real GPU contention, MPS/MIG sharing behaviour, or memory pressure.
- Thermal behaviour, throttling, or power.
- Network topology effects, including whether a topology-aware placement was
  actually *better* — only whether the scheduler produced it.
- Anything depending on wall-clock, because time is compressed (see below).
- Kubelet-side failures: image pulls, admission webhooks, device-plugin faults.

A scheduler that wins here has been shown to make better *decisions* on this
trace. It has not been shown to make jobs faster. Those are different claims and
this repo can only support the first.

**Time is compressed.** An eleven-hour trace is replayed in minutes by dividing
every timestamp by `--speedup` (default 60). Scheduling decisions are not
time-dependent at this resolution, but backoff ceilings, leases and controller
resync periods are. Phase 2 scales kube-scheduler's to match — within a stated
residual, because its backoff fields are whole seconds — and records what could
not be scaled ([docs/limitations.md](docs/limitations.md#time-is-compressed)).

## Configurations

| id | scheduler | status |
|---|---|---|
| **K0** | default `kube-scheduler` + device plugin | **built** |
| K1 | + Kueue (quota, admission) | planned |
| K2 | + Volcano (gang, DRF) | planned |
| K3 | + NVIDIA enhanced Volcano bin-packing | planned — tests NVIDIA's reported result ([the thesis](#the-thesis)) |
| K4 | NVIDIA KAI + DRA | planned |
| S0 | Slurm, via [slurm-scheduler-lab][ssl] on the same trace | planned — cross-substrate control |
| D-fifo | oldest-first, first-fit | **built** (degenerate) |
| D-random | random pod order, random node | **built** (degenerate) |
| D-largest | largest request first | **built** (degenerate) |
| D-preempt | strict priority; a pod that fits nowhere evicts lower-priority pods (fewest evictions, whole gangs) | **built** (degenerate) — the floor for K1/K2 preemption |

**The degenerate baselines are not filler.** A configuration that does not
clearly beat `D-random` has not been shown to schedule. The convention comes
from [slurm-rca-bench][rca], where a degenerate agent that reads no telemetry
sets the floor every real agent has to clear. If a real scheduler barely beats
FIFO here, the correct response is a harder trace — never a softer metric.

## Results

### Run 2 — 800 jobs / 1201 pods, 140-GPU heterogeneous fleet, seed 0

| config    | src     | makespan h | util % | GPU-h used | mean wait | p95 wait | frag A | frag B | gang DL | fair |
|-----------|---------|------------|--------|------------|-----------|----------|--------|--------|---------|------|
| K0        | cluster | 18.5       | 54.2%  | 1404.7     | 173m      | 559m     | 17.4%  | 51.9%  | 38.1%   | 1.00 |
| D-fifo    | cluster | 15.3       | 65.6%  | 1404.7     | 283m      | 544m     | 15.9%  | 63.2%  | 58.3%   | 1.00 |
| D-random  | cluster | 20.8       | 48.3%  | 1404.7     | 256m      | 678m     | 22.4%  | 57.9%  | 92.8%   | 1.00 |
| D-largest | cluster | 18.5       | 54.4%  | 1404.7     | 396m      | 618m     | 0.0%   | 55.6%  | 46.0%   | 1.00 |

Measured on a real control plane (kind + kube-scheduler); `src=cluster` on every
row. Pinned versions in [cluster/upstream.lock](cluster/upstream.lock). Produced
by the **Phase 1 harness**: n=1, the degenerate policies bound by an in-process
loop with no queue model (today's `--queue-model none`), kube-scheduler's
backoff unscaled (a 10 s real ceiling was 10 simulated minutes), runtime
counted from the bind poll, no D-preempt.
This table predates the current metric set and is reproduced as measured: it is
a single run per configuration (n=1), `gang DL` is now deprecated in favour of
stranded gang GPU-hours, and fragmentation definition C did not exist yet. A
results file produced today reports each configuration as `mean±sd` with `n`
and opens with an UNSTABLE banner for n=1 — see
[docs/metrics.md](docs/metrics.md#repeated-runs-mean--sd-and-the-stability-verdict).
`GPU-h used` is identical across configurations by construction — every job
completes, so delivered GPU-hours are fixed by the trace and **makespan is the
quantity that actually varies**. Utilization is a restatement of it.

**Read these numbers with three caveats, in this order.**

**1. Run-to-run variance is larger than most of the differences in the table.**
The same K0 configuration, same trace, same seed, run twice, gave 33.7% and
54.2% utilization (29.8 h and 18.5 h makespan). The trace is deterministic; the
harness is not, because the replay is driven by real wall-clock and API latency
feeds back into the simulated clock. **Do not quote an absolute number from this
table.** Only differences that survive repetition mean anything. *Phase 2:* the
mechanism now exists — `make bench` runs every configuration `REPEAT` times
(default 5), interleaved, and reports `mean±sd`, a stability verdict and
bootstrap intervals. **It has not been run on a cluster yet**, so this table is
still n=1 and whether five repeats tame the spread is unknown.

**2. K0 and the degenerate baselines are not measured through equal machinery.**
`kube-scheduler` runs its own queue with exponential backoff on unschedulable
pods and schedules one pod per scheduling cycle (binding then proceeds
asynchronously, several bindings in flight at once — `ScheduleOne` in
[schedule_one.go, v1.32.2](https://github.com/kubernetes/kubernetes/blob/v1.32.2/pkg/scheduler/schedule_one.go)).
In this table the degenerate policies were
bound by an in-process loop with global visibility of every pending pod on
every pass and no backoff at all. That asymmetry flatters the degenerate
baselines on makespan, and it is a property of this harness rather than of the
schedulers. *Phase 2:* the in-process binder now goes through a reproduction of
kube-scheduler's queue ([`queueing.py`](src/k8slab/queueing.py), checked
against the v1.32.2 source) with the same simulated backoff the time-scaled
kube-scheduler experiences — mechanism parity, not a tuned latency penalty
([why](docs/limitations.md#queue-mechanics-parity-not-a-tuned-penalty)). In the
reference model that mechanism alone moves D-fifo's makespan from 14.0 h to
15.8 h (table below). **No cluster run has been made with it**, so how much of
this table's K0-versus-baseline gap it closes is unmeasured.

**3. Nothing here has tested the claim the repo was built for.** K3 is
unimplemented. NVIDIA's reported Volcano bin-packing result is untested, and
the 34% figure attributed to it is unverified ([the thesis](#the-thesis)).

**What is worth noticing anyway**, because it held across both runs:

- **`D-largest` reports 0.0% fragmentation under definition A** and 55.6% under
  definition B, on the same data. The artefact reproduces on a real cluster
  exactly as it does in the reference model — which is the concrete argument
  that an unstated fragmentation definition is not a measurement. (It is also
  a property of the Phase 1 binder's global visibility: under the kube queue
  model the reference model's D-largest reads 23.1% under definition A — a
  second reason a fragmentation number needs its harness stated.)
- **`D-random` scored 92.8% on the gang deadlock rate** (95.7% in run 1),
  against 38.1% for K0 — but that metric is now **deprecated**, for a reason
  that applies directly to these numbers: its 60 s threshold equals one runner
  poll at speedup 60, so whether a gang bound on two consecutive polls counts
  as stalled depends on sub-second API latency rather than on the scheduler
  ([metrics.md](docs/metrics.md#deprecated-gang-deadlock-rate)). Every gang it
  counted in this table was `stalled` (all pods bound, bind spread above
  60 s) and none `deadlocked` — 129 of 139 for D-random, 53 for K0
  (`results/results.json`). Its replacement, stranded gang GPU-hours, has **no
  cluster measurement yet**. In the reference model (`src=model`, never
  comparable with this table) D-random strands 282.7 GPU-h against D-fifo's
  71.2 — a fourfold gap, so far shown in the model only.
- **K0 has the best mean wait of the four** (173 min) while finishing later than
  `D-fifo`. It is ordering work well and draining slowly — consistent with the
  backoff asymmetry in caveat 2, and the first thing K1 (Kueue) should change.

Kueue, Volcano, NVIDIA bin-packing, KAI/DRA and the Slurm control are not built.
Their rows do not exist yet rather than being empty, which is the honest way to
show an unbuilt comparison.

### Reference model — not a cluster run

The degenerate policies scored by the in-process reference model with the
Phase 2 defaults, produced on 2026-09-26 by exactly:

```bash
make bench-model   # = k8slab bench --reference-model --fleet fleets/default.yaml
                   #   --profile default --speedup 60 --repeat 5 --queue-model kube
                   #   --topology-penalty report --startup-delay 0:0 --results results-model
```

800 jobs / 1201 pods, 140-GPU default fleet, trace seed 0, 5 interleaved
repeats per configuration. **`src=model` on every row: never compare these with
a cluster row**, including the Run 2 table above.

| config    | src   | n | verdict       | makespan h | util %   | GPU-h used | mean wait m | p95 wait m | frag A % | frag B % | frag C % | gang strand GPU-h | size-wait ρ | starve ×   | place pen.* | fair      |
|-----------|-------|---|---------------|------------|----------|------------|-------------|------------|----------|----------|----------|-------------------|-------------|------------|-------------|-----------|
| D-fifo    | model | 5 | deterministic | 15.8±0.0   | 63.3±0.0 | 1404.7±0.0 | 122±0       | 520±0      | 29.0±0.0 | 55.7±0.0 | 14.0±0.0 | 71.2±0.0          | 0.56±0.00   | 7.85±0.00  | 1.55±0.00   | 1.00±0.00 |
| D-random  | model | 5 | unstable      | 15.6±0.8   | 64.3±3.1 | 1404.7±0.0 | 122±3       | 494±5      | 31.9±7.2 | 58.6±5.6 | 17.2±3.4 | 282.7±8.6         | 0.68±0.01   | 19.14±2.64 | 1.56±0.00   | 1.00±0.00 |
| D-largest | model | 5 | deterministic | 15.4±0.0   | 65.3±0.0 | 1404.7±0.0 | 126±0       | 476±0      | 23.1±0.0 | 53.6±0.0 | 14.6±0.0 | 80.7±0.0          | 0.58±0.00   | 7.20±0.00  | 1.52±0.00   | 1.00±0.00 |
| D-preempt | model | 5 | deterministic | 15.0±0.0   | 66.8±0.0 | 1404.7±0.0 | 202±0       | 435±0      | 4.0±0.0  | 42.5±0.0 | 12.4±0.0 | 71.3±0.0          | 0.28±0.00   | 1.59±0.00  | 1.57±0.00   | 1.00±0.00 |

\* ASSUMED topology penalty factors; not a measured slowdown. D-preempt evicted
562 pod attempts per run, losing 110.7 GPU-hours of Running work and holding
7.3 GPU-hours under grace locks (results.md, *Execution layer*); delivered
GPU-hours still equal the trace's 1404.7 because requeued work completes.
Three D-preempt columns depend on how evicted attempts are counted, and each
counts every attempt: `gang strand` includes the GPU-time members of evicted
gang attempts held before their gang assembled, not only the final attempt's
([metrics.md](docs/metrics.md#gang-behaviour)); `frag A` counts a pod as
pending only while no attempt of it was bound
([metrics.md](docs/metrics.md#definition-a--queue-relative)); and wait, and
the ρ and starvation columns built on it, count only the time a job was
pending — after an eviction it waits again, but the time an evicted attempt
was bound and Running is not waiting (its GPU-time is the 110.7 GPU-h lost;
[metrics.md](docs/metrics.md#wait-time)). How each was arrived at during
Phase 2, with the values the earlier counting gave, is in
[CHANGELOG.md](CHANGELOG.md).

How to read it:

- **`deterministic` is a property of the model, not of a cluster.** D-fifo,
  D-largest and D-preempt ignore the harness seed and, at this table's zero
  startup delay, nothing else in the model draws from it, so their five
  repeats are bit-identical (sd = 0). With a non-zero `STARTUP_DELAY` every
  policy's repeats vary. Only D-random's own randomness varies here, and it is
  `unstable` on all three fragmentation definitions (CV 22.6% for A, 9.5% for
  B, 19.9% for C): its fragmentation numbers cannot be quoted from five
  repeats, and results.md never calls a difference on them resolvable, even
  where the bootstrap interval excludes zero (D-fifo − D-random on C).
- **The queue model is a harness variable, and a large one.** The same command
  with `QUEUE_MODEL=none` (Phase 1's binder; into `results-model/queue-none`)
  gives D-fifo 14.0 h / 71.8% / mean wait 168 m and D-largest a definition-A
  reading of 0.0% with a starvation ratio of 0.53 — against 15.8 h / 63.3% /
  122 m, 23.1% and 7.20 here. With backoff a policy only orders the pods
  currently in activeQ: a large pod that failed to fit backs off while smaller
  ones are placed, so large jobs wait relatively longer under both policies
  (starvation ratio 1.77 → 7.85 for D-fifo, 0.53 → 7.20 for D-largest) and
  largest-first stops starving small jobs. Numbers from the two harnesses are
  not comparable, and results.json records which one produced each row.
- The resolvable differences in `results-model/results.md` between
  deterministic configurations are labelled `deterministic`: they are exact
  differences of two fixed numbers, not evidence about harness noise.

## Metrics

Defined in full, with formulas, in [docs/metrics.md](docs/metrics.md).
Limitations are in [docs/limitations.md](docs/limitations.md).

The one to read first is fragmentation, which is reported under **three**
definitions because they disagree:

- **A — queue-relative.** Free GPUs on nodes that cannot fit the *smallest
  currently-pending* pod.
- **B — queue-independent.** Free GPUs on nodes that cannot fit a fixed
  reference request (the largest per-pod ask in the trace).
- **C — structural.** Free GPUs inside a node (or rack, or switch) that has any
  GPU allocated. A pure function of capacity, topology and the free-GPU series,
  shared with [slurm-scheduler-lab][ssl] through a common golden test vector;
  the node level is the headline.

Under the Phase 1 harness, definition A reports **0.0%** for `D-largest` —
exactly 0.0 in the Run 2 cluster table, 0.003% before rounding in the reference
model with `--queue-model none` — which sounds like flawless packing and is an
artefact: largest-first starves small jobs, so a 1-GPU pod is essentially
always pending, so the smallest pending request is 1, so no free GPU is ever
"unusable". Under definition B the same runs read **55.6%** (cluster) and
**49.4%** (reference model), on identical data. Any published
fragmentation number that does not state its definition is not comparable to
anything — including the 34% figure this repo was started to test, which is
unverified and comes with no definition at all ([the thesis](#the-thesis)).

## Quickstart

No GPUs. No cloud account. One command each.

```bash
make up      # kind cluster + kwok + the simulated fleet, scheduler scaled to SPEEDUP
make bench   # replay the trace against every built configuration, REPEAT=5 times each
make down    # tear it all down
```

`make up` generates the speedup-scaled scheduler configuration into
`cluster/generated/` and creates the cluster from it; `make bench` refuses to
run if its `SPEEDUP` or `STARTUP_DELAY` differs from what the cluster was
brought up with. Five interleaved repeats of every configuration take roughly
five times as long as Phase 1's single pass. Cluster results go to
`results/phase2/` (`make bench`, and `k8slab bench` without `--results`);
`results/results.{md,json}` is the committed Phase 1 run. `k8slab bench`
refuses to write into a directory holding results of another schema or
source — such as that Phase 1 file — unless given `--force`, and `make clean`
deletes only reference-model results among results (and `cluster/generated/`
only while no `k8slab` kind cluster is up: that directory is mounted into the
running kube-scheduler and is what `make bench` checks `SPEEDUP` against;
`make down` removes it with the cluster).
Knobs: `REPEAT`, `QUEUE_MODEL` (kube|none), `TOPOLOGY` (off|report|extend),
`STARTUP_DELAY` (MIN:MAX real ms), `TRACE_SEEDS`, `SERIALISE=1`,
`FAIL_ON_UNSTABLE=1`; `make scheduler-config` prints the configuration.
None of this has been run against a cluster yet
([cluster/upstream.lock](cluster/upstream.lock) says `verified = false`).

Everything is pinned in [cluster/upstream.lock](cluster/upstream.lock).
Scheduler behaviour moves across minor versions, so an unpinned benchmark is
not a benchmark. `verified` in that file records whether a green run has
actually been observed on the pinned combination.

To score the degenerate policies without a cluster at all:

```bash
make bench-model
```

Rows produced that way are labelled `model` rather than `cluster` in every
table, and must never be compared against a cluster row.

## The fleet

Deliberately heterogeneous — 12x8 + 8x4 + 6x2 GPU nodes, 140 GPUs total — so
that request shapes and node shapes mismatch: a 4-GPU pod can fit an 8-GPU node
but not a 2-GPU one, and a scheduler's choice of where to put small jobs
matters. (This section used to say fragmentation is only measurable when nodes
differ. It is not: identical nodes fragment under a mixed trace too. On the
homogeneous fleet the reference model's C-node reads 16.3–25.2% across the
four degenerate policies and both queue models, differing by up to 7.9 points
between policies — see [docs/metrics.md](docs/metrics.md#the-control).)
`fleets/homogeneous.yaml` is the control: a workload of whole-node jobs on it
must report exactly zero structural fragmentation (definition C, node level),
and if it does not, the metric is wrong. It is a control for whole-node
workloads only. Under the default *mixed* trace it is not near zero, and the
definitions disagree about it: definition B reads lower than on the
heterogeneous fleet, but the headline C-node reads **higher** for every
deterministic policy under both queue models (reference-model numbers, and a
plausible but unmeasured reason, in
[docs/metrics.md](docs/metrics.md#the-control)).

Both fleets also declare a rack/switch topology and NVLink flags, emitted as
`topology.k8slab.io/*` node labels. That topology is a declared scenario for
measuring *where* schedulers place pods; the lab has no fabric.

## Phase 2 methodology

What changed in how a number is produced, each item with where it is defined.
The mechanisms exist and are tested; **none has been exercised on a cluster**.

- **Repeats and spreads.** Every configuration is run `REPEAT` times (default
  5), interleaved across configurations, with the trace seed fixed and a
  derived per-repeat harness seed (repeat 1 uses `--seed` itself). Results are
  `mean±sd` with `n` and a verdict — `unreplicated` (n=1, never stable),
  `unstable` (a gated metric's sd above its display resolution *and* CV above
  `--sigma-tolerance`, 5%), `deterministic` (bit-identical repeats) or
  `stable` — and results are rewritten after every run.
  `--trace-seeds` adds a separate across-trace study (one run per
  configuration per trace, compared paired by trace; its spread mixes workload,
  harness and policy randomness and gets no stability verdict).
  [metrics.md](docs/metrics.md#repeated-runs-mean--sd-and-the-stability-verdict)
- **Bootstrap screen, not a significance test.** Differences are called
  resolvable only if a seeded percentile-bootstrap interval excludes zero with
  n ≥ 5 on both sides and neither side is `unstable` on that metric — a screen
  with a measured ~13% false-alarm rate at n=5 (`scripts/bootstrap_null.py`),
  not a 5% test.
  [metrics.md](docs/metrics.md#pairwise-differences-the-bootstrap)
- **Queue parity.** The degenerate binder goes through kube-scheduler's queue
  mechanics (activeQ, backoffQ, the unschedulable pool, exponential backoff,
  requeue on freed capacity, one pod per cycle) with the same simulated
  backoff as the scaled kube-scheduler. `--queue-model none` is Phase 1.
  [limitations.md](docs/limitations.md#queue-mechanics-parity-not-a-tuned-penalty)
- **Time scaling and its residual.** kube-scheduler's backoff and
  unschedulable-pool timeout are divided by the speedup; the integer backoff
  fields leave a residual (60 simulated seconds instead of 1–10 at speedup
  60, against 600 unscaled), and the unscalable flush timers gate both
  values: at speedup 60 a backoff is experienced as 60 to under 120 simulated
  seconds, and a pod no event releases leaves the unschedulable pool after
  more than 300 and at most 2100 (uncompressed: 300–330). All of it is
  recorded in every results.json. A cluster results file states the
  kube-scheduler configuration only as configured by `make up`
  (`timescale.status: recorded`), because its record,
  `cluster/generated/cluster-state.json`, is written before the cluster is
  created and nothing reads the running scheduler's flags back; without the
  record the configuration is stated as unknown.
  [limitations.md](docs/limitations.md#time-is-compressed)
- **Startup delay.** `STARTUP_DELAY=MIN:MAX` real milliseconds (kwok Stage
  delay on a cluster, a seeded draw in the model); 0:0 is the default, and
  50:200 is an illustrative, uncalibrated setting — a judgment call, not a
  measured start-up latency. Runtime counts from Running; bind to Running is
  `startup_overhead_gpu_hours`.
  [limitations.md](docs/limitations.md#startup-latency-is-modelled-not-measured)
- **Topology modes.** `report` (default: ASSUMED factors only in
  `placement_penalty_mean`), `off`, or `extend` — a SCENARIO that stretches
  runtimes by the factors; its rows are labelled `+topo` and never pooled with
  or ranked against measured rows.
  [metrics.md](docs/metrics.md#topology-and-placement)
- **Definition C.** Structural fragmentation per topology level — free
  GPU-time inside a node (rack, switch) with anything allocated — the
  definition shared with slurm-scheduler-lab through a golden vector; the node
  level is the headline. [metrics.md](docs/metrics.md#definition-c--structural-per-topology-level)
- **Stranded GPU-hours.** GPU-time gang members hold before every member is
  Running, replacing the thresholded gang deadlock rate (deprecated).
  [metrics.md](docs/metrics.md#gang-behaviour)
- **Head-of-line and size bias.** Wait by job footprint, the size–wait rank
  correlation and the large-job starvation ratio, with a flag for
  "utilization bought by starving large jobs".
  [metrics.md](docs/metrics.md#admission-delay-by-job-size-head-of-line-blocking-and-starvation)
- **Preemption.** D-preempt evicts; evicted GPUs stay locked for the grace
  period (default 30 simulated s) and victims restart from zero.
  PriorityClasses keep `preemptionPolicy: Never` for everything else.
  [limitations.md](docs/limitations.md#preemption)
- **Diagnostics, not headline.** `SERIALISE=1` (kube-scheduler
  `parallelism: 1`) and `--admission-gate` change what is measured; their rows
  are labelled `+gated` and excluded from comparisons.
  [limitations.md](docs/limitations.md#serialisation-and-the-admission-gate-are-diagnostics)

## Layout

```
cluster/       kind config (generated/ holds make up's scaled scheduler config) and the version lock
fleets/        fleet definitions (node classes, GPU counts, declared topology)
src/k8slab/    fleet loader, trace generator, policies, queue model, binder,
               runner, reference model, metrics, statistics, report
docs/          metric definitions and limitations
scripts/       reproductions of numbers quoted in the docs (bootstrap_null.py,
               extend_prefix_rule.py)
tests/         unit tests; no cluster required (the runner is tested against a fake API)
```

## Related

- [slurm-scheduler-lab][ssl] — the Slurm side. Same trace vocabulary; S0 will
  use it as the cross-substrate control.
- [slurm-rca-bench][rca] — where the degenerate-baseline and
  measured-versus-asserted conventions come from.

## License

MIT.

[kwok]: https://kwok.sigs.k8s.io/
[kueue]: https://kueue.sigs.k8s.io/
[volcano]: https://volcano.sh/
[kai]: https://github.com/NVIDIA/KAI-Scheduler
[nv-volcano]: https://developer.nvidia.com/blog/practical-tips-for-preventing-gpu-fragmentation-for-volcano-scheduler/
[gheware]: https://devops.gheware.com/blog/posts/top-kubernetes-integrations-ai-gpu-acceleration-2026.html
[nv-kubecon]: https://blogs.nvidia.com/blog/nvidia-at-kubecon-2026/
[kccnc]: https://kccnceu2026.sched.com/
[ssl]: https://github.com/Zhanyl-tech/slurm-scheduler-lab
[rca]: https://github.com/Zhanyl-tech/slurm-rca-bench
