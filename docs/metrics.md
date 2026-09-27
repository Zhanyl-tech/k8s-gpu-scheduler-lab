# Metric definitions

Every formula the lab reports, written out. Quote a number from this repo only
together with the definition it was computed under.

Notation. The fleet has nodes $n$ with GPU capacity $C_n$; $\text{free}_n(t)$ is
the GPUs free on node $n$ at time $t$. $H$ is the horizon — the simulated time
at which the run stopped. Integrals over the free-GPU sample series (the runner
records one sample per poll, the reference model one per tick) are **left-point
step integrals**: each sample's value holds until the next sample's timestamp,
and the final sample only closes the last interval.

> Correction. Earlier versions of this page said "trapezoids". The code has
> always integrated left-point steps (`free0 * dt` in `metrics.compute`), and
> that is what every published fragmentation number was computed with. The page
> was wrong, not the numbers.

Contents: [utilization](#gpu-hours-used-gpu-hours-idle-utilization) ·
[wait](#wait-time) · [admission delay by job size](#admission-delay-by-job-size-head-of-line-blocking-and-starvation) ·
[fragmentation A, B, C](#fragmentation--three-definitions-and-they-disagree) ·
[gang behaviour](#gang-behaviour) · [topology and placement](#topology-and-placement) ·
[fairness](#fairness) · [preemption](#preemption) ·
[execution layer](#execution-layer-startup-overhead-evictions-topology-extension) ·
[repeated runs](#repeated-runs-mean--sd-and-the-stability-verdict) ·
[pairwise differences](#pairwise-differences-the-bootstrap) ·
[where the numbers came from](#where-the-numbers-on-this-page-came-from)

---

## GPU-hours used, GPU-hours idle, utilization

$$\text{GPU-hours used} = \frac{1}{3600}\sum_{p \in \text{pods}} g_p \cdot (\text{end}_p - \text{start}_p)$$

where $g_p$ is the GPUs pod $p$ requested. Capacity is $\left(\sum_n C_n\right)
\cdot H$, idle is capacity minus used, and

$$\text{utilization} = \frac{\text{GPU-hours used}}{\text{capacity}}$$

**A pod's runtime is the duration the trace assigned it**, counted from the
moment it was observed Running, not the wall-clock between this runner noticing
the binding and noticing the deletion. Using the observed interval adds up to
one poll period of phantom GPU time per pod — measured at **+11.9%** of total
GPU-hours at `--speedup 400`, which is larger than any effect this lab tries to
detect. The check that catches this: a run in which every job completes must
report exactly the GPU-hours the trace demands (`gpu_hours_demanded`). It does
(`test_gpu_hours_equal_trace_demand_when_all_jobs_complete`, and the runner
against a fake API server in `tests/test_runner.py`).

Two Phase 2 refinements of that invariant:

- **Extension scenario.** With `--topology-penalty extend` runtimes are
  stretched by ASSUMED placement factors, so delivered GPU-time exceeds demand
  by exactly the stretch: $\text{used} = \text{demanded} +
  \text{topology\_extension\_gpu\_hours}$ when every job completes. A job whose
  pods all bind in one pass runs exactly $d_j \cdot f_j$ ("demand × realised
  factor"); a gang member already Running when its last sibling binds is
  stretched only for its remaining work. Tested in both paths
  (`test_extend_delivers_demand_times_realised_factors`,
  `test_extend_mode_stretches_runtime_by_the_placement_factor`).
  $f_j$ is the job's placement factor over **every** member's node, including
  a member that already finished before its last sibling bound — a gang that
  never co-ran. That member has no remaining work and is not stretched; the
  members still bound are. Under an earlier rule, a job was stretched only if
  every member was *still bound* when the last one bound, so such a gang was
  not stretched at all, which exempted exactly the worst co-scheduling. That
  rule was replaced before this branch's first commit, so no committed tree
  contains it; `scripts/extend_prefix_rule.py` rebuilds it from the current
  `sim.py` (it swaps back that one condition and refuses to run if the
  condition has changed) and replays both rules. In one reference-model run
  (default fleet and trace, seed 0, harness seed 0, kube queue, `extend`) the
  gangs with $f_j > 1$ left entirely unstretched numbered 89 for D-random
  against 24–34 for the other three policies under the old rule, and extension
  read D-random 150.5 GPU-h against D-fifo 440.7, D-largest 444.2 and
  D-preempt 462.7; under the current rule none is left unstretched and the
  four read 280.1, 493.5, 482.3 and 533.7 (makespans 18.66, 19.36, 18.77 and
  19.12 h, from 17.24, 18.87, 18.36 and 18.90). Nothing is stretched while a
  job's placement is still partial: a member bound early runs at factor 1
  until the last member binds, then its remaining work takes the whole
  placement's factor
  (`test_extend_stretches_nothing_until_the_last_gang_member_is_bound`, in
  both paths). On a cluster the runner also stopped applying the
  factor to a member whose work ended since its last poll, which moved that
  member's end to the poll: phantom Running time booked as extension
  (`test_extend_never_moves_the_end_of_a_member_whose_work_already_ended`).
- **Evictions.** Delivered GPU-hours count useful work only: an evicted
  attempt's Running time is *lost* (`preempted_gpu_hours_lost`), except the
  share a checkpoint kept, which is delivered. So with requeued evictions the
  invariant still holds exactly. A bare pod the control plane destroyed is
  never recreated; its Running time is lost as well, so delivered GPU-hours
  fall short of demand by that pod's whole duration. (`jobs_completed` still
  counts its job, because it only asks whether every pod was bound; the pod is
  flagged `preempted` and counted in `preemptions`.)

Utilization is still a real measurement of scheduling quality, because the
horizon is not fixed — a scheduler that packs badly stretches $H$ and dilutes
its own utilization.

## Wait time

$$\text{wait}_j = \max_{p \in \text{pods}(j)} \text{scheduled}_p - \text{submit}_j$$

**To the last pod, not the first.** A gang that is half-placed has not started.
Crediting a job at its first binding would flatter exactly those schedulers that
admit gangs partially — the failure mode this lab exists to measure.

**Under eviction: pending time only.** A job with an evicted-and-requeued
attempt (D-preempt, or the model's eviction API) waited only while some pod of
it was *pending* — submitted and not bound. Its wait is the length of the union
of its pods' pending intervals $[\text{submit}_j, b_1) \cup [e_1, b_2) \cup
\dots \cup [e_k, \text{scheduled}_p)$, the same intervals
[Definition A](#definition-a--queue-relative) uses, still ending at the last
pod's final bind. Without evictions that union is exactly the formula above,
and the code keeps that formula for such jobs, so every run without evictions
is bit-identical to Phase 1. So the time a job waits in the queue again after
an eviction counts; the time an evicted attempt was bound or Running does not
— its cost is `preempted_gpu_hours_lost` and, through the rerun, makespan. A
gang is admitted only while every member is bound: a partially bound gang that
is evicted was never admitted and waits throughout.

> Correction. Until this revision wait was the formula above for every job.
> Each `PodEvent` describes a pod's *final* attempt, so for an evicted job
> "last bind − submit" counted every earlier attempt's bound and Running time
> as waiting, and fed that into every size-bias metric below. In the reference
> model (default kube harness, seed 0) D-preempt read mean wait 205.2 min; the
> pending-only value is 201.8 min. Its size–wait ρ moves from 0.27 to 0.28
> and its starvation ratio from 1.56 to 1.59 (1.62 to 1.66 under
> `--queue-model none`). No other policy evicts, so no other number moved.
> Wait to a job's *first* admission is a different quantity, and it is not
> reported: it would leave out the waiting after an eviction, which is real
> queueing.

Reported as mean and p95. The p95 is **not** nearest-rank, although this page
used to say so: it is Phase 1's rounded rank, the value of rank
$\operatorname{round}(0.95\,n)$ with Python's round-half-to-even, where
nearest-rank takes rank $\lceil 0.95\,n \rceil$. The two differ for some $n$
(for example every $n$ from 11 to 19). `p95_wait` keeps the rounded rank so every
Phase 1 wait number stays bit-identical (`metrics.percentile`); every percentile
added since — `wait_by_footprint` p95, `gang_assembly_p50`/`p95` — is true
nearest-rank (`metrics.nearest_rank`).

Wait is quantized to the poll interval, which is `speedup × TICK` simulated
seconds — 200 s at `--speedup 200`. Differences between configurations smaller
than that are not measurements.

## Admission delay by job size: head-of-line blocking and starvation

A scheduler can raise utilization by letting small jobs flow around large ones.
That is sometimes good packing and sometimes starvation, and the averages above
cannot tell which. These metrics can. Per job $j$, admission delay is
$\text{wait}_j$ exactly as defined above (to the last pod; pending time only
for an evicted job), and the job's footprint is its **total** GPUs,
$G_j = g_j \cdot \text{gang\_size}_j$.

**By footprint bucket.** Jobs are bucketed by $G_j$ into 1, 2, 3–4, 5–8, 9–16
and 17+ GPUs; each bucket reports `jobs`, `admitted` (every pod bound), and the
mean and nearest-rank p95 wait over admitted jobs (`wait_by_footprint`).
Zero-GPU jobs do not compete for GPUs and are left out.

**Size–wait rank correlation** (`footprint_wait_spearman`):

$$\rho = \text{Pearson}\big(\text{rank}(G),\ \text{rank}(\text{wait})\big)$$

over admitted jobs, with tied values given the mean of the ranks they span.
$\rho > 0$: bigger jobs wait longer. $\rho < 0$: smaller jobs wait longer —
what largest-first does under the Phase 1 binder, which shows every pending pod
to the policy on every pass (reference model, `--queue-model none`: D-largest
$\rho = -0.36$). Under the default kube queue model D-largest only orders the
pods currently in activeQ and reads $\rho = +0.58$ (both: default fleet and
trace, seed 0, five repeats, not a cluster). `null` when fewer than two jobs were admitted or either
variable is constant. Stdlib only; on Python 3.12+ a test checks it against
`statistics.correlation(..., method="ranked")`.

The plan called this a "head-of-line blocking index". It is not named that,
because it does not isolate head-of-line blocking: strict FIFO that stalls
behind an unplaceable head delays *every* job behind it regardless of size,
which raises mean wait while leaving $\rho$ near 0. What $\rho$ measures is size
bias in admission, so that is what it is called.

**Large-job starvation ratio** (`large_job_starvation_ratio`):

$$\text{starvation} = \frac{\operatorname{mean}\{\text{wait}_j : G_j \ge C_\text{max}\}}{\operatorname{mean}\{\text{wait}_j : G_j = 1\}}$$

where $C_\text{max}$ is the fleet's largest single-node GPU capacity
(`large_job_gpus`, 8 on both shipped fleets): a job needing at least that many
GPUs in total needs at least one whole largest node — it can start only once
such a node, or several nodes, has drained. That is what "large" means here. It
does **not** mean the job cannot fit on one node: a job of exactly $C_\text{max}$
GPUs fits one empty largest node, and on the default trace (seed 0) 103 of the
129 "large" jobs need exactly 8 GPUs, 64 of them as a single pod. (This page and the code comments used to say "cannot be
satisfied by any one node", which is false at the threshold.) `null` when
either set is empty or the 1-GPU mean is 0 — a ratio against nothing is not
infinitely bad, it is undefined.

**Right-censoring.** Jobs never admitted have no wait and are excluded from all
three, so a scheduler that never admits its large jobs *looks better* on them.
The per-bucket `jobs − admitted` count is the corrective, and results.md prints
it whenever it is non-zero.

**The flag.** results.md flags any configuration whose mean utilization beats
D-random's while its mean starvation ratio exceeds **2.0** *and* exceeds
D-random's own ratio — the "inflates utilization by starving large jobs"
pattern, which is relative: a lead cannot have been bought from D-random by
starving large jobs if D-random starves them more. The flag prints both ratios
and says whether the utilization lead is itself resolvable (see
[pairwise differences](#pairwise-differences-the-bootstrap)); only a resolvable
lead "matches" the pattern, otherwise the pattern "is not established". A
configuration over 2.0 with a lead that this rule does not flag is listed
under the flags as not flagged, with D-random's ratio, so nothing is hidden.

> Correction. The flag used to fire on the absolute 2.0 alone and assert
> "This is the ... pattern" whether or not the lead was resolvable. In the
> reference model (default kube harness, seed 0) it flagged D-largest — ratio
> 7.20, lead +0.9 pp, not resolvable — against a D-random at 19.14, and on the
> homogeneous fleet D-largest at 6.86 against a D-random at 15.51. Both are
> now listed as not flagged.

2.0 ("large jobs wait more than twice as long as single-GPU jobs") is a
**judgment call, not a calibrated threshold**: some excess is expected from any
scheduler because a large job needs a whole node to drain, and this lab has not
measured how much. It is a parameter of `report.build_report`, and the ratio is
always printed so a reader can draw their own line. For scale only, the
reference model on the default fleet and trace (seed 0, means over five
repeats, not a cluster) gives D-fifo 1.77, D-random 17.51 (sd 1.67), D-largest
0.53 and D-preempt 1.66 under the Phase 1 binder (`--queue-model none`), and
7.85, 19.14 (sd 2.64), 7.20 and 1.59 under the default kube queue model — the
harness moves the ratio by more than the threshold.

## Fragmentation — three definitions, and they disagree

There is no canonical definition in the literature, and the vendor claims that
motivated this repo do not state theirs. So the lab reports three.

### Definition A — queue-relative

A free GPU is *stranded* if it sits on a node that cannot fit the smallest
pod currently pending:

$$m(t) = \min\{\, g_p : p \text{ pending at } t \,\}, \qquad
S_A(t) = \sum_n \text{free}_n(t) \cdot \mathbb{1}\!\left[0 < \text{free}_n(t) < m(t)\right]$$

$$\text{frag}_A = \frac{\int_0^H S_A(t)\,dt}{\int_0^H \sum_n \text{free}_n(t)\,dt}$$

If nothing is pending, $S_A(t) = 0$: capacity nobody wants is idle, not
fragmented. Without that rule an empty cluster reports maximal fragmentation.

A pod is pending at $t$ when it has been submitted and is not bound at $t$.
Without evictions that is the half-open interval $[\text{submit}_j,
\text{scheduled}_p)$. A pod evicted and requeued $k$ times was bound during each
earlier attempt, so it is pending on $[\text{submit}_j, b_1) \cup [e_1, b_2)
\cup \dots \cup [e_k, \text{scheduled}_p)$, with $b_i$, $e_i$ the bind and
eviction times of attempt $i$ (its `Eviction` records). Until this revision the
evicted attempts were ignored and the whole $[\text{submit}_j,
\text{scheduled}_p)$ counted as pending — including time the pod held GPUs.
For D-preempt in the reference model (default kube harness, seed 0) that read
4.08% where the eviction-aware value is 3.98%. A pod the control plane
destroyed is its own final attempt and is never pending again: its interval
ends at its bind or, if it was destroyed before any bind was observed (seen
Terminating first, or vanished never seen bound), at its destruction. Until
this revision such a pod had no bind to end the interval, so its request stayed
in $m(t)$ until the horizon
(`test_definition_a_stops_counting_a_pod_destroyed_before_its_bind_was_seen`:
on one 8-GPU node with 2 GPUs free for 100 s, a 4-GPU pod destroyed at 10 s
stranded 200 GPU-s instead of 20). Only a cluster run can produce such a pod;
the reference model never destroys one, so no model number on this page moved.

$m(t)$ is computed for every sample by one sorted sweep over those intervals
rather than by rescanning every pod at every sample, which it did until an
earlier revision. Without evictions the sweep applies the same predicate to the
same floats, and `tests/test_metrics_equivalence.py` keeps the old scan
verbatim and checks every Phase 1 metric is bit-identical (`==`, not
approximately equal) on reference-model runs of every degenerate policy on both
fleets, on the full 800-job default trace, and on hand-built edge cases. With
evictions it is checked against a pod-by-pod scan of the definition above
(`test_definition_a_sweep_matches_the_definition_on_a_preempting_run`).

### Definition B — queue-independent

Same shape, but against a fixed reference request $R$ — the largest per-pod ask
anywhere in the trace — instead of the live queue:

$$S_B(t) = \sum_n \text{free}_n(t) \cdot \mathbb{1}\!\left[0 < \text{free}_n(t) < R\right]$$

$$\text{frag}_B = \frac{\int_0^H S_B(t)\,dt}{\int_0^H \sum_n \text{free}_n(t)\,dt}$$

### Definition C — structural, per topology level

The fleet's shape, with no reference to the queue or to any request size. At
level $L \in \{\text{node}, \text{rack}, \text{switch}\}$, every node $n$
belongs to one level-$L$ domain $D_L(n)$ (at the node level, the node itself). A
domain is **carved** at time $t$ if any node inside it has any GPU allocated:

$$\text{carved}_L(D, t) = \exists\, m \in D : \text{free}_m(t) < C_m$$

$$S_C^L(t) = \sum_n \text{free}_n(t) \cdot \mathbb{1}\!\left[\text{carved}_L(D_L(n), t)\right],
\qquad
\text{frag}_C^L = \frac{\int_0^H S_C^L(t)\,dt}{\int_0^H \sum_n \text{free}_n(t)\,dt}$$

and $\text{frag}_C^L = 0$ when the denominator is 0. At the node level a fully
allocated node contributes 0 free GPUs and an idle node is not carved, so
$\text{frag}_C^\text{node}$ is simply *the share of free GPU-time that sits on
partly allocated nodes*. At the rack level it is the share not inside a wholly
idle rack; at the switch level, not inside a wholly idle switch.

It is implemented once, in `src/k8slab/fragmentation.py`, as a pure function of
(per-node capacity, per-node domains, free-GPU sample series) with no
dependency on the observation, the queue or the rest of the lab. It refuses
malformed input rather than guessing: every sample — including the last one
and any followed by a zero-length interval, neither of which is integrated —
must give every node, and no other, a free count in $[0, C_n]$. (Until this
revision only samples that opened a positive-length interval were checked.) It
is standalone so that [slurm-scheduler-lab][ssl] can implement the identical
function. Both labs test
against the same golden vector, copied verbatim into
`tests/test_fragmentation.py`:

| node | rack | switch | capacity | free, t=0 | free, t=10 | free, t=20 (horizon) |
|---|---|---|---|---|---|---|
| n0 | r0 | s0 | 8 | 8 | 8 | 8 |
| n1 | r0 | s0 | 8 | 3 | 8 | 8 |
| n2 | r1 | s0 | 4 | 4 | 4 | 4 |
| n3 | r1 | s0 | 4 | 0 | 4 | 4 |

Total free GPU-time is $15 \cdot 10 + 24 \cdot 10 = 390$. From 0 to 10, n1 and
n3 are carved: at the node level that is 3 free GPUs; at the rack level both
racks are carved and all 15 free GPUs count; the single switch likewise. From
10 to 20 nothing is allocated and nothing is carved. So
$\text{frag}_C^\text{node} = 30/390 = 1/13 \approx 0.0769231$ and
$\text{frag}_C^\text{rack} = \text{frag}_C^\text{switch} = 150/390 = 5/13 \approx
0.3846154$.

**Properties, all tested.** $\text{frag}_C^\text{node} \le \text{frag}_C^\text{rack}
\le \text{frag}_C^\text{switch}$ whenever racks nest in switches (the derivation
guarantees nesting and the function rejects a non-nesting mapping, since an
inversion would otherwise be possible); an idle fleet reads 0 at every level; a
homogeneous fleet running only whole-node jobs reads exactly 0 at the node
level. On a single-shape fleet whose largest pod is one node,
$\text{frag}_C^\text{node}$ coincides with definition B (a node is carved exactly
when $0 < \text{free} < R$); on the heterogeneous default fleet it does not,
which is why both are reported.

**Why C is the one that maps onto Slurm.** Slurm has no pods and no
Kubernetes queue. What it does have is a whole-node allocation boundary,
`sbatch --exclusive`: "The job allocation can not share nodes ... with other
running jobs" ([sbatch](https://slurm.schedmd.com/sbatch.html), the Slurm
26.05 page, read 2026-09-26). Whether jobs get it is a site and partition
choice, not a universal practice: the same entry says "The default
shared/exclusive behavior depends on system configuration and the partition's
OverSubscribe option takes precedence over the job's option". (This paragraph
used to call it the boundary "every GPU site uses", which that page does not
support.) $\text{frag}_C^\text{node}$ is exactly the free GPU-time unavailable
to a whole-node `--exclusive` request, wherever a partition honours one.
$\text{frag}_C^\text{rack}$, reading the lab's rack as a `topology/tree` leaf
switch (the `SwitchName` line whose `Nodes` are the "Child nodes of the named
leaf switch", [topology.conf](https://slurm.schedmd.com/topology.conf.html)),
is the free GPU-time unavailable to a job that needs an **entire idle** leaf
switch. It is *not* what a `--switches` job cannot use. `--switches` "defines
the maximum count of leaf switches desired for the job allocation" — a cap on
how many leaf switches an allocation spans — and the tree plugin places a job
by finding "the lowest level switch in the hierarchy that can satisfy a job's
request and then allocate resources on its underlying leaf switches using a
best-fit algorithm" ([topology guide](https://slurm.schedmd.com/topology.html),
read 2026-09-26). Neither page says such a job needs an idle switch or keeps
other jobs off it, so a `--switches=1` job can use free GPUs in a leaf switch
other jobs partly hold — GPUs C-rack counts as carved. Whether any Slurm option
requests an entire idle leaf switch under `topology/tree` is **unverified**:
`--exclusive=topo` refuses to share a "topology segment", but the topology
guide offers segments (`--segment`) only "When a block, ring, or torus3d
topology is configured". (This paragraph used to pair the leaf switch with
`--switches` as a second allocation boundary, and so described C-rack as what
a `--switches` job could not use. That was wrong.) Both levels are computable
from the data each lab already has; only the node level has a verified Slurm
counterpart, which is one more reason it is the headline. Definition A
needs the pending queue, whose contents differ between Slurm backfill and a
Kubernetes scheduler for reasons that have nothing to do with fragmentation;
definition B needs a "largest per-pod request", and a Slurm job has no pods.

**Headline: the node level** (`fragmentation_structural`; the other two are
`fragmentation_structural_rack` and `fragmentation_structural_switch`).
Reasons, in order:

1. It depends only on node capacities — facts of the API objects. Rack and
   switch levels depend on the *declared* topology (`nodesPerRack`,
   `racksPerSwitch`), which is a scenario choice; change it and they move.
2. It passes the control exactly: whole-node jobs on the homogeneous fleet give
   0. The rack level cannot, because one busy node carves its idle neighbours.
3. It maps onto the one boundary both labs share without any topology file,
   and the only one with a verified Slurm counterpart (`--exclusive`, which a
   job requests and a partition's OverSubscribe setting can override).
4. Coarse levels saturate. With few domains, "some GPU allocated somewhere in
   this switch" is true for most of any busy run; the reference model's switch
   level reads 54–76% for the four degenerate policies on the default fleet
   under the default kube harness (D-largest 53.9%, D-fifo 62.2%, D-random
   65.6% with sd 10.4, D-preempt 75.7%) and 59–72% with `--queue-model none`
   (means over five repeats, seed 0).
5. The claim this repo exists to test speaks of "GPU node fragmentation". That
   is a reading of an unstated definition, not a reproduction of it.

**What C does not say.** It is structural, not "unusable": a partly allocated
node's free GPUs may be exactly what the queue wants (definition A's
question). And like every ratio against free GPU-time it can be flattered by
withholding work — more idle nodes, lower ratio — so read it next to
utilization and wait, never alone.

**Is the definition flawed?** It was considered carefully before implementing
it verbatim. Two properties could be mistaken for flaws: a level with a single
domain degenerates into "was anything allocated anywhere" (the shipped fleets
therefore give every level at least three domains), and C penalises partial
nodes whose free GPUs are usable (by design; A covers usability). Neither
changes the golden values, and neither is fixed by a variant that would not
also make the two labs harder to compare, so no variant was added.

### Why several

**Definition A reports 0.0% for `D-largest`** under the Phase 1 harness, i.e.
with every pending pod visible to the policy on every pass. Two sources, two
slightly different readings:

- the **Run 2 cluster table** (`results/results.json`): exactly 0.0, and 55.6%
  under definition B;
- the **reference model** with `--queue-model none`: 0.003% before rounding —
  not identically zero only because of brief windows at the start and tail of a
  run with nothing small queued — and 49.4% under definition B.

Under the kube queue model D-largest only orders the pods currently in activeQ,
small pods stop being permanently pending, and the reference model reads 23.1%
— see the README's reference-model table; this is a second reason a
fragmentation number needs its harness stated. Largest-request-first (under
the Phase 1 binder) starves small jobs, so a 1-GPU pod is essentially always
pending, so $m(t) = 1$, so no free GPU is ever below the threshold and nothing
is ever stranded. The metric says the worst-behaved policy in the set has
flawless packing.

Under definition B the same runs read **55.6%** (cluster) and **49.4%**
(reference model): against 0.0% and 0.003%, computed from identical data.

Neither is wrong. A measures "could the queue have used this right now", B
measures "is this fleet carved into shapes too small for the biggest ask", C
measures "how much free capacity is not in whole idle domains". They answer
different questions and a policy can look excellent under one and mediocre
under another: in the reference model on the default trace, D-fifo has a lower
node-level C than D-largest but a higher B — 14.5% against 21.2% under C-node
and 52.4% against 49.4% under B with `--queue-model none`, 14.0% against 14.6%
and 55.7% against 53.6% under the default kube queue model (means over five
repeats; both deterministic). **This is the reason a fragmentation claim without a stated
definition — including the 34% figure this repo was started to test — cannot be
reproduced or disputed.** That figure is unverified: the primary NVIDIA source,
[*Practical Tips for Preventing GPU Fragmentation for Volcano
Scheduler*][nv-volcano] (2025-03-31, read 2026-09-26), contains no 34% and no
fragmentation formula. What it reports is a count — nodes with all four GPUs
free, 18 before bin-packing and 214 after, on one DGX Cloud cluster of L40S
nodes, against Volcano's default placement — and average GPU utilization of
roughly 90%. Provenance in the [README](../README.md#the-thesis).

The denominator matters too. Against free GPU-time the number describes the
quality of the free pool; against total fleet GPU-time it describes how much of
the machine is lost. The lab reports A both ways (`fragmentation_rate` and
`fragmentation_of_fleet`), and they differ by the idle share of the fleet:

$$\text{fragmentation\_of\_fleet} = \text{fragmentation\_rate} \times
\frac{\int_0^H \sum_n \text{free}_n(t)\,dt}{\left(\sum_n C_n\right) H}$$

exactly, and that share is close to $1 - \text{utilization}$ but not equal to
it: GPUs that are allocated yet deliver nothing — bound but not yet Running,
grace-locked, or Running work an eviction threw away — are neither free nor
delivered (`test_the_two_definition_a_denominators_differ_by_the_idle_share`).
This page used to say the two differ by "roughly the utilization factor",
which is the other share: D-fifo in the README's reference-model table runs
at 63.3% utilization, and its `fragmentation_of_fleet` is 0.37 of its
`fragmentation_rate`, not 0.63.

### The control

`fleets/homogeneous.yaml` has one node shape. There, definition B and C-node
coincide, and **a workload of whole-node jobs must read exactly 0** under C-node
(`test_homogeneous_fleet_running_only_whole_node_jobs_has_zero_node_level`);
if it does not, the implementation is wrong.

> Correction. This section used to say the homogeneous fleet should read near
> zero under either definition. Under the default *mixed* trace it does not:
> 1-, 2- and 4-GPU jobs leave 8-GPU nodes partly allocated. What holds under a
> mixed trace is that **definition B** reads lower on the homogeneous fleet
> than on the heterogeneous one — the existing
> `test_homogeneous_fleet_barely_fragments_under_definition_b` asserts only
> that. **Definition C-node does not**: it reads *higher* on the homogeneous
> fleet for every deterministic policy, under both queue models. A previous
> version of this correction compared homogeneous C-node with default-fleet B,
> which hid that. Reference model, default trace, seed 0, means over five
> repeats (D-random: mean ± sd; it is `unstable`, so its row does not rank):
>
> | fleet, queue model | D-fifo | D-random | D-largest | D-preempt |
> |---|---|---|---|---|
> | homogeneous, none: B = C-node | 18.4 | 18.5 ± 3.4 | 25.2 | 17.3 |
> | default, none: B | 52.4 | 63.6 ± 1.7 | 49.4 | 48.4 |
> | default, none: C-node | 14.5 | 19.4 ± 0.5 | 21.2 | 13.6 |
> | homogeneous, kube: B = C-node | 16.3 | 22.5 ± 5.2 | 19.2 | 20.3 |
> | default, kube: B | 55.7 | 58.6 ± 5.6 | 53.6 | 42.5 |
> | default, kube: C-node | 14.0 | 17.2 ± 3.4 | 14.6 | 12.4 |
>
> A plausible reason, not separately measured: on the heterogeneous fleet small
> jobs can fill the 2- and 4-GPU nodes whole, leaving no partly allocated node
> behind, while on a fleet of 8-GPU nodes every small job carves an 8-GPU node.
> Either way the homogeneous fleet is a control for *whole-node* workloads
> (exactly 0 under C-node, tested), not a lower bound for mixed ones.

## Gang behaviour

A gang job has `gang_size > 1` pods that must all run together. Until the last
member is running, every earlier member holds GPUs that do nothing. That cost is
rarely reported next to the benefit of gang scheduling, and it is continuous —
so it is measured continuously.

For gang $j$ with per-pod GPUs $g_j$, and for each member $p$: $b_p$ is the
bind time (`PodEvent.scheduled_time`), $s_p$ the time it was observed Running
(`PodEvent.start_time`), $e_p$ its end, or $H$ if it never ended. A member holds
its GPUs on $[b_p, e_p)$ — from bind, because a bound pod's GPUs are allocated
whether or not anything runs. The gang is **assembled** at

$$A_j = \max_p s_p \quad \text{once every member has started.}$$

**Stranded gang GPU-time** is held time before assembly, each member capped at
its own end:

$$\text{stranded}_j =
\begin{cases}
\sum_p g_j \cdot \max\!\big(0,\ \min(e_p, A_j) - b_p\big) & \text{assembled} \\
\sum_{p \text{ bound}} g_j \cdot \max\!\big(0,\ \min(e_p, H) - b_p\big) & \text{never assembled}
\end{cases}$$

A member that finished before the last member started was stranded for its
whole life. A gang that never assembles is stranded for every member's whole
held time.

**Every attempt counts.** The formula above is applied to the final attempt
(the `PodEvent`s) *and* to every attempt that was evicted and requeued (the
`Eviction` records, grouped by job and eviction time — an eviction takes a
gang's bound members together). For an evicted attempt $e_p$ is the eviction
time; it assembled only if every member was bound and had started before it,
otherwise every bound member was stranded from its bind to the eviction. Until
this revision only the final attempt was counted, which understated
preemption: in the reference model (default kube harness, seed 0) D-preempt
read 65.2 GPU-h where counting its evicted gang attempts gives 71.3 — above
D-fifo's 71.2, so the ordering of the two in the README table flipped. A pod
the control plane destroyed is its own final attempt and is counted once.

Reported:

- `gang_stranded_gpu_hours` $= \sum_j \text{stranded}_j / 3600$ over every
  attempt — **gated** for stability.
- `gang_stranded_share` — that over GPU-hours used. With the default zero
  startup delay $b_p = s_p$, so the final attempts' stranded time is part of
  the delivered time: GPU-hours the lab counts as delivered and a real gang
  could not have used. (An evicted attempt's stranded time is not delivered;
  it is Running time lost to the eviction, or bind-to-Running time.)
- `gang_assembly_p50`, `gang_assembly_p95` — nearest-rank percentiles of
  $A_j - \min_p b_p$ (seconds) over gangs whose final attempt assembled;
  `null` if none assembled.
- `gang_assembled` — gangs whose every member started (final attempt).

**Bind versus Running.** With the default zero startup delay $b_p = s_p$.
With `--startup-delay` the reference model draws the gap from a seeded RNG and
the runner records $s_p$ on the first poll that shows the pod Running, so a
gang's bind-to-Running time counts as stranded (tested:
`test_a_gangs_startup_gap_is_stranded_time`) and every pod's appears in
`startup_overhead_gpu_hours`.

### Reconciliation with `gang_wasted_gpu_hours`

Phase 1 reported `gang_wasted_gpu_hours`. It measured the same quantity — GPU
time held idle while a gang assembles — with a cruder rule, so it is
**replaced**, not kept beside the new one: one quantity, one name. The old rule
was $\sum_{t \in T} g \cdot (\max T - t)$ for a gang whose bind spread exceeded
60 s, $g \cdot |T| \cdot (H - \min T)$ for one never fully bound, and 0
otherwise. The two agree exactly for a gang whose pods bind and start together
with a spread above 60 s and no member ending before the last starts (tested).
They differ in three places, each deliberate:

1. No threshold: a 30 s assembly costs 30 s of GPU time, not zero.
2. A never-assembled gang is charged what each member actually held, from its
   own bind to its own end. The old rule charged every bound member from the
   *first* bind to the *horizon*, even after it had ended.
3. Held time runs from bind and assembly is judged on Running.

Results written before this change — including `results/results.json` (Run 2 in
the README) — carry the old name and the old definition, and are not comparable
with `gang_stranded_gpu_hours`.

### Deprecated: gang deadlock rate

`gang_stalled`, `gang_deadlocked` and

$$\text{gang deadlock rate} = \frac{|\text{deadlocked}| + |\text{stalled}|}{|\text{gang jobs}|}$$

(deadlocked: some pods bound, at least one never; stalled: all bound, with bind
spread above `deadlock_threshold`, default 60 s) are **deprecated**. They are
still computed, unchanged and bit-identical, and still written to every run in
results.json for backwards compatibility; they are no longer in the headline
table, and no longer gate stability. Two reasons. It is a binary per gang, so
a 61-second and a 6-hour assembly count the same. And its threshold equals one
runner poll at the default speedup (`TICK` 1 s real × `--speedup` 60 = 60
simulated seconds), so whether a gang bound on two consecutive polls counts as
stalled depends on sub-second API latency rather than on the scheduler.

## Topology and placement

The lab has no network. It can still measure *where* a scheduler put a job's
pods, which is a property of its decisions — and that is all the placement
metrics claim.

**Declaring topology.** In a fleet file, per node class: `nvlink` (bool,
default false) and `nodesPerRack` (default: the whole class in one rack); at
fleet level, `topology: {racksPerSwitch: N}` (default: every rack under one
switch). Unknown keys are rejected — a misspelling must not silently fall back
to a default layout. `k8slab.topology.derive` assigns racks and switches
deterministically: node classes in file order, each class filling its own racks
`nodesPerRack` at a time (racks never mix classes; the last may be short),
racks numbered across the fleet, consecutive racks grouped `racksPerSwitch` at
a time into switches (a switch may mix classes). Racks nest in switches by
construction. A fleet that declares nothing gets the flattest layout — one rack
per class, one switch — and every result records `topology_declared: false`.

The shipped fleets: default is `dgx8` (NVLink) in rack-0..2, `mid4` in
rack-3..4, `edge2` in rack-5, two racks per switch, so switch-0 = rack-0,1,
switch-1 = rack-2,3, switch-2 = rack-4,5. Homogeneous is 17 `dgx8` in racks of
four (rack-4 holds one node), two racks per switch. Neither change alters any
node's shape. **The layouts, NVLink flags and everything derived from them are
a declared scenario, not hardware.**

**Node labels.** `k8slab nodes` emits `topology.k8slab.io/switch`,
`topology.k8slab.io/rack` and `topology.k8slab.io/nvlink` (`"true"`/`"false"`)
on every kwok node. The prefix is the lab's own: these are declared values on
fake nodes, `kubernetes.io/` and `k8s.io/` are reserved for Kubernetes core
components ([labels](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set)),
and a vendor prefix would imply the values were discovered. The only
`nvidia.com` / `kubernetes.io` labels remain the two Phase 1 already emitted.
Label-defined levels are exactly the shape Kueue Topology-Aware Scheduling
consumes: a `Topology` object (`kueue.x-k8s.io/v1beta2` at the lab's planned
Kueue pin, v0.19.2) lists `spec.levels[].nodeLabel` "from the widest (block) to
the narrowest (hostname)", and TAS computes free capacity per domain from them
([concept](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/),
[setup](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_topology_aware_scheduling/),
[v0.19.2 example](https://raw.githubusercontent.com/kubernetes-sigs/kueue/v0.19.2/site/static/examples/tas/sample-queues.yaml),
read 2026-09-26). So K1 can point a `Topology` at
`[topology.k8slab.io/switch, topology.k8slab.io/rack, kubernetes.io/hostname]`
without the fleet changing. That configuration has not been built or run.

**Placement tier** of a job whose every pod is bound: the widest domain its
pods span — `node` (one node), `rack` (several nodes, one rack), `switch`
(several racks, one switch) or `cross-switch`.

**Penalty factors — ASSUMED.** A factor per tier, as a scenario parameter
(`k8slab.topology.PenaltyFactors`, configurable; validated to be positive and
non-decreasing as placement widens):

| placement | default factor |
|---|---|
| one node with NVLink, or any job with ≤ 1 GPU in total | 1.0 |
| one node without NVLink, more than one GPU | 1.2 |
| several nodes, one rack | 1.4 |
| several racks, one switch | 1.8 |
| several switches | 2.2 |

**None of these numbers is a measurement.** The lab sends no NCCL traffic and
has no fabric, so it cannot tell whether a cross-switch job would run 2.2×
slower, or slower at all. The ordering is the only part with physical grounding
(more switch hops are never cheaper); the magnitudes are illustrative. 1.2 for
a single non-NVLink node sits between NVLink and intra-rack on the reasoning
that GPU-to-GPU traffic inside one host over PCIe avoids the NIC and switch but
lacks NVLink bandwidth; that placement relative to 1.4 is itself an assumption.
`k8slab.topology.placement_factor(placement, topology, factors)` is a pure
function the execution layer calls to stretch a job's duration under this
assumption — only with `--topology-penalty extend`, which produces SCENARIO
rows (`src` gains `+topo`) that are never pooled with or ranked against
measured rows. In the default `report` mode the factors affect only
`placement_penalty_mean`; in `off` mode not even that (it is `null`).

**Placement metrics — measured, always reported.**

- `placement_tier_share` — for fully placed multi-pod jobs, the share at each
  tier; `placement_multi_pod_jobs` is the denominator. With no such job every
  share is `null` (0/0 is undefined; it used to be written as 0.0, which read
  as "never placed at that tier" and was averaged across repeats as a zero).
- `placement_penalty_mean` — GPU-weighted mean factor over fully placed jobs
  with $G_j > 1$: $\sum_j G_j f_j / \sum_j G_j$. Single-GPU jobs cannot
  communicate and would only drag every configuration toward 1.0 by the trace's
  share of single-GPU work. It inherits the assumption; `penalty_factors` in
  every run records which factors were used, and runs scored under different
  factors are never aggregated together.

Context for reading them: the degenerate policies walk nodes in lexicographic
name order (`sorted(free)` in `baselines.py`: dgx8-0, dgx8-1, dgx8-10, …), which
is not rack order, so even D-fifo spreads most gangs across switches (74.1%
of its fully placed multi-pod jobs cross-switch in the reference model on the
default fleet with `--queue-model none`, 69.8% under the default kube queue
model). That is a property
of the baselines, and exactly what a topology-aware configuration should beat.

## Fairness

Per account $a$, with GPU-seconds demanded $D_a$ and delivered $V_a$:

$$r_a = \frac{V_a}{D_a}, \qquad \text{fairness} = \frac{\max_a r_a}{\min_a r_a}$$

1.00 is perfect: every account got the same fraction of what it asked for.
Accounts that received nothing are excluded from the ratio rather than making it
infinite; a run where an account is fully starved shows up as a missing row in
the per-account breakdown, which is the more legible signal.

$V_a$ is the **trace's work** delivered to account $a$: Running GPU-time of
its pods' final attempts, plus what checkpoints kept of evicted ones, minus the
account's share of `topology_extension_gpu_hours`. The subtraction matters only
under `--topology-penalty extend` (the extension is 0 otherwise), where a
stretched pod's Running time includes the ASSUMED stretch; both execution
paths book the stretch per account (`Observation.extension_by_account`), and
scoring refuses an observation whose split does not add up to the total. So
$r_a$ never exceeds 1 beyond float rounding (the default `extend` run's
fairness ratios come out within $3 \times 10^{-15}$ of 1), and an `extend` row
in which every job completed reads 1.00. Until this revision the stretch counted as service: in the reference
model's default `extend` run (seed 0, harness seed 0, kube queue, one run per
policy) every job completed, yet the service ratios read 1.13–1.43 and
fairness 1.087 (D-fifo), 1.087 (D-random), 1.081 (D-largest) and 1.129
(D-preempt) — which accounts had more multi-GPU, cross-switch work, i.e. the
premise, not the service each account received
(`test_fairness_subtracts_each_accounts_topology_extension`, and the
fairness assertions in the `extend` tests of both paths).

## Preemption

A pod whose API object disappears (or turns Terminating) without this runner
deleting it was destroyed by the control plane. In the first cluster run **66 of
476 pods** were preempted by kube-scheduler's priority preemption and, being
bare pods with no owning controller, were never recreated.

The PriorityClasses therefore keep `preemptionPolicy: Never`, so for every
non-preemptive configuration priority affects queue order and nothing else.
Preemption is measured deliberately instead, by its own configuration:
**D-preempt** (a degenerate baseline; rule in `src/k8slab/preemption.py`), whose
evictions are scored by the metrics in the next section. A control-plane
deletion is scored the same way, with `requeued = false`.

## Execution layer: startup overhead, evictions, topology extension

Each `PodEvent` describes a pod's **final** attempt; every
earlier attempt that was evicted is an `Eviction` record (bind, start, evict
and release times, GPUs, reason, lost and retained seconds). With $g$ the
attempt's GPUs:

- `startup_overhead_gpu_hours` $= \sum g \cdot (s - b)$ over final attempts
  (to the end or the horizon for one never Running) and requeued evicted
  attempts (to Running, or to the eviction) — GPUs allocated, doing nothing.
  0 with the default zero delay.
- `preemptions` — evicted attempts; every gang member counts.
- `preempted_gpu_hours_lost` $= \sum g \cdot (1-c)(e - s)$ over evicted
  attempts that had started, with $c$ the `--checkpoint-fraction` (default 0:
  restart from zero). The kept part, $g \cdot c(e-s)$, counts as delivered.
  An attempt whose work had already ended is never evicted, so $e - s$ never
  exceeds its duration: the model applies requested evictions after a pass
  releases finished pods (D-preempt's victims were always chosen after it), and
  on a cluster D-preempt does not offer a pod whose work ended since the last
  poll — still bound, because the runner deletes it only after the bind pass —
  as a victim, and a pod first seen Terminating after its work ended is
  recorded as completed at its end, as a finished pod that vanished already
  was. Until this revision both paths could evict finished work: in a one-node
  fake-cluster replay a 90 s job was evicted 120 s into its run and run again
  (`test_d_preempt_never_evicts_an_attempt_whose_work_ended_since_the_last_poll`,
  `test_a_requested_eviction_never_evicts_work_that_already_finished`).
- `grace_locked_gpu_hours` $= \sum g \cdot (\min(r, H) - e)$: GPUs an evicted
  pod kept locked from eviction $e$ to release $r = e +$ `--grace-seconds`
  (default 30 simulated seconds). Locked GPUs are not free in the free-GPU
  samples, so every fragmentation definition sees them as allocated. On a
  cluster the runner releases a lock on the first poll at or after $r$, before
  that poll's sample and bind pass, as the reference model does on its first
  pass at or after $r$. It used to release them after the bind pass, which
  withheld the GPUs a whole extra poll: in a one-node fake-cluster replay at
  speedup 60 with a 30 s grace period, the samples showed the node held for
  120 s after the eviction and the preemptor bound at 420 s; with the fix,
  60 s (one poll) and 360 s — while this metric reads 30 s either way
  (`test_an_expired_grace_lock_is_free_on_the_first_pass_after_it`).
- `topology_extension_gpu_hours` — delivered GPU-time that exists only because
  runtimes were stretched (`extend` mode; 0 otherwise). A pod the control plane
  destroyed contributes nothing: none of its Running time is delivered (it is
  all in `preempted_gpu_hours_lost`), so none of it can be extension. It is
  also booked against each stretched job's account, so
  [fairness](#fairness) can take it back out of that account's service.
- `gpu_hours_demanded` $= \sum_j G_j d_j / 3600$, the trace's work, so the
  invariants above can be checked from any results.json.

On a cluster the runner sees transitions only when it polls, so bind, Running
and deletion times are quantised to `speedup` simulated seconds
([limitations](limitations.md#startup-latency-is-modelled-not-measured)).

---

## Repeated runs: mean ± sd and the stability verdict

One run of this harness is one sample from a spread wider than most differences
between configurations (docs/limitations.md). So `report.py` groups every run
by configuration and reports each numeric column as **mean±sd** (sample sd,
$n-1$) with an **n** column; results.json keeps every raw run next to the
aggregate. Implemented in `src/k8slab/stats.py`.

**Stability rule.** For the gated metrics below, a configuration is *unstable*
on a metric when

$$\text{sd} > \text{floor} \quad\text{and}\quad \text{CV} = \frac{\text{sd}}{|\text{mean}|} > 0.05 .$$

If any configuration is unstable on any gated metric, results.md opens with an
**UNSTABLE** banner.

**Why a coefficient of variation and not "σ > 0.05".** The plan named an
absolute σ > 0.05. One absolute threshold cannot serve quantities in different
units: on utilization (a fraction) 0.05 is a five-percentage-point band, on
makespan (hours) it is three minutes, and on wait (seconds) it is a twentieth of
a second. It would be simultaneously far too lax, too strict and meaningless.
CV is dimensionless, so 5% means the same thing for every metric.

**Why a floor.** CV explodes near a zero mean. Definition A reads about
0.003% for D-largest in the reference model under the Phase 1 binder
(`--queue-model none`); a correlation can sit near 0. A spread of 0.00002 on a
mean of 0.00003 is a CV of 67% and is invisible in every table. Each gated
metric therefore has an absolute floor in its own unit, set to **half the last
digit results.md displays** — a spread below it cannot be seen, and a test
keeps the floors in step with the display precision:

| gated metric | floor | displayed as |
|---|---|---|
| `makespan_hours` | 0.05 h | 0.1 h |
| `utilization` | 0.0005 | 0.1 % |
| `mean_wait`, `p95_wait` | 30 s | whole minutes |
| `fragmentation_rate` (A), `fragmentation_ref` (B), `fragmentation_structural` (C-node) | 0.0005 | 0.1 % |
| `gang_stranded_gpu_hours` | 0.05 GPU-h | 0.1 GPU-h |
| `footprint_wait_spearman` | 0.005 | 0.01 |

These are the quantities configurations are ranked on. `gang_deadlock_rate`
is no longer gated (deprecated above).

**Verdicts,** per configuration: `unreplicated` (n = 1 — the sample sd is
undefined, and reporting 0 would assert perfect reproducibility from the one
piece of evidence that cannot show it; **n = 1 is never stable** and always
raises the banner), `unstable`, `deterministic` (n ≥ 2 bit-identical runs, e.g.
the reference model repeated at a zero startup delay — it passes the stability
rule trivially, sd = 0, and carries no variance information about the cluster
harness), or `stable`.

A metric undefined in some runs (`null`, e.g. no gang assembled) is summarised
over the runs where it is defined, with its own n. Aggregation refuses to mix
cluster and reference-model runs, fleets, or penalty-factor sets — and, since
Phase 2, runs produced under different harness settings (queue model, speedup,
startup delay, topology mode, grace period, checkpoint fraction, gates; every
results.json records them per run under `harness`). The per-repeat harness seed
may differ: that is what a repeat is. SCENARIO rows (`extend`, `src` `+topo`)
are never pooled with measured ones, and GATED rows (`+gated`) are shown but
left out of every comparison.

**Repeats in practice.** `make bench` / `make bench-model` run `REPEAT`
(default 5) repeats, interleaved across configurations. The trace seed
(`--seed`) is fixed; repeat $i$ uses harness seed $i = 0$: `--seed` itself (so
a one-repeat reference-model run with `--queue-model none` and the default
zero startup delay reproduces Phase 1; under the default kube queue model it
does not), $i \ge 1$: 32 bits of `sha256("k8slab-harness|<seed>|<i>")`. That
seed drives D-random's choices and the startup-delay draws. So in the reference
model at the default **zero** startup delay, D-fifo, D-largest and D-preempt
repeat bit-identically (`deterministic`) and D-random does not. **With a
non-zero delay every policy's repeats vary**: at `--startup-delay 50:200`
(speedup 60, kube queue, default trace, three repeats into
`results-model/startup-delay`) D-fifo's mean wait was 7114, 7079 and 7388 s and
its startup overhead 5.52, 5.42 and 5.52 GPU-h, and D-fifo, D-largest and
D-preempt were all `unstable`.
`--trace-seeds a,b,c` adds a separate study, one run per configuration per
trace, reported in its own section and never pooled with the repeats. Its
spread across traces **mixes** workload variance (the traces differ), harness
variance (on a cluster every run carries the run-to-run noise
[limitations](limitations.md#run-to-run-variance-is-large-and-it-is-not-yet-controlled-on-a-cluster)
describes) and, for D-random, the policy's own random choices, which differ on
every trace; with one run per trace nothing separates them. So the section
gives no stability verdict: its table has a `traces` column instead, and each
configuration's note says the sd is across traces. Its comparisons are
**paired by trace** — every configuration replayed the same traces, so the
statistic is the mean per-trace difference, and the bootstrap resamples those
differences, so a trace that is hard for everyone cancels instead of widening
the interval. Resolvable only if that interval excludes 0 over at least five
traces; the harness-stability rule is not applied, since whatever harness
noise each run carries is already inside the per-trace differences. (This
section and results.md used to call the spread "workload variance, not harness
variance" while printing, under it, each configuration's "UNSTABLE ... not
resolvable by this harness" note, and compared the traces as if they were
independent repeats.) Runs on different traces may also be aggregated in the
main table (a multi-seed study), but the verdict then says the spread includes
workload variance.

## Pairwise differences: the bootstrap

The README's rule is that only differences surviving repetition mean anything.
results.md applies it: each configuration against **every degenerate baseline
present, and against K0 when present** (a pair of baselines once, not twice),
on `makespan_hours`, `utilization`, `mean_wait` and
`fragmentation_structural`.

**Method.** A percentile bootstrap of the difference in means. Each of 4000
resamples draws $n_c$ runs with replacement from configuration $c$ and $n_b$
from baseline $b$, independently — repeats of two configurations are separate
runs, not pairs — and records $\bar{x}_c^* - \bar{x}_b^*$. With the $R$
resampled differences sorted, the 95% interval excludes
$k = \lfloor R \cdot 0.025 \rfloor = 100$ of them from **each** tail and spans
the rest: $[d_{(k)}, d_{(R-1-k)}]$, 0-based, i.e. the 101st smallest to the
101st largest (`stats.tail_count`; a $10^{-9}$ guard keeps $k$ from being
rounded down by float error, and $k$ is capped so the interval never inverts).
This page used to call the bounds the 2.5th and 97.5th nearest-rank quantiles.
They were not — nearest-rank 2.5% of 4000 is the 100th smallest — and at other
confidence levels the two separately rounded indices cut different counts from
the two tails (at 0.90, $1 - 0.9 = 0.09999999999999998$ cut 199 below and 200
above). At the default 0.95 and 4000 resamples the old indices were already
100 from each end, so no interval in any results file moved. The generator is
`random.Random("<seed>|<config>|<baseline>|<metric>")`, seed 0 by default:
string seeds are hashed with SHA-512, so every interval is reproducible,
independent of `PYTHONHASHSEED`, and unmoved by adding other configurations.
Stdlib only.

**Resolvable** only if the interval excludes 0, both sides have **n ≥ 5**,
**neither side is `unstable` on that metric**, and the two are comparable at
all (same source, trace and fleet — otherwise the row says why it was
refused). The unstable rule is the stability section's own: an unstable metric
is one this harness cannot resolve, so no comparison on it may be called
resolvable, even when its interval excludes 0; the row shows the interval with
the note "unstable on …". (Until this revision results.md said both at once,
e.g. D-random's definition-C differences in the reference model.) Identical
repeats on both sides give a degenerate interval $[d, d]$; that is labelled
`deterministic`.

**Its weakness at small n, measured.** The percentile bootstrap is
anti-conservative with few runs: the resampled means have almost no support
(two runs give three distinct means), their variance is biased low by a
factor $(n-1)/n$, and nothing accounts for the heavier tails of a small-sample mean.
Under a null of *no difference* — both sides drawn from one normal distribution
— the 95% interval excluded 0 in:

| runs per side | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| false "resolvable" rate | 33.3% | 20.2% | 13.0% | 9.0% |

against a nominal 5% (2000 null trials per n, 4000 resamples;
`scripts/bootstrap_null.py`, below).
Hence the n ≥ 5 minimum, and hence the wording: at the plan's five repeats,
**"resolvable" is a screen with roughly a one-in-eight false-alarm rate, not a
5% test**, and real harness noise is not guaranteed to be as benign as a normal.
One more assumption is doing work: repeats are treated as independent draws. If
every K0 repeat runs before every D-fifo repeat on one machine, slow drift in
that machine becomes a "difference"; interleaving repeats is the execution
layer's defence.

## Computation cost

Every metric pass is linear in pods, jobs or samples, except definition A's
pending-queue query (a sort, $O((P+S)\log P)$) and definition C's per-sample sum
over nodes and levels (memoised on identical consecutive states). Measured on
the development machine (macOS, Python 3.12.13) for one default-profile
reference-model run: `metrics.compute` took 0.45–0.54 s before the sweep,
dominated by the $O(S \times P)$ pending scan (≈10,000 samples × 1,201 pods), and
0.022–0.030 s after it, with definition C, gang, size-bias and placement metrics
added. With this revision's full sample validation in definition C and the
eviction-aware passes it measured 0.029–0.034 s (D-fifo, D-random, D-largest,
seven timings each).

## Where the numbers on this page came from

Every number above that is not a formula or a quoted Phase 1 result was
produced in this repository on 2026-09-26, by the in-process **reference
model, not a cluster** — never compare them with a cluster row — with these
commands (default profile, trace seed 0, speedup 60, five interleaved repeats
unless stated; results directories are gitignored, so re-run to check):

- **Default kube queue model** (the README's reference-model table; D-largest
  23.1% under A, the kube-harness starvation ratios, C levels and placement
  shares): `make bench-model`, i.e.
  `.venv/bin/k8slab bench --reference-model --fleet fleets/default.yaml
  --profile default --speedup 60 --repeat 5 --queue-model kube
  --topology-penalty report --startup-delay 0:0 --results results-model`.
- **Phase 1 binder** (D-largest 0.003% under A and 49.4% under B, D-fifo's
  C-node 14.5%, the `--queue-model none` starvation ratios, 74.1%
  cross-switch): the same with `--queue-model none --results
  results-model/queue-none`. Its D-fifo and D-largest values equal the
  earlier n = 1 figures on this page; D-random's are now means over five
  repeats (seed 0 alone gave starvation 17.02 and B 66.2%, the figures this
  page used to quote).
- **Homogeneous control:** the same two commands with `--fleet
  fleets/homogeneous.yaml --results results-model/homogeneous` (queue model
  none) and `--results results-model/homogeneous-kube` (kube).
- **Startup-delay repeats:** `.venv/bin/k8slab bench --reference-model
  --profile default --speedup 60 --repeat 3 --topology-penalty report
  --startup-delay 50:200 --fleet fleets/default.yaml --queue-model kube
  --config D-fifo --config D-largest --config D-preempt --results
  results-model/startup-delay`.
- **Bootstrap false-alarm rates:** `scripts/bootstrap_null.py` (run from the
  repository root with the venv's Python; a few minutes). ONE generator,
  `random.Random(12345)`, is created once and consumed for n = 2, 3, 5, 10 in
  that order; each of 2000 trials per n draws
  `a = [gen.gauss(0, 1) for _ in range(n)]`, then `b` the same way, and is
  scored by `k8slab.stats.bootstrap_diff_interval(a, b,
  rng=random.Random(trial))` at its default 4000 resamples; the rate is the
  share of intervals excluding 0. It printed 33.25%, 20.20%, 13.00% and 9.05%
  (the table above, rounded), and printed the same again when re-run after the
  tail-count change (same indices at the default). Re-creating the generator for each n is a
  different experiment and gives different numbers; this page used to leave
  that ambiguous. `tests/test_stats.py::test_documented_small_n_weakness_holds`
  re-runs a smaller replica.
- **Timings:** `time.perf_counter()` around `metrics.compute` on
  `sim.run(fleet, generate(profile_named("default"), seed=0), policy)`.
- **Pre-fix values quoted as corrections** (D-preempt's Definition A 4.08%,
  gang stranding 65.2 GPU-h): the same default-kube run scored with the
  eviction records withheld from the metric in question.
- **Wait under eviction** (D-preempt's mean wait 205.2 → 201.8 min, ρ 0.27 →
  0.28, starvation 1.56 → 1.59, and 1.62 → 1.66 with `--queue-model none`):
  the default-kube and Phase 1 binder commands above, run before and after the
  change; nothing else in those runs moved. D-preempt's repeats are
  bit-identical, so one run reproduces each figure (`tests/test_golden_model.py`
  pins the README's deterministic rows from one run).
- **The starvation-flag correction** (D-largest 7.20 against D-random 19.14;
  on the homogeneous fleet 6.86 against 15.51): the flags and ratios in
  `results-model/results.md` and `results-model/homogeneous-kube/results.md`.
- **Topology extension** (extension GPU-h and makespans under the current
  rule and the earlier one, for gangs that never co-ran): one run per policy,
  harness seed 0, `.venv/bin/k8slab bench --reference-model --fleet
  fleets/default.yaml --profile default --speedup 60 --repeat 1 --queue-model
  kube --topology-penalty extend --startup-delay 0:0` (*Execution layer* table
  of its results.md) gives the current values. `scripts/extend_prefix_rule.py`
  replays the same inputs under both rules and prints both sets, with the
  counts of gangs with a factor above 1 that were never stretched (89, 34, 25,
  24 for D-random, D-fifo, D-largest, D-preempt under the old rule; 0 under the
  current one): fully placed gang jobs whose placement factor exceeds 1 and
  whose every member ran no longer than its trace duration plus 1e-6 s. The
  tolerance matters — float rounding leaves `end − start` a hair under the
  duration for many members, and an exact comparison counts far fewer (41 for
  D-random). Re-run on 2026-09-26: both rules reproduced every figure above.
- **Fairness under `extend`** (service ratios 1.13–1.43 and fairness
  1.081–1.129 counting the stretch; 1.00 without it): the same four runs
  scored by `metrics.compute` as they are, and with `extension_gpu_seconds`
  and `extension_by_account` withheld from the observation — which is the
  pre-fix computation, since it never read either.

[ssl]: https://github.com/Zhanyl-tech/slurm-scheduler-lab
[nv-volcano]: https://developer.nvidia.com/blog/practical-tips-for-preventing-gpu-fragmentation-for-volcano-scheduler/
