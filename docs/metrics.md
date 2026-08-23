# Metric definitions

Every formula the lab reports, written out. Quote a number from this repo only
together with the definition it was computed under.

Notation. The fleet has nodes $n$ with GPU capacity $C_n$; $\text{free}_n(t)$ is
the GPUs free on node $n$ at time $t$. $H$ is the horizon — the simulated time
at which the run stopped. Integrals are evaluated as trapezoids over the sample
series the runner records once per poll.

---

## GPU-hours used, GPU-hours idle, utilization

$$\text{GPU-hours used} = \frac{1}{3600}\sum_{p \in \text{pods}} g_p \cdot (\text{end}_p - \text{start}_p)$$

where $g_p$ is the GPUs pod $p$ requested. Capacity is $\left(\sum_n C_n\right)
\cdot H$, idle is capacity minus used, and

$$\text{utilization} = \frac{\text{GPU-hours used}}{\text{capacity}}$$

**A pod's runtime is the duration the trace assigned it**, not the wall-clock
between this runner noticing the binding and noticing the deletion. Using the
observed interval adds up to one poll period of phantom GPU time per pod —
measured at **+11.9%** of total GPU-hours at `--speedup 400`, which is larger
than any effect this lab tries to detect. The check that catches this: a run in
which every job completes must report exactly the GPU-hours the trace demands.
It does.

Utilization is still a real measurement of scheduling quality, because the
horizon is not fixed — a scheduler that packs badly stretches $H$ and dilutes
its own utilization.

## Wait time

$$\text{wait}_j = \max_{p \in \text{pods}(j)} \text{scheduled}_p - \text{submit}_j$$

**To the last pod, not the first.** A gang that is half-placed has not started.
Crediting a job at its first binding would flatter exactly those schedulers that
admit gangs partially — the failure mode this lab exists to measure.

Reported as mean and p95, nearest-rank.

Wait is quantized to the poll interval, which is `speedup × TICK` simulated
seconds — 200 s at `--speedup 200`. Differences between configurations smaller
than that are not measurements.

## Fragmentation — two definitions, and they disagree

There is no canonical definition in the literature, and the vendor claims that
motivated this repo do not state theirs. So the lab reports two.

### Definition A — queue-relative

A free GPU is *stranded* if it sits on a node that cannot fit the smallest
pod currently pending:

$$m(t) = \min\{\, g_p : p \text{ pending at } t \,\}, \qquad
S_A(t) = \sum_n \text{free}_n(t) \cdot \mathbb{1}\!\left[0 < \text{free}_n(t) < m(t)\right]$$

$$\text{frag}_A = \frac{\int_0^H S_A(t)\,dt}{\int_0^H \sum_n \text{free}_n(t)\,dt}$$

If nothing is pending, $S_A(t) = 0$: capacity nobody wants is idle, not
fragmented. Without that rule an empty cluster reports maximal fragmentation.

### Definition B — queue-independent

Same shape, but against a fixed reference request $R$ — the largest per-pod ask
anywhere in the trace — instead of the live queue:

$$S_B(t) = \sum_n \text{free}_n(t) \cdot \mathbb{1}\!\left[0 < \text{free}_n(t) < R\right]$$

$$\text{frag}_B = \frac{\int_0^H S_B(t)\,dt}{\int_0^H \sum_n \text{free}_n(t)\,dt}$$

### Why both

**Definition A reports 0.0% for `D-largest`** — 0.003% before rounding, not
identically zero only because of brief windows at the start and tail of a run
with nothing small queued. Largest-request-first starves small jobs, so a 1-GPU
pod is essentially always pending, so $m(t) = 1$, so no free GPU is ever below
the threshold and nothing is ever stranded. The metric says the worst-behaved
policy in the set has flawless packing.

Under definition B the same run reads **49.4%** — four orders of magnitude
apart, computed from identical data.

Neither is wrong. A measures "could the queue have used this right now", B
measures "is this fleet carved into unusable shapes". They answer different
questions and a policy can look excellent under one and mediocre under the
other. **This is the reason a fragmentation claim without a stated definition —
including the 34% figure this repo exists to test — cannot be reproduced or
disputed.**

The denominator matters too. Against free GPU-time the number describes the
quality of the free pool; against total fleet GPU-time it describes how much of
the machine is lost. The lab reports A both ways; they differ by roughly the
utilization factor.

### The control

`fleets/homogeneous.yaml` has one node shape. Fragmentation there is
structurally near-impossible. If either definition reports a large number on
that fleet, the implementation is wrong — that is what the fleet is for.

## Gang behaviour

For each job with `gang_size > 1`, let $T$ be the set of times at which its pods
were bound.

- **Deadlocked** — some pods bound, at least one never bound. Those GPUs are
  held for nothing until the horizon.
- **Stalled** — all pods bound, but $\max T - \min T$ exceeds
  `deadlock_threshold` (default 60 s). Early pods hold GPUs idle while the gang
  assembles.

$$\text{gang deadlock rate} = \frac{|\text{deadlocked}| + |\text{stalled}|}{|\text{gang jobs}|}$$

Wasted GPU-hours are the held-but-idle GPU-time: $\sum_{t \in T} g \cdot (\max T
- t)$ for a stalled gang, and $g \cdot |T| \cdot (H - \min T)$ for a deadlocked
one.

This is the metric that should separate a gang-aware scheduler from one without
gang support. The default scheduler has no gang concept at all, which is the
point of measuring K0 first.

## Fairness

Per account $a$, with GPU-seconds demanded $D_a$ and delivered $V_a$:

$$r_a = \frac{V_a}{D_a}, \qquad \text{fairness} = \frac{\max_a r_a}{\min_a r_a}$$

1.00 is perfect: every account got the same fraction of what it asked for.
Accounts that received nothing are excluded from the ratio rather than making it
infinite; a run where an account is fully starved shows up as a missing row in
the per-account breakdown, which is the more legible signal.

## Preemption

Recorded, not scored. A pod whose API object disappears without this runner
deleting it was destroyed by the control plane. In the first cluster run **66 of
476 pods** were preempted by kube-scheduler's priority preemption and, being
bare pods with no owning controller, were never recreated.

Phase 1 therefore creates its PriorityClasses with `preemptionPolicy: Never`, so
priority affects queue order and nothing else. Preemption is a scheduling
behaviour worth measuring deliberately; it is not something that should be
silently deleting a third of the workload in every other configuration's run.
