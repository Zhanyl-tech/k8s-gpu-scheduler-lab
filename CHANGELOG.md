# Changelog

What changed in how numbers are produced and reported. A result is only
comparable with another produced by the same harness; results.json records
which one (`schema`, `harness`, `timescale`).

## Unreleased — Phase 2: measurement and execution layers

**No cluster run has been made with anything in this section.** Everything
below is tested against the reference model and a fake API server; the Run 2
cluster table in the README is Phase 1's and is unchanged.

### Execution layer

- **Repeated runs.** `k8slab bench --repeat N` (Makefile `REPEAT ?= 5`) runs
  every configuration N times, interleaved across configurations, with the
  trace seed fixed and a per-repeat harness seed derived from `--seed` (repeat
  1 uses `--seed` itself). Results are rewritten, atomically, after every run.
  `--trace-seeds a,b,c` adds a separate across-trace-seed study.
  `--sigma-tolerance` sets the stability tolerance (a coefficient of
  variation); `--fail-on-unstable` exits 3 when any configuration is unstable.
- **Queue parity** (`src/k8slab/queueing.py`, `src/k8slab/binder.py`).
  `--queue-model kube` (new default for results) puts the degenerate binder
  through kube-scheduler's queue mechanics, verified against the v1.32.2
  source, with simulated backoff derived from the speedup; `--queue-model
  none` reproduces Phase 1 bit for bit (`tests/test_sim_equivalence.py`).
  `--cycle-latency` (default 0) exists; no latency penalty is applied by
  default.
- **Time scaling on the cluster.** `make up` generates a speedup-scaled
  `KubeSchedulerConfiguration` (`k8slab scheduler-config`), mounts it through
  kind `extraMounts` and kubeadm `extraVolumes`, disables leader election and
  scales `--pod-max-in-unschedulable-pods-duration`. `make bench` refuses to
  run at a different `SPEEDUP` or `STARTUP_DELAY` than the cluster was built
  with. Every results.json records the scaling and its residual.
- **Startup delay.** `--startup-delay MIN:MAX` (real ms; kwok `pod-ready` Stage
  delay on a cluster, seeded draw in the model). Runtime counts from Running;
  new metric `startup_overhead_gpu_hours`.
- **Preemption.** New degenerate baseline **D-preempt**; evictions lock GPUs
  for `--grace-seconds` (default 30) and requeue the victims
  (`--checkpoint-fraction`, default 0). New metrics `preemptions`,
  `preempted_gpu_hours_lost`, `grace_locked_gpu_hours`. On a cluster, pods the
  control plane deletes keep their GPUs held for the grace period in the
  free-GPU samples.
- **Topology extension.** `--topology-penalty {off,report,extend}` (default
  report). `extend` stretches runtimes by the ASSUMED placement factor; such
  rows are labelled `+topo` and never pooled with or ranked against measured
  rows. New metric `topology_extension_gpu_hours`; the invariant becomes
  delivered = demanded + extension.
- **Diagnostics.** `--serialise` / `SERIALISE=1` (kube-scheduler
  `parallelism: 1`) and `--admission-gate`; rows labelled `+gated`, excluded
  from comparisons.
- **Runner.** Injectable API, clock and sleep, tested against a fake cluster;
  reads the topology back from Node labels and refuses a cluster that does not
  carry the fleet; deletes leftover lab pods before and after each run (a
  stalled K0 run used to leave Pending pods that kube-scheduler would bind into
  the next configuration's run).

### Measurement layer (from the Phase 2 measurement work)

- Fragmentation **Definition C** (structural, per topology level), shared with
  slurm-scheduler-lab through a golden vector; node level is the headline.
- `gang_stranded_gpu_hours` replaces `gang_wasted_gpu_hours`;
  `gang_deadlock_rate` is deprecated (still written).
- Size-bias metrics: wait by footprint, `footprint_wait_spearman`,
  `large_job_starvation_ratio`, and the starvation flag.
- Declared rack/switch/NVLink topology, placement tiers, ASSUMED penalty
  factors.
- Aggregation into `mean±sd` with stability verdicts; seeded bootstrap
  comparisons.

### Results format

- results.json **schema 3** (was 2; Phase 1 wrote a bare list): adds top-level
  `harness`, `timescale` and optional `trace_seed_study`; each config gains
  `scenario`, `gated`, `harness`; each run gains `harness`, `harness_seed`,
  `gpu_hours_demanded`, the execution-layer metrics and `notes`.
- `src` labels gain `+topo` and `+gated` suffixes.
- results.md gains *How these runs were produced*, *Execution layer* and, when
  requested, *Across trace seeds* sections.
- `aggregate()` refuses to pool runs produced under different harness settings.
- Still schema 3, with keys added by the second review: top-level
  `flag_exemptions`, a `paired` field on every comparison, and `design` and
  `bootstrap` in `trace_seed_study`, whose configs' `note` is now the study's
  own note rather than a stability verdict.

### Fixes

- `timescale.py` emitted `podInitialBackoffSeconds: 0` at any speedup above 1.
  kube-scheduler's validation rejects values `<= 0`, so the scheduler would
  not have started. The floor is now 1 s and the residual is reported.
- The generated config now carries `clientConnection.kubeconfig`: with
  `--config`, kube-scheduler ignores kubeadm's `--kubeconfig` flag.
- `make bench` writes to `results/phase2/` instead of `results/`, which holds
  the committed Phase 1 cluster run the README cites. So does `k8slab bench`
  without `--results` now (`results-model/` with `--reference-model`); the CLI
  used to default to `results/` and overwrite the Phase 1 file whenever the
  Makefile was bypassed. It also refuses to write into a directory holding
  results of another schema or source unless given `--force`. `make clean`
  deletes neither `results/` nor `results/phase2/` (only `results-model/`).

### Corrections from the first review

Each was found in review and reproduced before it was fixed;
every changed number below was re-produced with the commands in
docs/metrics.md, *Where the numbers on this page came from* (reference model,
not a cluster).

- **Runner grace locks** were released after the bind pass, so an expired lock
  still blocked the binder and the free-GPU samples for one more poll. They
  are now released before the sample and the bind pass, as in the reference
  model (`test_an_expired_grace_lock_is_free_on_the_first_pass_after_it`).
- **`gang_stranded_gpu_hours`** now counts evicted-and-requeued gang attempts,
  not only the final one. D-preempt's reference-model value moves from 65.2 to
  71.3 GPU-h, just above D-fifo's 71.2.
- **Definition A** no longer counts an evicted pod as pending while an earlier
  attempt was bound; D-preempt's reference-model value moves from 4.1% to 4.0%.
- **Percentiles.** `p95_wait` keeps Phase 1's rounded rank (bit-identical), and
  is no longer called nearest-rank; `gang_assembly_p50`/`p95` and
  `wait_by_footprint` p95 are now true nearest-rank (`metrics.nearest_rank`).
- **Topology extension** no longer includes the stretch of a pod the control
  plane destroyed: none of its time is delivered.
- **`placement_tier_share`** is `null` per tier, not 0.0, when no multi-pod job
  was placed.
- **Definition C** validates every sample, including the final one and those
  followed by zero-length intervals, and rejects unknown nodes.
- **Comparisons** on a metric either side is `unstable` on are never called
  resolvable (results.md used to say both).
- **Time-scaling records** state what was in force: `status` is `recorded`
  (cluster with `make up`'s record; it read `applied` until the third review),
  `unknown` (no record; the scaled values
  are kept only as `intended`), `model` or `not applied` (reference model with
  `--queue-model none`). They also record the flush-tick-gated ranges: at
  speedup 60 a backoff is experienced as [60, 120) simulated s and the
  unschedulable-pool timeout as (300, 2100], not the "300, no residual" the
  docs said.
- **Cluster runs** record `topology_declared` from the fleet file, not `true`
  whenever node labels exist.
- The CLI's terminal table uses `--sigma-tolerance`, like results.md.
- results.md links to docs/metrics.md correctly from any directory in the
  repository (it was broken under `results/phase2/`).
- **Registry.** `SPECS` is the registry of configurations; `POLICIES` holds
  only Phase 1's whole-pass functions (D-preempt has none), with a test that
  the two cannot drift. Dead code removed: `Binder.attempts`,
  `SchedulingQueue.unpop`, `QueueParams.uncompressed`, `Metrics.rows/format`.
- **New tests** pin the runner's Phase 1 binder parity in `--queue-model none`
  (the RNG is re-seeded every pass) and that a destroyed pod's account is not
  credited with its lost work.
- **Docs.** Upstream kube-scheduler behaviour is now described as the v1.32.2
  source has it: DefaultPreemption's node order and best-effort PDBs, and that
  nominations are reserved only against equal- or lower-priority pods (so
  D-preempt's strict reservation is labelled a simplification). The large-job
  threshold "needs at least a whole largest node", not "cannot fit on one
  node". The homogeneous control reads lower under definition B but higher
  under C-node. The Run 2 and reference-model D-largest numbers are labelled
  by source. "Bit-identical repeats" and "one repeat reproduces Phase 1" hold
  only at a zero startup delay (and `--queue-model none` for the latter).
  50:200 ms is an uncalibrated judgment call, not "recommended". The
  bootstrap false-alarm table has a committed script
  (`scripts/bootstrap_null.py`). The architecture diagram shows definition C,
  stranded GPU-hours and D-preempt, and sacct traces as planned.

### Corrections from the second review

Each was confirmed in review, reproduced before it was fixed,
and is pinned by a test that fails on the old behaviour (checked by reverting
each fix in a scratch copy). Changed numbers are reference-model numbers from
the commands in docs/metrics.md, *Where the numbers on this page came from*.

- **Finished work is never evicted.** Requested evictions in the model ran
  before the pass released finished pods, and on a cluster D-preempt's bind
  pass ran before `_retire`, so an attempt whose work had ended since the last
  pass or poll could be evicted, charged lost time longer than the job, and
  rerun. Requests are now applied after the release (and only if the named pod
  is itself still bound); the runner leaves finished attempts out of the
  victim view and out of `_evict`; a pod first seen Terminating after its work
  ended is recorded as completed. `Attempt.set_factor` no longer moves a
  finished attempt's end.
- **`--topology-penalty extend` stretches gangs that never co-ran.** A job
  with a member that finished before its last sibling bound was exempt from
  the stretch in both paths, so the worst co-schedulers paid the least
  scenario penalty. The factor now uses every member's node; members still
  bound are stretched. One extend run per policy (seed 0, kube queue):
  extension D-random 150.5 → 280.1 GPU-h, D-fifo 440.7 → 493.5. (The old rule
  was never committed; `scripts/extend_prefix_rule.py`, added in the third
  review, rebuilds it from the current tree and reproduces the old values.)
- **Wait under eviction is pending time only.** For an evicted job, "last
  bind − submit" counted the time earlier attempts were bound and Running as
  waiting. D-preempt: mean wait 205.2 → 201.8 min, size–wait ρ 0.27 → 0.28,
  starvation ratio 1.56 → 1.59 (1.62 → 1.66 with `--queue-model none`).
  Unchanged for every run without evictions.
- **Bootstrap tails.** The interval now excludes the same count,
  `floor(R·(1−confidence)/2)`, from each tail (`stats.tail_count`), and is
  documented as that rather than as nearest-rank quantiles it was not; at
  confidence 0.90 the old indices cut 199 values below and 200 above. The
  default (0.95, 4000 resamples) is unchanged, so no published interval moved.
- **Across-trace-seed study.** Its spread was called "WORKLOAD variance, not
  harness variance" above notes saying the opposite. It is now described as
  mixing workload, harness and policy randomness, gets no stability verdict,
  and its comparisons are paired by trace (`stats.compare_paired`).
- **Starvation flag.** It now requires more starvation than D-random's own
  ratio, prints both ratios, and says "not established" when the utilization
  lead is not resolvable. The reference model's D-largest flag (7.20 against
  D-random's 19.14, lead not resolvable) is now listed as not flagged.
- **Runner preflight** refuses kwok nodes outside the fleet (left by `make up`
  with another `FLEET`), and a bind to one mid-run stops the run at once.
- **CLI** refuses a repeated `--config`, which used to pool back-to-back
  copies into a spurious n (two copies at `--repeat 1` read n=2,
  `deterministic`).
- **Fleet files** reject unknown top-level keys (a misspelt `topolgy:` loaded
  as "no topology declared").
- **`make clean`** keeps `cluster/generated/` while the kind cluster is up.
- **Docs.** Definition C's Slurm mapping no longer calls `--switches` a
  whole-leaf-switch boundary (it only caps the leaf-switch count; a
  whole-idle-switch option under `topology/tree` is unverified). The README no
  longer says kube-scheduler "binds one pod at a time" (binding is
  asynchronous), nor that identical nodes cannot fragment.
- **New tests** for mechanisms nothing pinned: the capacity-freed event in the
  model and the runner, the nomination reservation, the backoffQ expiry check
  at a flush tick, interleaved run order, the SERIALISE state record gating K0,
  the n ≥ 5 rule with unequal n, and golden values for the README's
  reference-model table (`tests/test_golden_model.py`). The runner's
  finished-attempt guard is pinned twice: what the binder is offered as
  victims is recorded, and `_evict` is called directly on a finished attempt.
  Removing the victim-view filter alone used to leave every test passing,
  because `_evict`'s own guard hid it from the eviction records.
- The Slurm topology guide is quoted as it reads: segments are offered "When a
  block, ring, or torus3d topology is configured" (docs/metrics.md said "is
  used").

### Corrections from the third review

Each was confirmed in review and reproduced before it was
fixed; each fix has a test that fails without it. No number published in a
committed file changed.

- **Fairness under `--topology-penalty extend`** counted the ASSUMED stretch
  as service. Both execution paths now book the extension per account
  (`Observation.extension_by_account`) and fairness subtracts it, so an
  account's service is the trace's work it received; scoring refuses a split that does not add up. In the
  reference model's default extend run (seed 0, one run per policy) every job
  completed, yet fairness read 1.087 / 1.087 / 1.081 / 1.129 (D-fifo, D-random,
  D-largest, D-preempt) with service ratios up to 1.43; all four now read 1.00.
- **Definition A** kept a pod the control plane destroyed before its bind was
  observed pending until the horizon (the runner records no bind for it). Its
  pending interval now ends at its destruction. Cluster-only: the reference
  model never destroys a pod, so no model number moved.
- **Validation.** `--startup-delay nan`, `0:inf`, a NaN `--grace-seconds` or
  `--cycle-latency`, and a NaN `--sigma-tolerance` passed every range check
  (comparisons with NaN are false) and produced exit-0 runs of NaN or nothing;
  `--speedup 0` ended in a traceback, and `cluster-config write --startup-delay
  0:inf` crashed after writing files. `StartupDelay`, `RunInfo` and
  `TimeScale` now refuse non-finite values, and the CLI refuses them (and a
  speedup that is not above 0) as usage errors before anything runs.
- **Time-scaling status `recorded`, not `applied`.** A cluster results file
  said the scaled kube-scheduler configuration was applied, and results.md
  "as experienced by kube-scheduler", on the strength of
  `cluster-state.json` — written by `make up` before the cluster exists and
  never checked against the running scheduler. It is now stated as configured
  by `make up`, not read back.
- **Tests for mechanisms nothing pinned**, each checked by mutation in a
  scratch copy: the startup delay's separate RNG stream (D-random's binds are
  identical with the delay off and on; the old test passed with the streams
  merged); the runner's checkpoint path, with and without `extend` (rerunning
  the whole job, booking it all as lost, or dropping the extension correction
  all used to pass); and `extend`'s rule that nothing is stretched until every
  gang member is bound, in the model and the runner.
- **Dead code removed:** `Binder.forget` (no caller), `_Track.deleted_by_us`
  (only ever set together with `ended`, so both of its checks were
  unreachable or always true), and `Runner._observe`'s "freed" result, which
  was always False.
- **Docs.** metrics.md said Definition A's two denominators differ by "roughly
  the utilization factor"; they differ by the idle share, close to
  1 − utilization, exactly `fragmentation_of_fleet = fragmentation_rate ×
  free GPU-time / fleet GPU-time`. It called `sbatch --exclusive` the
  boundary "every GPU site uses"; the sbatch page says whole-node exclusivity
  is site- and partition-configured. The README and limitations.md said
  NVIDIA's Volcano post gives no workload; it names four workload classes and
  "thousands of GPUs", and gives no node count, job mix or time window. The
  `NodeClass` and `test_fleet.py` docstrings still said identical nodes cannot
  fragment. The README described edits to a reference-model table that was
  never published; it now says what each D-preempt cell counts and leaves the
  history here. The pre-fix topology-extension figures now have a recipe
  (`scripts/extend_prefix_rule.py`).

### Provenance of the 34% figure

- The README stated as fact that NVIDIA's enhanced Volcano bin-packing "was
  reported at KubeCon EU 2026 to cut GPU node fragmentation by 34% versus the
  default scheduler". The primary source, NVIDIA's technical blog post
  *Practical Tips for Preventing GPU Fragmentation for Volcano Scheduler*
  (2025-03-31,
  https://developer.nvidia.com/blog/practical-tips-for-preventing-gpu-fragmentation-for-volcano-scheduler/),
  contains no 34%, no fragmentation formula and no mention of KubeCon. It
  reports before-and-after snapshots on one DGX Cloud cluster of four-GPU L40S
  nodes, against Volcano's default placement: nodes with all four GPUs free
  18 → 214, and average GPU utilization of roughly 90%. The README, the
  K3 row, docs/limitations.md and docs/metrics.md now cite that post and state
  those metrics and conditions. The 34% figure and its KubeCon attribution are
  marked **unverified**: the only source found is a third-party blog post
  (2026-05-13) that names no talk, speaker or recording. All pages read
  2026-09-26. The README's "Nobody has published a controlled comparison"
  is now "I could not find" one.

### Known gaps

- No cluster run with any of the above (`cluster/upstream.lock`:
  `[scheduler_config] verified = false`).
- Nothing reads the running kube-scheduler's flags back, so a cluster results
  file states its configuration only as recorded by `make up`.
- On a cluster, bind and Running are quantised to the poll interval, and the
  in-process binder observes them more coarsely than kube-scheduler's binds
  (docs/limitations.md).
- D-preempt may over-preempt while a grace lock is running (and, on a
  cluster, while a finished pod awaits deletion), and reserves a nominated
  preemptor's GPUs against higher-priority pods too (kube-scheduler does not).
- K1–K4 and S0 are not built.

## Phase 1

Substrate (kind + kwok), trace generator, K0, three degenerate baselines,
fragmentation definitions A and B, and the Run 2 cluster table.
