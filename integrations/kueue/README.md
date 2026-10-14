# Kueue (K1)

Kueue admits a Pod against quota (queue-name label plus a scheduling gate), then
`kube-scheduler` binds it. The pod wiring lives in `src/k8slab/runner.py`; gangs
use Kueue's pod-group label and total-count annotation, so all members are
admitted together.

```bash
./scripts/run-kueue-bench.sh [fleet] [jobs] [speedup]
```

That brings up the `k8slab` cluster, installs Kueue, applies
`k8slab kueue-queue --fleet <fleet>` (one ClusterQueue whose `nominalQuota` is
the fleet's total GPU and CPU), and runs `integrations/bench_extended.py K1`.
Results land in `results-kueue-extended/`.

The quota covers the whole fleet, so K1 measures admission and ordering rather
than a quota cap. Shrink the quota to demo quota-limited admission.
