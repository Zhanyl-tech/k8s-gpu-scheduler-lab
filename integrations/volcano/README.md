# Volcano (K2)

Pods set `schedulerName: volcano`. Gangs get an explicit
`scheduling.volcano.sh/v1beta1` PodGroup per job (`_ensure_pod_groups` in
`src/k8slab/runner.py`) with `minMember` set to the gang size, and each pod
names it via the `scheduling.k8s.io/group-name` annotation. Volcano's own
pg-controller would create one `minMember: 1` PodGroup per bare pod, which is
not a gang. `minResources` stays unset so the gang plugin admits on member
count alone. PodGroups use the installer's `default` Queue.

```bash
./scripts/run-volcano-bench.sh [fleet] [jobs] [speedup]
```

That brings up the `k8slab` cluster, installs Volcano, and runs
`integrations/bench_extended.py K2`. Results land in `results-volcano-extended/`.
Run `make down` first if Kueue is installed: its pod webhook gates every pod in
the lab namespace.
