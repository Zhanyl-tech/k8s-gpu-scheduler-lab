from __future__ import annotations

import random

from k8slab.baselines import POLICIES, PendingPod, fifo, largest_first, random_policy


def _pods(spec: list[tuple[int, int]]) -> list[PendingPod]:
    """spec is [(gpus, submit_time)]."""
    return [
        PendingPod(job_id=i + 1, pod_index=0, gpus=g, submit_time=float(t),
                   priority=0, gang_size=1)
        for i, (g, t) in enumerate(spec)
    ]


def test_no_policy_overbooks_a_node() -> None:
    """The invariant every policy must hold: never bind past capacity."""
    free = {"a": 8, "b": 4, "c": 2}
    pods = _pods([(8, 0), (4, 1), (2, 2), (4, 3), (1, 4), (8, 5)])
    for name, policy in POLICIES.items():
        used: dict[str, int] = {}
        for pod, node in policy(pods, dict(free), random.Random(0)):
            used[node] = used.get(node, 0) + pod.gpus
        for node, total in used.items():
            assert total <= free[node], f"{name} overbooked {node}: {total} > {free[node]}"


def test_policies_never_mutate_caller_state() -> None:
    free = {"a": 8, "b": 4}
    pods = _pods([(4, 0), (4, 1)])
    for policy in POLICIES.values():
        snapshot = dict(free)
        policy(pods, free, random.Random(0))
        assert free == snapshot


def test_fifo_is_submission_ordered() -> None:
    free = {"a": 1}
    pods = _pods([(1, 50), (1, 10), (1, 30)])
    placed = fifo(pods, free, random.Random(0))
    assert len(placed) == 1
    assert placed[0][0].submit_time == 10.0


def test_largest_first_takes_the_biggest() -> None:
    free = {"a": 8}
    pods = _pods([(1, 0), (8, 99), (2, 1)])
    placed = largest_first(pods, free, random.Random(0))
    assert placed[0][0].gpus == 8


def test_fifo_and_largest_are_deterministic() -> None:
    free = {"a": 8, "b": 4}
    pods = _pods([(2, 0), (4, 1), (1, 2)])
    for policy in (fifo, largest_first):
        first = [(p.job_id, n) for p, n in policy(pods, dict(free), random.Random(1))]
        second = [(p.job_id, n) for p, n in policy(pods, dict(free), random.Random(2))]
        assert first == second, "a degenerate baseline must not depend on the rng seed"


def test_random_policy_uses_its_rng() -> None:
    free = {f"n{i}": 1 for i in range(12)}
    pods = _pods([(1, i) for i in range(12)])
    a = [n for _, n in random_policy(pods, dict(free), random.Random(1))]
    b = [n for _, n in random_policy(pods, dict(free), random.Random(9))]
    assert a != b


def test_zero_gpu_pods_are_never_bound() -> None:
    """A zero-GPU request would otherwise 'fit' everywhere and skew placement."""
    free = {"a": 8}
    pods = _pods([(0, 0)])
    for policy in POLICIES.values():
        assert policy(pods, dict(free), random.Random(0)) == []
