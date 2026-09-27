"""Reproduce docs/metrics.md's bootstrap false-alarm table.

Under a null of NO difference -- both sides drawn from one standard normal --
how often does ``k8slab.stats.bootstrap_diff_interval``'s 95% interval exclude
zero? That rate is what "resolvable" means at small n (docs/metrics.md,
"Its weakness at small n, measured").

The procedure, exactly: ONE generator, ``random.Random(12345)``, created once
and consumed for n = 2, 3, 5, 10 in that order. For each n, 2000 trials; each
trial draws ``a`` then ``b`` (n values each, ``gen.gauss(0, 1)``) and scores
them with ``bootstrap_diff_interval(a, b, rng=random.Random(trial))`` at its
default 4000 resamples. The rate is the share of intervals excluding 0.

Re-creating the generator for each n is a different experiment and gives
different numbers; the docs used to leave that ambiguous.

Run from the repository root:

    .venv/bin/python scripts/bootstrap_null.py

It takes a few minutes (32 million resampled means).
"""

from __future__ import annotations

import random
import sys

from k8slab.stats import bootstrap_diff_interval

SIZES = (2, 3, 5, 10)
TRIALS = 2000
SEED = 12345


def false_alarm_rates(
    sizes: tuple[int, ...] = SIZES, trials: int = TRIALS, seed: int = SEED
) -> dict[int, float]:
    """Share of null trials whose interval excludes 0, per n."""
    gen = random.Random(seed)  # once, shared across every n, in order
    rates: dict[int, float] = {}
    for n in sizes:
        hits = 0
        for trial in range(trials):
            a = [gen.gauss(0, 1) for _ in range(n)]
            b = [gen.gauss(0, 1) for _ in range(n)]
            low, high = bootstrap_diff_interval(a, b, rng=random.Random(trial))
            hits += low > 0 or high < 0
        rates[n] = hits / trials
    return rates


def main() -> int:
    for n, rate in false_alarm_rates().items():
        print(f"n={n:>2}: {rate:.2%} of {TRIALS} null trials excluded 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
