"""Time and memory budgets (run with: pytest -m slow)."""
import time
import tracemalloc

import numpy as np
import pytest

import cami

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def big():
    rng = np.random.default_rng(0)
    x = rng.random(10**6)
    return x, 0.6 * np.roll(x, 1) + 0.4 * rng.random(10**6)


def best_time(func, repeats=3):
    func()
    times = []
    for __ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def peak_mb(func):
    tracemalloc.start()
    func()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1e6


def test_cami_two_sided_time(big):
    assert best_time(lambda: cami.cami(*big, symbolic_length=3, n_symbols=4, two_sided=True)) < 0.45  # tightened as phases land
