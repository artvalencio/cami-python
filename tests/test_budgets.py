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
    assert best_time(lambda: cami.cami(*big, symbolic_length=3, n_symbols=4, two_sided=True)) < 0.2


def test_total_correlation_time(big):
    x, y = big
    four = np.c_[x, y, np.roll(x, 3), np.roll(y, 5)]
    assert best_time(lambda: cami.total_correlation(four, symbolic_length=2), repeats=1) < 0.5


def test_auto_tau_time():
    from cami import _core
    slow_sine = np.sin(np.arange(20000) / 2000.0)
    assert best_time(lambda: _core.first_zero_autocorrelation(slow_sine)) < 0.03


def test_cami_memory(big):
    # input: two float64 series of 8 MB each
    assert peak_mb(lambda: cami.cami(*big, symbolic_length=3, n_symbols=4, two_sided=True)) < 30


def test_cami_rate_time(big):
    assert best_time(lambda: cami.cami_rate(*big), repeats=2) < 0.2


def test_surrogate_test_time():
    rng = np.random.default_rng(0)
    x = rng.random(10**5)
    y = 0.6 * np.roll(x, 1) + 0.4 * rng.random(10**5)
    assert best_time(lambda: cami.surrogate_test(x, y, n_surrogates=99, seed=0), repeats=2) < 0.12
