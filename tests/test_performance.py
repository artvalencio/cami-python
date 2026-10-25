import time

import numpy as np
import pytest

import cami


@pytest.mark.parametrize('func', [cami.cami, cami.mutual_info, cami.transfer_entropy, cami.directionality])
def test_large_alphabets_and_long_words_need_no_dense_tables(func, x, y):
    # dense tables would need 10**9 cells (8 GB) and take minutes
    start = time.perf_counter()
    assert np.isfinite(func(x, y, n_symbols=10, symbolic_length=3))
    assert time.perf_counter() - start < 2.0


@pytest.mark.slow
def test_cami_on_long_series_is_fast():
    rng = np.random.default_rng(0)
    x = rng.random(100_000)
    y = np.roll(x, 1)
    start = time.perf_counter()
    cami.cami(x, y, symbolic_length=3, n_symbols=4, two_sided=True)
    assert time.perf_counter() - start < 2.0
