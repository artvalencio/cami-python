"""The dense (bincount) and sparse (sort-based) counting paths must agree exactly."""
import numpy as np
import pytest

from cami import _core

LOG = np.log2


@pytest.mark.parametrize('sizes', [(2,), (8, 8), (4, 16, 4), (1000, 3), (2**20, 2**20)])
def test_dense_and_sparse_counting_give_identical_entropies(sizes):
    rng = np.random.default_rng(len(sizes))
    words = [(rng.integers(0, min(n, 50), 5000), n) for n in sizes]
    dense = _core.entropy_of_codes(LOG, *words, method='dense') if np.prod(sizes, dtype=float) < 2**31 else None
    sparse = _core.entropy_of_codes(LOG, *words, method='sparse')
    auto = _core.entropy_of_codes(LOG, *words)
    assert auto == sparse
    if dense is not None:
        assert dense == sparse


def test_dense_counting_is_used_for_small_alphabets():
    assert _core.counting_method(size=4**9, n_samples=10**6) == 'dense'
    assert _core.counting_method(size=10**9, n_samples=10**6) == 'sparse'
    assert _core.counting_method(size=2**70, n_samples=10) == 'sparse'


@pytest.mark.parametrize('n_vars,size', [(1, 4), (3, 8), (4, 1000), (5, 2**15)])
def test_row_entropy_matches_lexicographic_row_counting(n_vars, size):
    rng = np.random.default_rng(n_vars)
    rows = rng.integers(0, min(size, 6), (4000, n_vars))
    __, counts = np.unique(rows, axis=0, return_counts=True)
    expected = _core.entropy_from_probs(counts / counts.sum(), LOG)
    assert _core.entropy_of_rows(rows, LOG, size) == expected
