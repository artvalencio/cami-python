"""Chunked counting (used for long series) must give exactly the same results."""
import numpy as np
import pytest

import cami
from cami import _core

PARAMS = [dict(), dict(symbolic_length=(3, 2, 1), n_symbols=3, tau=2), dict(n_symbols=10, symbolic_length=3)]


@pytest.mark.parametrize('params', PARAMS)
def test_chunked_measures_are_identical(params, x, y, monkeypatch):
    expected = cami.measures(x, y, **params)
    expected_rate = cami.cami_rate(x, y, **{k: v for k, v in params.items() if k != 'symbolic_length'})
    monkeypatch.setattr(_core, 'CHUNK_SIZE', 777)
    assert cami.measures(x, y, **params) == expected
    assert cami.cami_rate(x, y, **{k: v for k, v in params.items() if k != 'symbolic_length'}) == expected_rate


def test_chunked_sparse_counting_is_identical(x, y, monkeypatch):
    expected = cami.measures(x, y, n_symbols=3, symbolic_length=2)
    monkeypatch.setattr(_core, 'CHUNK_SIZE', 500)
    monkeypatch.setattr(_core, 'counting_method', lambda size, n: 'sparse')
    assert cami.measures(x, y, n_symbols=3, symbolic_length=2) == expected
