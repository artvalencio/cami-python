import numpy as np
import pytest

import cami

TYPES_AND_LENGTH_LOSS = {
    'equal-divs': 0, 'equal-points': 0,
    'equal-growth': 1, 'equal-growth-points': 1,
    'equal-concavity': 2, 'equal-concavity-points': 2,
}


def equal_width_symbols(data, n_symbols):
    lo, hi = data.min(), data.max()
    parts = [lo + i * (hi - lo) / n_symbols for i in range(1, n_symbols)]
    return np.searchsorted(parts, data, side='right')


@pytest.mark.parametrize('symbolic_type,loss', TYPES_AND_LENGTH_LOSS.items())
@pytest.mark.parametrize('n_symbols', [2, 3, 5])
def test_every_type_produces_valid_symbols(symbolic_type, loss, n_symbols, x, z):
    Sx, Sy = cami.symbolic_encoding(x, z, symbolic_type=symbolic_type, n_symbols=n_symbols)
    for S in (Sx, Sy):
        assert len(S) == len(x) - loss
        assert set(np.unique(S)) <= set(range(n_symbols))


@pytest.mark.parametrize('symbolic_type,data', [
    ('equal-divs', lambda x: x),
    ('equal-growth', lambda x: np.diff(x)),
    ('equal-concavity', lambda x: np.diff(x, 2)),
])
def test_equal_width_types_encode_the_right_derivative(symbolic_type, data, x, z):
    Sx, __ = cami.symbolic_encoding(x, z, symbolic_type=symbolic_type, n_symbols=4)
    np.testing.assert_array_equal(Sx, equal_width_symbols(data(x), 4))


@pytest.mark.parametrize('symbolic_type', ['equal-points', 'equal-growth-points', 'equal-concavity-points'])
def test_equal_points_types_balance_symbol_counts(symbolic_type, x, z):
    Sx, __ = cami.symbolic_encoding(x, z, symbolic_type=symbolic_type, n_symbols=4)
    counts = np.bincount(Sx.astype(int), minlength=4)
    assert counts.min() >= 0.95 * len(Sx) / 4


def test_user_divisions_put_boundary_values_in_upper_symbol():
    Sx, Sy = cami.symbolic_encoding([0.1, 0.3, 0.5, 0.6, 0.9], [0, 0, 0, 0, 0], x_divs=[0.3, 0.6], y_divs=[0.3, 0.6])
    np.testing.assert_array_equal(Sx, [0, 1, 1, 2, 2])
    np.testing.assert_array_equal(Sy, [0, 0, 0, 0, 0])


def test_unsorted_user_divisions_are_rejected(x, z):
    with pytest.raises(ValueError, match='increasing'):
        cami.symbolic_encoding(x, z, x_divs=[0.6, 0.3], y_divs=[0.3, 0.6])


def test_unknown_symbolic_type_is_rejected(x, z):
    with pytest.raises(ValueError, match='symbolic type'):
        cami.symbolic_encoding(x, z, symbolic_type='equal-banana')


@pytest.mark.parametrize('symbolic_type', TYPES_AND_LENGTH_LOSS)
def test_measures_accept_every_symbolic_type(symbolic_type, x, y):
    assert np.isfinite(cami.entropy(x, symbolic_type=symbolic_type))
    assert np.isfinite(cami.cami(x, y, symbolic_type=symbolic_type))
    assert np.isfinite(cami.mutual_info(x, y, symbolic_type=symbolic_type))
