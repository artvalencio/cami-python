"""One-pass measures and the shared word/count machinery must match the individual functions exactly."""
import numpy as np
import pytest

import cami
from cami import _core

PARAMS = [dict(), dict(symbolic_length=(2, 1, 1), n_symbols=3), dict(symbolic_length=3, tau=2, delay=1),
          dict(x_divs=[0.3, 0.6], y_divs=[0.4, 0.7], symbolic_type='equal-points'), dict(units='nat')]


@pytest.mark.parametrize('params', PARAMS)
def test_measures_match_individual_functions(params, x, y):
    res = cami.measures(x, y, **params)
    cami_xy, cami_yx = cami.cami(x, y, two_sided=True, **params)
    te_xy, te_yx = cami.transfer_entropy(x, y, two_sided=True, **params)
    assert res == {'mutual_info': cami.mutual_info(x, y, **params), 'cami_xy': cami_xy, 'cami_yx': cami_yx,
                   'te_xy': te_xy, 'te_yx': te_yx, 'directionality': cami.directionality(x, y, **params)}


@pytest.mark.parametrize('sizes,method', [((4, 4, 4), 'dense'), ((4, 4, 4), 'sparse'),
                                          ((1000, 1000, 50), 'sparse'), ((64, 16, 8), 'dense')])
def test_pair_entropies_match_separate_counting(sizes, method):
    rng = np.random.default_rng(7)
    a, b, c = [(rng.integers(0, min(n, 30), 20000).astype(np.int32), n) for n in sizes]
    got = _core.pair_entropies(a, b, c, np.log2, method=method)
    expected = {'a': _core.entropy_of_codes(np.log2, a), 'b': _core.entropy_of_codes(np.log2, b),
                'ab': _core.entropy_of_codes(np.log2, a, b), 'bc': _core.entropy_of_codes(np.log2, b, c),
                'abc': _core.entropy_of_codes(np.log2, a, b, c)}
    assert got == expected


@pytest.mark.parametrize('tau', [1, 3])
def test_extended_word_codes_equal_direct_codes(tau, x, y):
    prepared = _core.Prepared(x, y, None, None, 'equal-divs', 3, tau, 0)
    for length in range(1, 6):  # each length extends the previous one
        n = prepared.n
        np.testing.assert_array_equal(prepared.past('x', length),
                                      _core.past_window(prepared.symbols['x'], length, tau, 3, tau * length, n))
        np.testing.assert_array_equal(prepared.future('y', length),
                                      _core.future_window(prepared.symbols['y'], length, tau, 3, 0, n - tau * (length - 1)))
