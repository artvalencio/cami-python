import numpy as np
import pytest

import cami

N_SURR = 99


@pytest.fixture
def coupled(x, z):
    """y[n] = 0.3 x[n-1] + 0.7 noise: weak but real coupling."""
    return x[:2000], (0.3 * np.roll(x, 1) + 0.7 * z)[:2000]


@pytest.mark.parametrize('method', ['circular', 'shuffle'])
def test_coupling_is_significant(coupled, method):
    res = cami.surrogate_test(*coupled, measure='cami', n_surrogates=N_SURR, method=method, seed=1)
    assert res['value'] == cami.cami(*coupled)
    assert res['p_value'] == pytest.approx(1 / (N_SURR + 1))
    assert res['significant']
    assert res['value'] > res['threshold']


def test_independent_series_are_not_significant(x, z):
    res = cami.surrogate_test(x[:2000], z[:2000], measure='transfer_entropy', n_surrogates=N_SURR, seed=1)
    assert res['p_value'] > 0.05
    assert not res['significant']


def test_result_fields_and_surrogate_count(coupled):
    res = cami.surrogate_test(*coupled, n_surrogates=20, seed=0)
    assert set(res) == {'value', 'surrogates', 'threshold', 'p_value', 'significant'}
    assert res['surrogates'].shape == (20,)
    assert res['threshold'] == pytest.approx(np.quantile(res['surrogates'], 0.95))


def test_seed_makes_results_reproducible(coupled):
    a = cami.surrogate_test(*coupled, n_surrogates=10, seed=5)
    b = cami.surrogate_test(*coupled, n_surrogates=10, seed=5)
    c = cami.surrogate_test(*coupled, n_surrogates=10, seed=6)
    np.testing.assert_array_equal(a['surrogates'], b['surrogates'])
    assert not np.array_equal(a['surrogates'], c['surrogates'])


@pytest.mark.parametrize('measure', ['cami', 'mutual_info', 'transfer_entropy', 'directionality'])
def test_named_measures_and_forwarded_parameters(coupled, measure):
    res = cami.surrogate_test(*coupled, measure=measure, n_surrogates=5, seed=0, n_symbols=3, symbolic_length=2)
    assert res['value'] == getattr(cami, measure)(*coupled, n_symbols=3, symbolic_length=2)


def test_callable_measure(coupled):
    res = cami.surrogate_test(*coupled, measure=lambda a, b: float(np.corrcoef(a, b)[0, 1]), n_surrogates=5, seed=0)
    assert np.isfinite(res['value'])


def test_invalid_arguments_are_rejected(coupled):
    with pytest.raises(ValueError, match='measure'):
        cami.surrogate_test(*coupled, measure='entropy-of-nothing')
    with pytest.raises(ValueError, match='method'):
        cami.surrogate_test(*coupled, method='bootstrap')
    with pytest.raises(ValueError, match='alpha'):
        cami.surrogate_test(*coupled, alpha=1.5)
