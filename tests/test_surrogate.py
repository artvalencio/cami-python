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


FAST_CONFIGS = [dict(), dict(symbolic_type='equal-points', n_symbols=3), dict(x_divs=[0.4], y_divs=[0.6]),
                dict(symbolic_length=(2, 1, 1), tau=2)]


@pytest.mark.parametrize('measure', ['cami', 'mutual_info', 'transfer_entropy', 'directionality'])
@pytest.mark.parametrize('method', ['circular', 'shuffle'])
@pytest.mark.parametrize('config', range(len(FAST_CONFIGS)))
def test_fast_path_matches_general_path(coupled, measure, method, config):
    kw = dict(measure=measure, n_surrogates=6, method=method, seed=3, **FAST_CONFIGS[config])
    fast = cami.surrogate_test(*coupled, **kw)
    general = cami.surrogate_test(*coupled, _fast=False, **kw)
    np.testing.assert_array_equal(fast['surrogates'], general['surrogates'])
    assert fast['value'] == general['value']


@pytest.mark.parametrize('kw', [dict(symbolic_type='equal-growth'), dict(delay=1), dict(tau=None)])
def test_cases_without_fast_path_still_work(coupled, kw):
    res = cami.surrogate_test(*coupled, n_surrogates=4, seed=0, **kw)
    np.testing.assert_array_equal(res['surrogates'], cami.surrogate_test(*coupled, n_surrogates=4, seed=0, _fast=False, **kw)['surrogates'])


def test_parallel_workers_do_not_change_results(coupled):
    serial = cami.surrogate_test(*coupled, n_surrogates=12, seed=9)
    parallel = cami.surrogate_test(*coupled, n_surrogates=12, seed=9, n_jobs=3)
    np.testing.assert_array_equal(serial['surrogates'], parallel['surrogates'])


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_progress_callback(coupled, n_jobs):
    calls = []
    cami.surrogate_test(*coupled, n_surrogates=5, seed=0, n_jobs=n_jobs, progress=lambda done, total: calls.append((done, total)))
    assert calls == [(k, 5) for k in range(1, 6)]


def test_progress_callback_can_stop_the_test(coupled):
    class Stop(Exception):
        pass

    def stop_after_two(done, total):
        if done == 2:
            raise Stop

    with pytest.raises(Stop):
        cami.surrogate_test(*coupled, n_surrogates=50, seed=0, progress=stop_after_two)
