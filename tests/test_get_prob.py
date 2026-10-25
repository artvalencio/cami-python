import warnings

import numpy as np
import pytest

import cami

PARAMS = [dict(), dict(symbolic_length=2, tau=3), dict(symbolic_length=(3, 1, 2), n_symbols=3)]


@pytest.fixture(params=PARAMS, ids=['default', 'L2-tau3', 'L312-ns3'])
def probs(request, x, y):
    n_symbols = request.param.get('n_symbols', 2)
    Sx, Sy = cami.symbolic_encoding(x, y, n_symbols=n_symbols)
    return cami.get_prob(Sx, Sy, **request.param)


def test_probabilities_are_normalized(probs):
    for p in probs[:6]:
        assert p.sum() == pytest.approx(1.0)


def test_marginals_are_sums_of_the_joint_distribution(probs):
    p_xp, p_yp, p_yf, p_ypf, p_xyp, p_xypf = probs[:6]
    np.testing.assert_allclose(p_xyp, p_xypf.sum(axis=2))
    np.testing.assert_allclose(p_ypf, p_xypf.sum(axis=0))
    np.testing.assert_allclose(p_xp, p_xyp.sum(axis=1))
    np.testing.assert_allclose(p_yp, p_xyp.sum(axis=0))
    np.testing.assert_allclose(p_yp, p_ypf.sum(axis=1))
    np.testing.assert_allclose(p_yf, p_ypf.sum(axis=0))


def test_constant_series_carries_no_information(x):
    const = np.ones(len(x))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert cami.entropy(const) == 0
        assert cami.mutual_info(const, x) == pytest.approx(0, abs=1e-15)
        assert cami.cami(const, x) == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize('seed', range(5))
def test_information_measures_are_non_negative(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.random(300), rng.random(300)
    te_ab, te_ba = cami.transfer_entropy(a, b, symbolic_length=2, two_sided=True)
    assert cami.mutual_info(a, b, symbolic_length=2) >= -1e-12
    assert te_ab >= -1e-12 and te_ba >= -1e-12
