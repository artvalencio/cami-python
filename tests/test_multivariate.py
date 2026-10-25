import numpy as np
import pandas as pd
import pytest

import cami

TOL = 0.01


@pytest.fixture
def w():
    return np.random.default_rng(31).random(5000)


def test_joint_entropy_of_independent_variables_adds_up(x, z, w):
    assert cami.multivar_joint_entropy(np.c_[x, z]) == pytest.approx(2, abs=TOL)
    assert cami.multivar_joint_entropy(np.c_[x, z, w]) == pytest.approx(3, abs=TOL)
    assert cami.multivar_joint_entropy(np.c_[x, z], symbolic_length=2) == pytest.approx(4, abs=2 * TOL)


def test_joint_entropy_of_copies_equals_entropy(x):
    assert cami.multivar_joint_entropy(np.c_[x, x, x]) == pytest.approx(cami.entropy(x), abs=1e-12)


def test_joint_entropy_of_single_variable_equals_entropy(x):
    for kw in [dict(), dict(symbolic_length=3, tau=2, n_symbols=3)]:
        assert cami.multivar_joint_entropy(x[:, None], **kw) == pytest.approx(cami.entropy(x, **kw), abs=1e-12)


def test_total_correlation_of_independent_variables_is_zero(x, z, w):
    assert cami.total_correlation(np.c_[x, z]) == pytest.approx(0, abs=TOL)
    assert cami.total_correlation(np.c_[x, z, w]) == pytest.approx(0, abs=TOL)


def test_total_correlation_counts_redundancy(x, z):
    h = cami.entropy(x)
    assert cami.total_correlation(np.c_[x, x]) == pytest.approx(h, abs=1e-12)
    assert cami.total_correlation(np.c_[x, x, x]) == pytest.approx(2 * h, abs=1e-12)
    assert cami.total_correlation(np.c_[x, z, x]) == pytest.approx(h, abs=TOL)


def test_total_correlation_of_two_variables_is_their_mutual_information(x, z):
    v = 0.6 * x + 0.4 * z
    assert cami.total_correlation(np.c_[x, v]) == pytest.approx(cami.mutual_info(x, v), abs=TOL)


def test_user_divisions_per_variable(x, z):
    assert cami.total_correlation(np.c_[x, z], x_divs=[[0.5], [0.5]]) == pytest.approx(0, abs=TOL)
    h = cami.multivar_joint_entropy(np.c_[x, 10 * z], x_divs=[[0.25, 0.5, 0.75], [2.5, 5, 7.5]])
    assert h == pytest.approx(4, abs=2 * TOL)


def test_user_divisions_must_match_variables_and_lengths(x, z):
    with pytest.raises(ValueError):
        cami.total_correlation(np.c_[x, z], x_divs=[[0.5]])
    with pytest.raises(ValueError):
        cami.total_correlation(np.c_[x, z], x_divs=[[0.5], [0.3, 0.6]])


def test_dataframe_input_and_units(x, z):
    df = pd.DataFrame({'a': x, 'b': z})
    assert cami.multivar_joint_entropy(df) == cami.multivar_joint_entropy(np.c_[x, z])
    assert cami.multivar_joint_entropy(df, units='nat') == pytest.approx(2 * np.log(2), abs=TOL)


def test_rows_with_edge_nans_are_trimmed_together(x, z):
    data = np.c_[x, z]
    data[0, 0] = np.nan
    data[-1, 1] = np.nan
    assert cami.multivar_joint_entropy(data) == cami.multivar_joint_entropy(np.c_[x, z][1:-1])


@pytest.mark.parametrize('tau', [None, 'mean', 'max', 'median'])
def test_automatic_tau_options(tau, x, z):
    assert cami.total_correlation(np.c_[x, z], tau=tau) == pytest.approx(0, abs=TOL)
