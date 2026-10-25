"""Analytic properties every estimator must satisfy (characterization tests)."""
import numpy as np
import pytest

import cami

TOL = 0.01


def test_entropy_of_uniform_binary_series_is_one_bit(x):
    assert cami.entropy(x) == pytest.approx(1.0, abs=TOL)


def test_entropy_of_uniform_series_with_four_symbols_is_two_bits(x):
    assert cami.entropy(x, n_symbols=4) == pytest.approx(2.0, abs=TOL)


def test_joint_entropy_of_independent_series_is_sum_of_entropies(x, z):
    assert cami.joint_entropy(x, z) == pytest.approx(2.0, abs=TOL)


def test_joint_entropy_of_identical_series_equals_entropy(x):
    assert cami.joint_entropy(x, x) == pytest.approx(cami.entropy(x), abs=TOL)


def test_mutual_info_of_independent_series_is_zero(x, z):
    assert cami.mutual_info(x, z) == pytest.approx(0.0, abs=TOL)


def test_mutual_info_of_series_with_itself_is_its_entropy(x):
    assert cami.mutual_info(x, x) == pytest.approx(cami.entropy(x), abs=TOL)


def test_cami_detects_direction_of_coupling(x, y):
    cami_xy, cami_yx = cami.cami(x, y, two_sided=True)
    assert cami_xy == pytest.approx(1.0, abs=TOL)
    assert cami_yx == pytest.approx(0.0, abs=TOL)


def test_transfer_entropy_is_cami_minus_mutual_info(x, y):
    te = cami.transfer_entropy(x, y)
    assert te == pytest.approx(cami.cami(x, y) - cami.mutual_info(x, y), abs=1e-3)


def test_directionality_is_difference_of_cami_directions(x, y):
    cami_xy, cami_yx = cami.cami(x, y, two_sided=True)
    assert cami.directionality(x, y) == pytest.approx(cami_xy - cami_yx, abs=1e-12)
    assert cami.directionality(x, y) > 0.9


@pytest.mark.parametrize('units,factor', [('nat', np.log(2)), ('ban', np.log10(2))])
def test_units_rescale_bits(x, y, units, factor):
    assert cami.cami(x, y, units=units) == pytest.approx(cami.cami(x, y) * factor, rel=1e-9)
    assert cami.mutual_info(x, y, units=units) == pytest.approx(cami.mutual_info(x, y) * factor, rel=1e-9)
    assert cami.entropy(x, units=units) == pytest.approx(cami.entropy(x) * factor, rel=1e-9)


def test_delay_aligns_lagged_copy(x, y):
    # y[n] = x[n-1]; delay=1 pairs x[n] with y[n+1] = x[n]
    assert cami.mutual_info(x, y, delay=1) == pytest.approx(1.0, abs=TOL)
