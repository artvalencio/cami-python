"""Input handling shared by every public function."""
import numpy as np
import pandas as pd
import pytest

import cami

DIVS = [0.3, 0.6]

PAIR_FUNCS = {
    'cami': lambda x, y, **kw: cami.cami(x, y, two_sided=True, **kw),
    'mutual_info': cami.mutual_info,
    'transfer_entropy': lambda x, y, **kw: cami.transfer_entropy(x, y, two_sided=True, **kw),
    'directionality': cami.directionality,
    'joint_entropy': cami.joint_entropy,
    'pointwise': lambda x, y, **kw: cami.pointwise(x, y, make_plot=False, **kw),
    'cami_rate': lambda x, y, **kw: cami.cami_rate(x, y, L_limit=3, **kw),
    'mutual_info_rate': lambda x, y, **kw: cami.mutual_info_rate(x, y, L_limit=3, **kw),
    'transfer_entropy_rate': lambda x, y, **kw: cami.transfer_entropy_rate(x, y, L_limit=3, **kw),
}


def assert_same(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


@pytest.fixture
def short_xy(x, y):
    return x[:600].copy(), y[:600].copy()


@pytest.mark.parametrize('name', PAIR_FUNCS)
def test_numpy_array_divisions_equal_list_divisions(name, short_xy):
    x, y = short_xy
    f = PAIR_FUNCS[name]
    expected = f(x, y, x_divs=list(DIVS), y_divs=list(DIVS))
    assert_same(f(x, y, x_divs=np.array(DIVS), y_divs=np.array(DIVS)), expected)


def test_entropy_numpy_array_divisions_equal_list_divisions(x):
    assert cami.entropy(x, x_divs=np.array(DIVS)) == cami.entropy(x, x_divs=list(DIVS))


@pytest.mark.parametrize('name', PAIR_FUNCS)
def test_pandas_series_input_equals_array_input(name, short_xy):
    x, y = short_xy
    f = PAIR_FUNCS[name]
    sx = pd.Series(x, index=np.arange(len(x)) + 100)  # non-zero-based index
    sy = pd.Series(y, index=np.arange(len(y)) + 100)
    assert_same(f(sx, sy), f(x, y))


def test_entropy_pandas_series_input_equals_array_input(x):
    assert cami.entropy(pd.Series(x, index=np.arange(len(x)) + 7)) == cami.entropy(x)


@pytest.mark.parametrize('name', PAIR_FUNCS)
def test_caller_arrays_are_not_modified(name, short_xy):
    x, y = short_xy
    x[10] = np.nan
    y[20] = np.nan
    x_before, y_before = x.copy(), y.copy()
    PAIR_FUNCS[name](x, y)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_entropy_does_not_modify_caller_array(x):
    x = x.copy()
    x[10] = np.nan
    cami.entropy(x)
    assert np.isnan(x[10])


def test_interior_nans_are_interpolated(x, y):
    xn = x.copy()
    xn[100] = np.nan
    xi = x.copy()
    xi[100] = (x[99] + x[101]) / 2
    assert cami.mutual_info(xn, y) == cami.mutual_info(xi, y)


def test_edge_nans_are_trimmed(x, y):
    xn, yn = x.copy(), y.copy()
    xn[0] = np.nan
    yn[-1] = np.nan
    assert cami.mutual_info(xn, yn) == cami.mutual_info(x[1:-1], y[1:-1])


def test_negative_delay_mirrors_positive_delay(x, y):
    assert cami.mutual_info(y, x, delay=-1) == pytest.approx(cami.mutual_info(x, y, delay=1))
    assert cami.mutual_info(y, x, delay=-1) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize('name', PAIR_FUNCS)
@pytest.mark.parametrize('alias,canonical', [('bit', 'bits'), ('nats', 'nat'), ('bans', 'ban')])
def test_unit_aliases_are_accepted(name, alias, canonical, short_xy):
    x, y = short_xy
    f = PAIR_FUNCS[name]
    assert_same(f(x, y, units=alias), f(x, y, units=canonical))


@pytest.mark.parametrize('name', PAIR_FUNCS)
def test_unknown_unit_is_rejected(name, short_xy):
    with pytest.raises(ValueError, match='[Uu]nits'):
        PAIR_FUNCS[name](*short_xy, units='hartley')


def test_numpy_integer_parameters_are_accepted(x, y):
    expected = cami.cami(x, y, n_symbols=3, symbolic_length=2, tau=2)
    got = cami.cami(x, y, n_symbols=np.int64(3), symbolic_length=np.int64(2), tau=np.int32(2))
    assert got == expected
    assert cami.entropy(x, symbolic_length=np.int64(2)) == cami.entropy(x, symbolic_length=2)


def test_mismatched_lengths_are_rejected(x, y):
    with pytest.raises(ValueError, match='[Ll]ength'):
        cami.cami(x, y[:-5])
