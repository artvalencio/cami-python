import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import cami

PARAMS = [dict(), dict(symbolic_length=2, tau=2), dict(symbolic_length=(2, 1, 1)), dict(symbolic_length=(1, 2, 3), tau=3)]


@pytest.fixture
def noisy_pair(x, z):
    """y[n] = 0.6 x[n-1] + 0.4 noise: partially driven by x."""
    y = 0.6 * np.roll(x, 1) + 0.4 * z
    return x, y


@pytest.mark.parametrize('params', PARAMS)
def test_time_average_of_raw_pointwise_values_equals_global_measures(params, noisy_pair):
    x, y = noisy_pair
    pmi, pcami_xy, pte_xy, pcami_yx, pte_yx, pdi = cami.pointwise(
        x, y, method='raw', make_plot=False, two_sided=True, **params)
    mi_length = params.get('symbolic_length', 1)
    mi_length = mi_length[:2] if isinstance(mi_length, tuple) else mi_length
    cami_xy, cami_yx = cami.cami(x, y, two_sided=True, **params)
    te_xy, te_yx = cami.transfer_entropy(x, y, two_sided=True, **params)
    assert np.nanmean(pcami_xy) == pytest.approx(cami_xy, abs=1e-9)
    assert np.nanmean(pcami_yx) == pytest.approx(cami_yx, abs=1e-9)
    assert np.nanmean(pte_xy) == pytest.approx(te_xy, abs=1e-9)
    # mutual_info's window uses a future of length lyp; exact only when lyf == lyp
    lengths = params.get('symbolic_length', 1)
    same_window = not isinstance(lengths, tuple) or lengths[1] == lengths[2]
    mi = cami.mutual_info(x, y, **{**params, 'symbolic_length': mi_length})
    assert np.nanmean(pmi) == pytest.approx(mi, abs=1e-9 if same_window else 1e-4)
    np.testing.assert_allclose(pdi, pcami_xy - pcami_yx)


def test_pointwise_index_is_start_of_the_past_of_x(noisy_pair):
    x, y = noisy_pair
    pmi, pcami, pte = cami.pointwise(x, y, method='raw', make_plot=False)
    # L=1, tau=1: entry i pairs x[i], y[i] with the future y[i+1]
    assert not np.isnan(pcami[:-1]).any()
    assert np.isnan(pcami[-1])


def test_normalized_values_are_bounded(noisy_pair):
    pmi, pcami, pte = cami.pointwise(*noisy_pair, make_plot=False)
    assert np.nanmax(np.abs(pmi)) <= 1 + 1e-12


def test_no_figure_without_make_plot(noisy_pair):
    plt.close('all')
    cami.pointwise(*noisy_pair, make_plot=False)
    assert plt.get_fignums() == []


def test_save_true_writes_the_figure(noisy_pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cami.pointwise(*noisy_pair, make_plot=True, save=True)
    plt.close('all')
    assert os.path.exists(tmp_path / 'fig.jpg')


def test_float32_output(noisy_pair):
    out64 = cami.pointwise(*noisy_pair, method='raw', make_plot=False)
    out32 = cami.pointwise(*noisy_pair, method='raw', make_plot=False, dtype=np.float32)
    assert all(a.dtype == np.float32 for a in out32)
    np.testing.assert_allclose(out32[1], out64[1], rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize('method', ['raw', 'normalized'])
def test_sparse_counting_gives_the_same_pointwise_values(noisy_pair, method, monkeypatch):
    from cami import _core
    dense = cami.pointwise(*noisy_pair, method=method, symbolic_length=2, two_sided=True, make_plot=False)
    monkeypatch.setattr(_core, 'counting_method', lambda size, n: 'sparse')
    sparse = cami.pointwise(*noisy_pair, method=method, symbolic_length=2, two_sided=True, make_plot=False)
    for d, s in zip(dense, sparse):
        np.testing.assert_allclose(s, d, rtol=1e-12, atol=1e-12, equal_nan=True)
