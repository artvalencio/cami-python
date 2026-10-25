import numpy as np
import pytest

import cami

KEYS = {'cami_xy', 'cami_yx', 'mutual_info', 'te_xy', 'te_yx', 'directionality'}


@pytest.fixture
def trials():
    """40 short experiments of 150 points where X drives Y."""
    rng = np.random.default_rng(7)
    x = rng.random((150, 40))
    y = np.vstack([rng.random((1, 40)), x[:-1]])
    return x, y


def test_returns_all_measures(trials):
    assert set(cami.multithread_causality(*trials)) == KEYS


@pytest.mark.parametrize('params', [dict(), dict(symbolic_length=(2, 1, 1), n_symbols=3), dict(tau=2, delay=1)])
def test_single_trial_matches_pairwise_functions(params, x, y):
    res = cami.multithread_causality(x[:, None], y[:, None], **params)
    cami_xy, cami_yx = cami.cami(x, y, two_sided=True, **params)
    te_xy, te_yx = cami.transfer_entropy(x, y, two_sided=True, **params)
    assert res['cami_xy'] == pytest.approx(cami_xy, abs=1e-12)
    assert res['cami_yx'] == pytest.approx(cami_yx, abs=1e-12)
    assert res['te_xy'] == pytest.approx(te_xy, abs=1e-12)
    assert res['te_yx'] == pytest.approx(te_yx, abs=1e-12)
    assert res['directionality'] == pytest.approx(cami_xy - cami_yx, abs=1e-12)
    assert res['mutual_info'] == pytest.approx(res['cami_xy'] - res['te_xy'], abs=1e-12)


def test_pooling_short_trials_detects_coupling(trials):
    res = cami.multithread_causality(*trials)
    assert res['cami_xy'] == pytest.approx(1, abs=0.02)
    assert res['cami_yx'] == pytest.approx(0, abs=0.02)
    assert res['directionality'] > 0.9


def test_words_do_not_cross_trial_boundaries():
    # each trial alone has y[n] = x[n-1]; concatenating would break that at the joins
    rng = np.random.default_rng(3)
    x = [rng.random(n) for n in (5, 9, 4)]
    y = [np.r_[rng.random(), xi[:-1]] for xi in x]
    res = cami.multithread_causality(x, y, x_divs=[0.5], y_divs=[0.5])
    words = sum(len(xi) - 1 for xi in x)
    # with perfect coupling CaMI equals the entropy of the pooled past of x
    xs = np.concatenate([xi[:-1] for xi in x]) >= 0.5
    p = xs.mean()
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    assert words == len(xs)
    assert res['cami_xy'] == pytest.approx(h, abs=1e-12)


def test_trials_of_different_lengths_and_nan_padding(trials):
    x, y = trials
    padded_x, padded_y = x.copy(), y.copy()
    padded_x[100:, 0] = np.nan
    padded_y[100:, 0] = np.nan
    as_list = cami.multithread_causality([x[:100, 0]] + list(x[:, 1:].T), [y[:100, 0]] + list(y[:, 1:].T))
    assert cami.multithread_causality(padded_x, padded_y) == pytest.approx(as_list)


def test_mismatched_trials_are_rejected(trials):
    x, y = trials
    with pytest.raises(ValueError):
        cami.multithread_causality(x, y[:, :-1])
    with pytest.raises(ValueError):
        cami.multithread_causality(x, y[:-1])
