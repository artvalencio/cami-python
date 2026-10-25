"""The fast automatic tau must pick exactly the lag the original loop picks."""
import warnings

import numpy as np
import pytest

from cami import _core


def reference_tau(data):
    """The original implementation, kept as an oracle."""
    old_corr_val = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(1, len(data)):
            new_corr_val = np.corrcoef(data[:-i], data[i:])[0, 1]
            if old_corr_val > 0 and new_corr_val <= 0:
                return i
            old_corr_val = new_corr_val
    return 1


def series():
    rng = np.random.default_rng(0)
    out = {}
    for n in (1, 2, 3, 5, 50, 2000):
        out['iid n=%d' % n] = rng.random(n)
    for phi in (0.5, 0.9, 0.99, 0.999):
        e = rng.standard_normal(5000)
        ar = np.zeros(5000)
        for t in range(1, 5000):
            ar[t] = phi * ar[t - 1] + e[t]
        out['AR(1) phi=%g' % phi] = ar
    t = np.arange(4000)
    for period in (40, 42, 7.3, 1000, 3999):
        out['sine period=%g' % period] = np.sin(2 * np.pi * t / period)
    out['two sines'] = np.sin(t / 30) + 0.5 * np.sin(t / 7)
    out['slow sine'] = np.sin(np.arange(20000) / 2000.0)
    out['trend + noise'] = t / 4000 + 0.01 * rng.random(4000)
    out['constant'] = np.ones(300)
    out['constant then noise'] = np.r_[np.zeros(200), rng.random(200)]
    out['steps'] = np.repeat(rng.random(40), 25)
    logistic = np.empty(3000)
    logistic[0] = 0.3
    for k in range(1, 3000):
        logistic[k] = 4 * logistic[k - 1] * (1 - logistic[k - 1])
    out['logistic map'] = logistic
    out['large offset'] = 1e6 + np.sin(t / 50)
    return out


SERIES = series()


@pytest.mark.parametrize('name', SERIES)
def test_fast_tau_matches_the_original_loop(name):
    assert _core.first_zero_autocorrelation(SERIES[name]) == reference_tau(SERIES[name])
