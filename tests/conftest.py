import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

N = 5000


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def x(rng):
    """i.i.d. uniform series."""
    return rng.random(N)


@pytest.fixture
def z():
    """Uniform series independent of x."""
    return np.random.default_rng(999).random(N)


@pytest.fixture
def y(x):
    """Lagged copy of x: y[n] = x[n-1], so X drives Y."""
    return np.roll(x, 1)
