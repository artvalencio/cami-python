import numpy as np
import pytest

import cami


def test_all_nan_input_is_rejected(x):
    with pytest.raises(ValueError, match='numeric'):
        cami.cami(np.full(100, np.nan), x[:100])
    with pytest.raises(ValueError, match='numeric'):
        cami.entropy(['a', 'b', 'c'])


@pytest.mark.parametrize('func', [
    lambda x, y: cami.cami(x, y, symbolic_length=3, tau=2),
    lambda x, y: cami.mutual_info(x, y, symbolic_length=3, tau=2),
    lambda x, y: cami.transfer_entropy(x, y, symbolic_length=3, tau=2),
    lambda x, y: cami.joint_entropy(x, y, symbolic_length=3, tau=2),
    lambda x, y: cami.entropy(x, symbolic_length=3, tau=2),
])
def test_series_too_short_for_words_is_rejected(func, x, y):
    with pytest.raises(ValueError, match='too short'):
        func(x[:6], y[:6])

