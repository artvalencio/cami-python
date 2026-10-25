"""Snapshot of numerical outputs, guarding refactors against unintended changes."""
import json

import numpy as np
import pytest

from golden import GOLDEN_FILE, cases

with open(GOLDEN_FILE) as fh:
    EXPECTED = json.load(fh)


@pytest.mark.parametrize('name,func', cases(), ids=[c[0] for c in cases()])
def test_matches_golden_value(name, func):
    np.testing.assert_allclose(np.atleast_1d(func()), EXPECTED[name], rtol=1e-9, atol=1e-12)
