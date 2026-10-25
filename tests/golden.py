"""Shared definition of the golden-value cases.

Run `python tests/golden.py` to regenerate tests/golden_values.json. Only do so
when a change in the numbers is intended, and say why in the commit message.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_FILE = os.path.join(HERE, 'golden_values.json')


def golden_data():
    rng = np.random.default_rng(2024)
    n = 2000
    x = rng.random(n)
    y = np.empty(n)
    y[0] = rng.random()
    y[1:] = 0.7 * x[:-1] + 0.3 * rng.random(n - 1)
    return x, y


PAIR_PARAMS = [
    dict(),
    dict(symbolic_type='equal-points', n_symbols=3),
    dict(symbolic_length=2, tau=2),
    dict(symbolic_length=(2, 1, 1)),
    dict(x_divs=[0.3, 0.6], y_divs=[0.4, 0.7]),
    dict(delay=1),
]


def cases():
    import cami
    x, y = golden_data()
    out = []
    for i, p in enumerate(PAIR_PARAMS):
        out.append(('cami_%d' % i, lambda p=p: cami.cami(x, y, two_sided=True, **p)))
        out.append(('transfer_entropy_%d' % i, lambda p=p: cami.transfer_entropy(x, y, two_sided=True, **p)))
        out.append(('directionality_%d' % i, lambda p=p: cami.directionality(x, y, **p)))
        if not isinstance(p.get('symbolic_length'), tuple):
            out.append(('mutual_info_%d' % i, lambda p=p: cami.mutual_info(x, y, **p)))
    out.append(('entropy_0', lambda: cami.entropy(x)))
    out.append(('entropy_1', lambda: cami.entropy(x, symbolic_type='equal-points', n_symbols=3, symbolic_length=2)))
    out.append(('entropy_2', lambda: cami.entropy(x, x_divs=[0.2, 0.5])))
    out.append(('joint_entropy_0', lambda: cami.joint_entropy(x, y)))
    out.append(('joint_entropy_1', lambda: cami.joint_entropy(x, y, symbolic_length=(2, 1), n_symbols=3)))
    out.append(('mutual_info_rate_0', lambda: cami.mutual_info_rate(x, y, L_limit=4)))
    out.append(('cami_rate_0', lambda: cami.cami_rate(x, y, L_limit=4)))
    out.append(('transfer_entropy_rate_0', lambda: cami.transfer_entropy_rate(x, y, L_limit=4)))
    return out


def compute_all():
    return {name: np.atleast_1d(f()).tolist() for name, f in cases()}


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(HERE))
    with open(GOLDEN_FILE, 'w') as fh:
        json.dump(compute_all(), fh, indent=1, sort_keys=True)
    print('wrote', GOLDEN_FILE)
