"""Benchmarks for cami-python: best-of-k wall time and tracemalloc peak per case.

    python benchmarks/run.py LABEL [--quick] [--only SUBSTRING]

Results are stored under LABEL in benchmarks/results.json and printed next to
the 'baseline' entry when one exists.
"""
import json
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cami  # noqa: E402

RESULTS = os.path.join(HERE, 'results.json')


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    return x, 0.6 * np.roll(x, 1) + 0.4 * rng.random(n)


def cases(quick):
    big = 10**5 if quick else 10**6
    x, y = data(big)
    xs, ys = data(10**5)
    slow_sine = np.sin(np.arange(20000) / 2000.0)
    four = np.c_[x, y, np.roll(x, 3), np.roll(y, 5)]
    return {
        'cami L=1 ns=2': lambda: cami.cami(x, y),
        'cami two_sided L=3 ns=4': lambda: cami.cami(x, y, symbolic_length=3, n_symbols=4, two_sided=True),
        'cami L=3 ns=10': lambda: cami.cami(x, y, symbolic_length=3, n_symbols=10),
        'transfer_entropy two_sided L=2': lambda: cami.transfer_entropy(x, y, symbolic_length=2, two_sided=True),
        'mutual_info L=2': lambda: cami.mutual_info(x, y, symbolic_length=2),
        'entropy L=3': lambda: cami.entropy(x, symbolic_length=3),
        'cami_rate L_limit=6': lambda: cami.cami_rate(x, y),
        'pointwise two_sided L=3 ns=4': lambda: cami.pointwise(x, y, symbolic_length=3, n_symbols=4, two_sided=True, make_plot=False),
        'total_correlation 4 vars L=2': lambda: cami.total_correlation(four, symbolic_length=2),
        'multithread 200 trials x 1000': lambda: cami.multithread_causality(x[:200000].reshape(1000, 200), y[:200000].reshape(1000, 200), symbolic_length=2),
        'surrogate_test 99 (N=1e5)': lambda: cami.surrogate_test(xs, ys, n_surrogates=99, seed=0),
        'auto tau slow sine (N=2e4)': lambda: cami.cami(slow_sine, slow_sine, tau=None),
    }


def measure(func, repeats):
    times = []
    for __ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    func()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {'time_ms': round(min(times) * 1000, 2), 'peak_mb': round(peak / 1e6, 2)}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'latest'
    quick = '--quick' in sys.argv
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else ''
    stored = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    baseline = stored.get('baseline', {})
    results = dict(stored.get(label, {}))
    for name, func in cases(quick).items():
        if only not in name:
            continue
        func()  # warm-up
        results[name] = r = measure(func, 1 if quick else 3)
        base = baseline.get(name)
        compare = ''
        if base and label != 'baseline':
            compare = '  (baseline %9.1f ms %7.1f MB, %5.1fx faster)' % (
                base['time_ms'], base['peak_mb'], base['time_ms'] / max(r['time_ms'], 1e-3))
        print('%-34s %9.1f ms %7.1f MB%s' % (name, r['time_ms'], r['peak_mb'], compare), flush=True)
    if not quick:
        stored[label] = results
        with open(RESULTS, 'w') as fh:
            json.dump(stored, fh, indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
