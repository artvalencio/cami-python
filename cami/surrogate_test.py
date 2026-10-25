def surrogate_test(x,y,measure='cami',n_surrogates=100,method='circular',alpha=0.05,seed=None,n_jobs=1,progress=None,_fast=True,**kwargs):
    ''' Tests whether an information measure between x and y is larger
        than expected by chance, by recomputing it on surrogate data in
        which the relationship between x and y has been destroyed.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first (cause) time-series
    y: list, tuple, np.array, pd.Series
        The second (effect) time-series
    measure: str or callable, optional
        'cami' (default), 'mutual_info', 'transfer_entropy', 'directionality',
        or any function f(x, y, **kwargs) returning a float.
    n_surrogates: int, optional
        Number of surrogates. The smallest attainable p-value is
        1/(n_surrogates+1). Default: 100.
    method: str, optional
        How surrogates of x are made (y is kept as is):
            - 'circular' (default): x is circularly shifted by a random
                offset between 10% and 90% of its length, which keeps
                its autocorrelation
            - 'shuffle': x is randomly permuted
    alpha: float, optional
        Significance level. Default: 0.05.
    seed: int, None, optional
        Seed of the random generator, for reproducible results.
    n_jobs: int, optional
        Number of threads computing surrogates (-1: one per CPU). The
        results do not depend on it. Default: 1.
    progress: callable, None, optional
        Called as progress(done, n_surrogates) after each surrogate, e.g. to
        update a progress bar; an exception it raises stops the test.
    **kwargs:
        Passed to the measure (e.g. n_symbols, symbolic_length, tau).

    Returns
    -------
    results: dict
        'value': the measure on the original data
        'surrogates': np.array with the measure on each surrogate
        'threshold': the (1-alpha) quantile of the surrogate values
        'p_value': (1 + #surrogates >= value) / (n_surrogates + 1)
        'significant': whether p_value <= alpha

    See Also
    --------
    cami: calculates the Causal Mutual Information
    multithread_causality: information measures from many short trials

    Example
    -------
    res = surrogate_test(x, y, measure='transfer_entropy', n_symbols=3, seed=0)
    if res['significant']: print('X->Y information flow, p =', res['p_value'])
    '''
    import numpy as np
    from . import _core
    from .cami import cami
    from .mutual_info import mutual_info
    from .transfer_entropy import transfer_entropy
    from .directionality import directionality
    measures = {'cami': cami, 'mutual_info': mutual_info, 'transfer_entropy': transfer_entropy,
                'directionality': directionality}
    if callable(measure):
        func = measure
    elif measure in measures:
        func = measures[measure]
    else:
        raise ValueError('Error: measure must be one of %s or a callable' % sorted(measures))
    if method not in ('circular', 'shuffle'):
        raise ValueError("Error: method must be 'circular' or 'shuffle'")
    if not 0 < alpha < 1:
        raise ValueError('Error: alpha must be between 0 and 1')
    n_surrogates = _core.as_int(n_surrogates, 'n_surrogates')
    n_jobs = _core.as_int(n_jobs, 'n_jobs')
    x, y = _core.clean_pair(x, y)
    n = len(x)
    margin = max(1, n // 10)
    #one independent generator per surrogate: results do not depend on n_jobs
    #(spawned one at a time when needed: same children as spawn(n_surrogates))
    seeds = np.random.SeedSequence(seed)
    generators = (np.random.default_rng(seeds.spawn(1)[0]) for __ in range(n_surrogates))

    def reorder(series, rng):
        if method == 'shuffle':
            return series[rng.permutation(n)]
        return np.roll(series, rng.integers(margin, n - margin + 1))

    value = func(x, y, **kwargs)
    #fast path: shuffling/rolling x then encoding it with divisions placed on its values
    #gives the shuffled/rolled symbols, so surrogates reuse the encoding and y's words
    fast = (_fast and not callable(measure) and kwargs.get('delay', 0) == 0 and kwargs.get('tau', 1) is not None
            and (kwargs.get('x_divs') is not None or kwargs.get('symbolic_type', 'equal-divs') in ('equal-divs', 'equal-points')))
    if fast:
        params = dict(kwargs)
        log = _core.log_function(params.pop('units', 'bits'))
        lengths = _core.parse_symbolic_length(params.pop('symbolic_length', 1))
        prepared = _core.Prepared(x, y, params.pop('x_divs', None), params.pop('y_divs', None),
                                  params.pop('symbolic_type', 'equal-divs'), params.pop('n_symbols', 2),
                                  params.pop('tau', 1), params.pop('delay', 0))
        if params:
            raise TypeError('Error: unexpected parameters %s' % sorted(params))
        #y's words, and their part of the joint key, are the same for every surrogate
        __, past_y, future_y = prepared.words('xy', lengths)
        y_key = _core.effect_key(past_y, future_y)
        start, stop = _core.pair_window(n, lengths, prepared.tau)

        def one(rng):
            surrogate = prepared.with_x_symbols(reorder(prepared.symbols['x'], rng))
            past_x = (_core.past_window(surrogate.symbols['x'], lengths[0], prepared.tau, prepared.n_symbols, start, stop),
                      prepared.n_symbols ** lengths[0])
            cami_xy, mi, te = _core.measures_from_entropies(
                _core.pair_entropies(past_x, past_y, future_y, log, bc_key=y_key))
            if measure == 'directionality':
                return cami_xy - _core.measures_from_entropies(surrogate.entropies('yx', lengths, log))[0]
            return {'cami': cami_xy, 'mutual_info': mi, 'transfer_entropy': te}[measure]
    else:
        def one(rng):
            return func(reorder(x, rng), y, **kwargs)

    surrogates = np.empty(n_surrogates)
    if n_jobs == 1:
        values = (one(rng) for rng in generators)
    else:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None)
        values = pool.map(one, generators)
    try:
        for k, surrogate_value in enumerate(values):
            surrogates[k] = surrogate_value
            if progress is not None:
                progress(k + 1, n_surrogates)
    finally:
        if n_jobs != 1:
            pool.shutdown(wait=True, cancel_futures=True)
    p_value = (1 + np.sum(surrogates >= value)) / (n_surrogates + 1)
    return {
        'value': value,
        'surrogates': surrogates,
        'threshold': float(np.quantile(surrogates, 1 - alpha)),
        'p_value': float(p_value),
        'significant': bool(p_value <= alpha),
    }
