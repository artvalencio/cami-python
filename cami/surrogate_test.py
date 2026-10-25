def surrogate_test(x,y,measure='cami',n_surrogates=100,method='circular',alpha=0.05,seed=None,**kwargs):
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
    x, y = _core.clean_pair(x, y)
    rng = np.random.default_rng(seed)
    margin = max(1, len(x) // 10)

    def surrogate():
        if method == 'shuffle':
            return rng.permutation(x)
        return np.roll(x, rng.integers(margin, len(x) - margin + 1))

    value = func(x, y, **kwargs)
    surrogates = np.array([func(surrogate(), y, **kwargs) for __ in range(n_surrogates)])
    p_value = (1 + np.sum(surrogates >= value)) / (n_surrogates + 1)
    return {
        'value': value,
        'surrogates': surrogates,
        'threshold': float(np.quantile(surrogates, 1 - alpha)),
        'p_value': float(p_value),
        'significant': bool(p_value <= alpha),
    }
