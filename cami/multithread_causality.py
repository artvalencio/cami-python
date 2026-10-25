def multithread_causality(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,delay=0,units='bits'):
    ''' Calculates the Causal Mutual Information, Mutual Information,
        Transfer Entropy and Directionality Index from many short,
        independent experiments (trials) of the same phenomenon,
        instead of a single long time-series.

        Symbolic words are formed inside each trial only (never across
        trials) and the probabilities are estimated from the words of all
        trials together. The partition divisions are placed using the data
        of all trials. A single trial gives the same results as cami(),
        mutual_info() and transfer_entropy().

    Parameters
    ----------
    x: 2D np.array, pd.DataFrame, list of 1D series
        The trials of the first variable: columns of a 2D array (rows are
        time), or a list of series which may have different lengths.
        NaN padding at the end of a column is removed.
    y: 2D np.array, pd.DataFrame, list of 1D series
        The trials of the second variable, matching x trial by trial.
    x_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the x variable. Select None for placing
        the divisions according to one of the symbolic-type options.
        Must have same length as y_divs. Default: None.
    y_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the y variable. Default: None.
    symbolic_type: str, optional
        Type of symbolic encoding, as in cami(). Default: 'equal-divs'.
    n_symbols: int, optional
        Number of symbols (partition resolution). Default: 2.
    symbolic_length: int, list or tuple (2-3 elements), optional
        Symbolic lengths of past of x, past of y and future of y, as in cami().
    tau: int, None, optional
        Time-delay of reconstruction. If None, the largest first zero of
        auto-correlation over all trials is used. Default: 1.
    delay: int, optional
        Time-delay between cause and effect, applied inside every trial.
    units: str, optional
        'bits' (default), 'nat' or 'ban'.

    Returns
    -------
    results: dict
        'cami_xy', 'cami_yx': Causal Mutual Information X->Y and Y->X
        'mutual_info': Mutual Information of X and Y
        'te_xy', 'te_yx': Transfer Entropy X->Y and Y->X
        'directionality': Directionality Index, cami_xy - cami_yx

    See Also
    --------
    cami: Causal Mutual Information from a single time-series
    surrogate_test: significance of a measure against surrogate data

    Example
    -------
    res = multithread_causality(x_trials, y_trials, n_symbols=3, symbolic_length=2)
    res['directionality']
    '''
    import numpy as np
    from . import _core
    log = _core.log_function(units)
    x_trials, y_trials = _core.as_trials(x), _core.as_trials(y)
    if len(x_trials) != len(y_trials):
        raise ValueError('Error: x and y must have the same number of trials')
    pairs = [_core.apply_delay(*_core.clean_pair(a, b), delay) for a, b in zip(x_trials, y_trials)]
    x_divs, y_divs = _core.resolve_pair_divisions(x_divs, y_divs)
    if x_divs is None:
        n_symbols = _core.as_int(n_symbols, 'n_symbols')
        symbols = []
        for series in zip(*pairs):  #all x trials, then all y trials
            transformed = [_core.transform(s, symbolic_type) for s in series]
            pooled = np.concatenate([t for t, __ in transformed])
            divisions = _core.place_divisions(pooled, transformed[0][1], n_symbols)
            symbols.append([_core.encode(t, divisions) for t, __ in transformed])
        Sx, Sy = symbols
    else:
        Sx = [_core.encode(a, x_divs) for a, __ in pairs]
        Sy = [_core.encode(b, y_divs) for __, b in pairs]
        n_symbols = len(x_divs) + 1
    tau = _core.resolve_tau(tau, *[s for pair in pairs for s in pair])
    lengths = _core.parse_symbolic_length(symbolic_length)

    def pooled_words(S_cause, S_effect):
        words = [_core.pair_words(a, b, lengths, tau, n_symbols, allow_empty=True) for a, b in zip(S_cause, S_effect)]
        pooled = tuple((np.concatenate([w[i][0] for w in words]), words[0][i][1]) for i in range(3))
        if len(pooled[0][0]) == 0:
            raise ValueError('Error: trials too short for the chosen symbolic_length and tau')
        return pooled

    a, b, c = pooled_words(Sx, Sy)
    cami_xy = _core.cami_from_words(a, b, c, log)
    mutual_info = _core.mutual_info_from_words(a, b, log)
    a, b, c = pooled_words(Sy, Sx)
    cami_yx = _core.cami_from_words(a, b, c, log)
    return {
        'cami_xy': cami_xy,
        'cami_yx': cami_yx,
        'mutual_info': mutual_info,
        'te_xy': cami_xy - mutual_info,
        'te_yx': cami_yx - _core.mutual_info_from_words(a, b, log),
        'directionality': cami_xy - cami_yx,
    }
