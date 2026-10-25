def measures(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,delay=0,units='bits'):
    ''' Calculates the Mutual Information, the Causal Mutual Information
        and the Transfer Entropy in both directions, and the
        Directionality Index, in a single pass over the data.

        The results are identical to calling mutual_info, cami,
        transfer_entropy and directionality separately with the same
        parameters, but the series are cleaned and encoded only once and
        every measure comes from the same symbolic word counts.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
    x_divs, y_divs, symbolic_type, n_symbols, symbolic_length, tau, delay, units:
        As in cami().

    Returns
    -------
    results: dict
        'mutual_info': Mutual Information of X and Y
        'cami_xy', 'cami_yx': Causal Mutual Information X->Y and Y->X
        'te_xy', 'te_yx': Transfer Entropy X->Y and Y->X
        'directionality': Directionality Index, cami_xy - cami_yx

    See Also
    --------
    cami, mutual_info, transfer_entropy, directionality

    Example
    -------
    res = measures(x, y, symbolic_type='equal-points', n_symbols=4, symbolic_length=2)
    res['directionality']
    '''
    from . import _core
    log = _core.log_function(units)
    prepared = _core.Prepared(x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay)
    lengths = _core.parse_symbolic_length(symbolic_length)
    cami_xy, mutual_info, te_xy = _core.measures_from_entropies(prepared.entropies('xy', lengths, log))
    cami_yx, __, te_yx = _core.measures_from_entropies(prepared.entropies('yx', lengths, log))
    return {
        'mutual_info': mutual_info,
        'cami_xy': cami_xy,
        'cami_yx': cami_yx,
        'te_xy': te_xy,
        'te_yx': te_yx,
        'directionality': cami_xy - cami_yx,
    }
