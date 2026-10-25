def entropy(x,x_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,units='bits'):
    ''' Calculates the Shannon Entropy
        of a variable from its time-series,
        allowing for different choices
        of binning/symbolic encoding.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The time-series
    x_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the x variable. Select None for placing
        the divisions according to one of the symbolic-type options.
        Default: None.
    symbolic-type: str, optional
        Type of binning or symbolic encoding. Options:
            - 'equal-divs': equal-sized divisions are
                applied across the range of x and y, i.e.
                uniform-length marginal partitions (default)
            - 'equal-points': divisions are selected such
                that the same number of points is present
                in each box, i.e. uniform-density
                partition
            - 'equal-growth': equal-sized divisions are
                applied according to the growth (first
                derivative) of the data
            - 'equal-growth-points': divisions are selected
                such that each growth box has the same number
                of points
            - 'equal-concavity': equal-sized divisions are
                applied according to the concavity (second
                derivative) of the data
            - 'equal-concavity-points': divisions are selected
                such that each concavity box has the same
                number of points
    n_symbols: int, optional
        Number of symbolic symbols, or partition resolution.
        If n_symbols=2, the partition is binary (0,1) (default).
        Is n_symbols=3, it is ternary (0,1,2), and so on.
    symbolic_length: int, optional
        The symbolic length of the encoding, i.e., how many
        data points are used to build one symbol. Symbolic
        length = 1 means simple binning (default), whereas
        values greater than 1 mean that more than one data point
        is required to produce an unit of meaningful information.
    tau: int, None, optional
        Time-delay of reconstruction va method of embedding (Takens'), in number
        of steps. If None, calculates tau as the first zero of auto-correlation.
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    
    Returns
    -------
    H: float
        The value of the Shannon Entropy of this variable encoded
        by the chosen parameters.

    See Also
    --------
    joint_entropy: calculates the Joint Entropy of two
        variables from their time-series, allowing for
        different choices of binning/symbolic encoding.
    cami: calculates the Causal Mutual Information from
        time-series data
    transfer_entropy: calculates the Transfer Entropy
        from time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    mi_rate: calculates the Mutual Information Rate
        with respect to the symbolic length
    pointwise: calculates the pointwise contributions to
        the information measurements
    total_correlation: calculates the ammount of mutual
        of a set of variables

    Example
    -------
    H=entropy(x,symbolic_type='equal-points',n_symbols=10,symbolic_length=1,tau=1)
    '''
    from . import _core
    log = _core.log_function(units)
    x = _core.clean_series(x)
    x_divs = _core.as_divisions(x_divs)
    n_symbols = _core.as_int(n_symbols, 'n_symbols')
    if x_divs is None:
        Sx = _core.symbolize(x, symbolic_type, n_symbols)
    else:
        Sx = _core.encode(x, x_divs)
        n_symbols = len(x_divs) + 1
    tau = _core.resolve_tau(tau, x)
    lx = _core.as_int(symbolic_length, 'symbolic_length')
    if tau * lx >= len(Sx):
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    codes = _core.past_window(Sx, lx, tau, n_symbols, tau * lx, len(Sx))
    return _core.entropy_of_codes(log, (codes, n_symbols ** lx))
