def joint_entropy(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,delay=0,units='bits'):
    ''' Calculates the Joint Entropy
        of two variables from their time-series,
        allowing for different choices
        of binning/symbolic encoding.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
    x_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the x variable. Select None for placing
        the divisions according to one of the symbolic-type options.
        Must have same length as y_divs.
        Default: None.
    y_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the y variable. Select None for placing
        the divisions according to one of the symbolic-type options.
        Must have same length as x_divs.
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
    symbolic_length: int, list (2 elements), tuple (2 elements), optional
        The symbolic length of the encoding, i.e., how many
        data points are used to build one symbol. Options:
            - int: same number of points in x, y and future-y points
                e.g.: symbolic-length=2 means 2 points of x produce a
                single information unit in X and 2 points of y produce a
                single information unit of Y.
            - list or tuple (2 elements): the first element is the
                number of points of x correposponding to a single information
                unit of X, and the second element is the number of
                points of y corresponding to a single information unit of Y.
    tau: int, None, optional
        Time-delay of reconstruction va method of embedding (Takens'), in number
        of steps. If None, calculates tau as the first zero of auto-correlation
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    
    Returns
    -------
    H: float
        The value of the Joint Entropy of these variables encoded
        by the chosen parameters.

    See Also
    --------

    entropy: calculates the Entropy of a variable from its
        time-series data for a given choice of
        binning/symbolic encoding
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
    H=joint_entropy(x,y,symbolic_type='equal-points',n_symbols=10,symbolic_length=(2,1),tau=1)
    H=joint_entropy(x,y,x_divs=[0.2,0.5],y_divs=[0.3,0.7])
    '''
    from . import _core
    log = _core.log_function(units)
    x, y, Sx, Sy, n_symbols, tau = _core.prepare_pair(x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay)
    lx, ly, __ = _core.parse_symbolic_length(symbolic_length, max_parts=2)
    start, stop = tau * max(lx, ly), len(Sx)
    if stop <= start:
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    return _core.entropy_of_codes(log, (_core.past_window(Sx, lx, tau, n_symbols, start, stop), n_symbols ** lx),
                                  (_core.past_window(Sy, ly, tau, n_symbols, start, stop), n_symbols ** ly))
