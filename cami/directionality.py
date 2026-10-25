def directionality(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,delay=0,units='bits',tau=1):
    ''' Calculates the Directionality
        Index pointing the net causal
        information flow between two
        variables, given their
        observable time-series.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
    x_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the x variable. Select None for placing
        the divisions according to one of the symbolic_type options.
        Must have same length as y_divs.
        Default: None.
    y_divs: float,list,tuple, np.array, pd.Series, None, optional
        Partition divisions for the y variable. Select None for placing
        the divisions according to one of the symbolic_type options.
        Must have same length as x_divs.
        Default: None. 
    symbolic_type: str, optional
        Type of symbolic encoding. Options:
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
        Number of symbolic symbols, or marginal partition
        resolution. If n_symbols=2, the partition is
        binary (0,1) (default). If n_symbols=3, it is ternary
        (0,1,2), and so on.
    symbolic_length: int, list (2-3 elements), tuple (2-3 elements), optional
        The symbolic length of the encoding, i.e., how many
        data points are used to build one symbol. Options:
            - int: same number of points in x, y and future-y points
                e.g.: symbolic_length=2 means 2 points of x and
                    2 points of y might predict 2 points of the
                    future of y. (default: symbolic_length=1)
            - list or tuple (2 elements): the first element is the
                number of points of x and the second the number of
                points of y and future-y points to build the symbolic
                sequences.
                e.g.: symbolic_length=(3,2) means 3 points of x and
                    2 points of y might predict 2 points of the future
                    of y
            - list or tuple (3 elements): the first element is the
                number of points of x, the second the number of
                points of y and the third the number of points of
                the future of y to build the symbolic sequences.
                e.g.: symbolic_length=(3,2,1) means 3 points of x and
                    2 points of y might predict 1 point of the future
                    of y
    tau: int, None, optional
        Time-delay of reconstruction via the method of embedding (Takens'), in number
        of steps. If None, tau is the first zero crossing of the auto-correlation
        (the selected value is logged to the 'cami' logger).
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits' (or 'bit'): log2 is adopted (default)
            - 'nat' (or 'nats'): ln is adopted
            - 'ban' (or 'bans'): log10 is adopted
    
    Returns
    -------
    di: float
        The value of the Directionality Index (net transfer entropy).
        Positive if X->Y, or negative if Y->X.
    
    Notes
    -----
    DI = CaMI(X->Y) - CaMI(Y->X): positive when more causal information flows
    from X to Y than from Y to X (a net transfer entropy).

    See Also
    --------
    transfer_entropy: calculates the Transfer Entropy
        from time-series data
    cami: calculates the Causal Mutual Information from
        time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    pointwise: calculates the pointwise contributions to
        the information measurements

    Example
    -------
    di = directionality(x,y,symbolic_type='equal-points',n_symbols=10,symbolic_length=1)
    di = directionality(x,y,x_divs=[0.2,0.6],y_divs=[0.3,0.7],symbolic_length=3)
    '''
    from .cami import cami
    cami_xy, cami_yx = cami(x, y, x_divs=x_divs, y_divs=y_divs, symbolic_type=symbolic_type, n_symbols=n_symbols,
                            symbolic_length=symbolic_length, tau=tau, delay=delay, units=units, two_sided=True)
    return cami_xy - cami_yx
