def multivar_joint_entropy(x,x_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,units='bits'):
    ''' Calculates the Joint Entropy
        of a multivariate systems from
        their time-series,
        allowing for different choices
        of binning/symbolic encoding.

    Parameters
    ----------
    x: list, tuple, np.array, pd.DataFrame
        The multivariate time-series, with each column x[:,i] representing one variable
    x_divs: 2D list, 2D tuple, None, optional
        Partition divisions for the x multivariable. In x_divs=[[a0,b0,c0],[a1,b1,c1],...],
        [a0,b0,c0] are the divisions for the first variable (x[:,0]),
        [a1,b1,c1] are the divisions for the second variable (x[:,1]) and so on.
        Select None for automatically placing the divisions of each variable
        according to one of the symbolic-type options.
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
        data points represent one unit of information.
    tau: int, None, 'mean', 'max', 'median', optional
        Time-delay of reconstruction via method of embedding (Takens'), in number
        of steps. If None or 'mean', calculates tau as the mean over variables of the
        first zero of auto-correlation; 'max' and 'median' aggregate likewise.
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
    H=multivar_joint_entropy(x,symbolic_type='equal-points',n_symbols=10,symbolic_length=2,tau=1)
    '''
    from . import _core
    log = _core.log_function(units)
    words, size = _core.multivariate_words(x, x_divs, symbolic_type, n_symbols, symbolic_length, tau)
    return _core.entropy_of_rows(words, log, size)
