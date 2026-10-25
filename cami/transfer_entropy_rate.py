def transfer_entropy_rate(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,tau=1,L_limit=6,delay=0,units='bits',make_plot=False,verbose=False):
    ''' Calculates the Transfer Entropy Rate
        between two variables given their
        observable time-series.

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
    L_limit: int, optional
        Limit maximum of the symbolic sequence length to calculate
        Transfer Entropy from. It should be sufficient to enable
        identification of linear TE x L section, but values too high
        come with computational cost without significant benefits.
        Default: 6.
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bit' or 'bits': log2 is adopted (default)
            - 'nat' or 'nats': ln is adopted
            - 'ban' or 'bans': log10 is adopted
    make_plot: bool, optional
        Whether to plot TE x L graph. Default: False.
    verbose: bool, optional
        Displays extra information about the calculation. Default: False.
        
    Returns
    -------
    ter: float
        The value of the Transfer Entropy Rate in the direction X->Y

    See Also
    --------
    mutual_info_rate: calculates the Mutual Information Rate
        between two variables from time-series data
    cami: calculates the Causal Mutual Information
        from time-series data
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

    Examples
    -------
    ter = transfer_entropy_rate(x,y,symbolic_type='equal-points',n_symbols=10,units='bits')
    ter = transfer_entropy_rate(x,y,x_divs=[0.1,0.4],y_divs=[0.2,0.4],L_limit=8)
    '''
    import numpy as np
    from . import _core
    from .transfer_entropy import transfer_entropy
    _core.log_function(units)
    #checking if minimum L_limit is satified
    if L_limit<3:
        raise ValueError('L_limit must be equal or greater than 3')
    x, y = _core.clean_pair(x, y)
    x, y = _core.apply_delay(x, y, delay)
    #calculate the measure as a function of symbolic length
    values = np.array([transfer_entropy(x, y, x_divs=x_divs, y_divs=y_divs, symbolic_type=symbolic_type, n_symbols=n_symbols,
                          symbolic_length=L, tau=tau, units=units) for L in range(1, L_limit+1)])
    #linear regression over the part closest to linear
    ter, idx = _core.linear_rate(values)
    if verbose:
        print('Transfer Entropy Rate based on fitting from L=',idx+1,' to L=',idx+3,': ', ter)
    if make_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(L_limit)+1, values)
        plt.xlabel('Symbolic length (L)')
        plt.ylabel('Transfer Entropy ['+units+']')
        plt.show()
    return ter
