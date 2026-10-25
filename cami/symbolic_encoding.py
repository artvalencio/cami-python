def symbolic_encoding(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2):
    ''' Generate a symbolic encoding from the given
        x,y time-series.

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
            Must have same length as x_div.
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
        binary (0,1) (default). Is n_symbols=3, it is ternary
        (0,1,2), and so on.
    Returns
    -------
    Sx,Sy: np.array
        The symbolic enconded time-series of x,y

    See Also
    --------
    get_prob: calculates the probabilities of symbolic sequences at each condition  
    cami: calculates the Causal Mutual Information from time-series data 

    Examples
    -------
    Sx,Sy = symbolic_encoding(x,y,symbolic_type='equal-points',n_symbols=10)
    Sx,Sy = symbolic_encoding(x,y,x_divs=[0.25,0.75],y_divs=[0.3,0.6])
    '''
    from . import _core
    x, y = _core.to_float_array(x), _core.to_float_array(y)
    if len(x) != len(y):
        raise ValueError('Error: Length of x and y must be equal!')
    x_divs, y_divs = _core.resolve_pair_divisions(x_divs, y_divs)
    if x_divs is None:
        n_symbols = _core.as_int(n_symbols, 'n_symbols')
        Sx, Sy = _core.symbolize(x, symbolic_type, n_symbols), _core.symbolize(y, symbolic_type, n_symbols)
    else:
        Sx, Sy = _core.encode(x, x_divs), _core.encode(y, y_divs)
    return Sx.astype(float), Sy.astype(float)
