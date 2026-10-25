def pointwise(x,y,method='normalized',x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,delay=0,units='bits',two_sided=False,
              make_plot=True,alpha=0.5,figsize=None,dpi=None,labelsize=None,ticksize=None,titlesize=None,save=False,dtype=None):
    ''' Calculates the Pointwise information
        measures between two variables given
        their observable time-series.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
    method: str, optional
        Whether to calculate the (regular) pointwise information measures
        or the normalized information measures (default). Options:
            - 'normalized' or 'normalised' (default): calculates the
                pointwise information measures normalized by -1/log(joint_prob)
            - 'raw' or 'regular' or 'non-normalized' or 'non-normalised': does not
                perfom any normalization procedure
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
        Time-delay of reconstruction via method of embedding (Takens'), in number
        of steps. If None, tau is the first zero crossing of the auto-correlation.
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits' (or 'bit'): log2 is adopted (default)
            - 'nat' (or 'nats'): ln is adopted
            - 'ban' (or 'bans'): log10 is adopted
    two_sided: bool, optional
        Whether to calculate in the directions X->Y and Y->X or only on X->Y.
        Default: False (only X->Y)
    make_plot: bool,optional
        Whether to make scatter plot with the results or not. Default: True.
    alpha: float, optional
        Points transparency level for plot. Default: 0.5
    figsize: 2D tuple or list, None, optional
        Size of plot. Default: None (matplotlib standard)
    dpi: int, None, optional
        Plot resolution (dots per inch). Default: None (matplotlib standard)
    labelsize: int, None, optional
        Font size for plot labels. Default: None (matplotlib standard)
    ticksize: int, None, optional
        Font size for plot tick values. Default: None (matplotlib standard)
    titlesize: int, None, optional
        Font size for subplot titles. Default: None (matplotlib standard)
    dtype: numpy float type, optional
        Type of the returned series, e.g. np.float32 to halve their memory.
        Default: None (float64).
    save: bool, 'colab', optional
        True/False: whether to save or only display plot.
        'colab': option to display and download plot when using Google Colab
        Default: False
    
    Returns
    -------
    point_mi: ndarray
        The value of the Pointwise Mutual Information of X,Y
    point_cami_xy: ndarray
        The value of the Pointwise Causal Mutual Information
        in the direction of X->Y
    point_te_xy: ndarray
        The value of the Pointwise Transfer Entropy
        in the direction of X->Y
    point_cami_yx: ndarray (if two_sided=True)
        The value of the Pointwise Causal Mutual Information
        in the direction of Y->X
    point_te_yx: ndarray (if two_sided=True)
        The value of the Pointwise Transfer Entropy
        in the direction of Y->X
    point_di: ndarray (if two_sided=True)
        The value of the Pointwise Directionality Index

    See Also
    --------
    mutual_info: calculates the Mutual Information from
        time-series data
    cami: calculates the Causal Mutual Information from
        time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    transfer_entropy_rate: calculates the Transfer Entropy
        Rate with respect to the symbolic length
    directionality: calculates the net flow of causal
        information between variables X and Y

    Example
    -------
    point_mi,point_cami_xy,point_te_xy,point_cami_yx,point_te_yx,point_di = pointwise(x,y,
        symbolic_type='equal-points',n_symbols=10,symbolic_length=1,two_sided=True,make_plot=True)
    '''
    import numpy as np
    from . import _core
    log = _core.log_function(units)
    normalized = method not in ('raw', 'regular', 'non-normalized', 'non-normalised')
    prepared = _core.Prepared(x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay)
    x, y = prepared.x, prepared.y
    lengths = _core.parse_symbolic_length(symbolic_length)
    lx = lengths[0]
    dtype = np.float64 if dtype is None else dtype

    #entry i holds the value for the words around time n = i + tau*lx (i is where the past of x starts)
    start, stop = _core.pair_window(prepared.n, lengths, prepared.tau)
    offset = start - prepared.tau * lx

    def new_series():
        return np.full(len(x), np.nan, dtype=dtype)

    def fill(direction, series):
        '''Writes the pointwise MI, CaMI and TE of each sample into series; returns global CaMI, MI.'''
        views = tuple(None if s_ is None else s_[offset:offset + stop - start] for s_ in series)
        entropies = _core.pointwise_values(*prepared.words(direction, lengths), log, normalized, views)
        prepared.drop_future(direction)
        return _core.measures_from_entropies(entropies)[:2]

    point_mi, point_cami_xy, point_te_xy = new_series(), new_series(), new_series()
    cami_xy, mutual_info = fill('xy', (point_mi, point_cami_xy, point_te_xy))
    te_xy = cami_xy - mutual_info
    if two_sided:
        point_cami_yx, point_te_yx = new_series(), new_series()
        cami_yx, mi_yx = fill('yx', (None, point_cami_yx, point_te_yx))
        point_di = point_cami_xy - point_cami_yx
        te_yx = cami_yx - mi_yx

    #log overall information measures
    summary = ['Information values:', 'Mutual Information: %s' % mutual_info,
               'Causal Mutual Information X->Y: %s' % cami_xy]
    if two_sided:
        summary.append('Causal Mutual Information Y->X: %s' % cami_yx)
    summary.append('Transfer Entropy X->Y: %s' % te_xy)
    if two_sided:
        summary += ['Transfer Entropy Y->X: %s' % te_yx, 'Directionality Index: %s' % (cami_xy - cami_yx)]
    _core.logger.info('\n'.join(summary))

    #plotting
    if make_plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if labelsize != None:
            plt.rcParams['axes.labelsize'] = labelsize
        if ticksize != None:
            plt.rcParams['xtick.labelsize'] = ticksize
            plt.rcParams['ytick.labelsize'] = ticksize
        if titlesize != None:
            plt.rcParams['axes.titlesize'] = titlesize
        if two_sided:
            panels = [(point_cami_xy, r'$CaMI_{X\rightarrow Y}$'), (point_cami_yx, r'$CaMI_{Y\rightarrow X}$'),
                      (point_te_xy, r'$TE_{X\rightarrow Y}$'), (point_te_yx, r'$TE_{Y\rightarrow X}$'),
                      (point_mi, r'Mutual Information'), (point_di, r'Directionality Index')]
            layout = 320
        else:
            panels = [(point_cami_xy, r'$CaMI_{X\rightarrow Y}$'), (point_te_xy, r'$TE_{X\rightarrow Y}$'),
                      (point_mi, r'Mutual Information')]
            layout = 310
        for n, (values, title) in enumerate(panels):
            ax = fig.add_subplot(layout + n + 1)
            vlim = np.nanmax(np.abs(values))
            im = ax.scatter(x[:len(values)], y[:len(values)], c=values, cmap='bwr', marker='.', linewidths=0,
                            edgecolors='none', alpha=alpha, vmin=-vlim, vmax=vlim)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save is True or save in ('True', 'colab'):
            plt.savefig('fig.jpg')
        plt.show()
        if save == 'colab':
            from google.colab import files
            files.download('fig.jpg')

    #return results
    if two_sided:
        return point_mi, point_cami_xy, point_te_xy, point_cami_yx, point_te_yx, point_di
    return point_mi, point_cami_xy, point_te_xy
