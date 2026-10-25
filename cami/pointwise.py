def pointwise(x,y,method='normalized',x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=1,delay=0,units='bits',two_sided=False,
              make_plot=True,alpha=0.5,figsize=None,dpi=None,labelsize=None,ticksize=None,titlesize=None,save=False):
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
        binary (0,1) (default). Is n_symbols=3, it is ternary
        (0,1,2), and so on.
    symbolic_length: int, list (2-3 elements), tuple (2-3 elements), optional
        The symbolic length of the encoding, i.e., how many
        data points are used to build one symbol. Options:
            - int: same number of points in x, y and future-y points
                e.g.: symbolic-length=2 means 2 points of x and
                    2 points of y might predict 2 points of the
                    future of y. (default: symbolic-length=1)
            - list or tuple (2 elements): the first element is the
                number of points of x and the second the number of
                points of y and future-y points to build the symbolic
                sequences.
                e.g.: symbolic-length=(3,2) means 3 points of x and
                    2 points of y might predict 2 points of the future
                    of y
            - list or tuple (3 elements): the first element is the
                number of points of x, the second the number of
                points of y and the third the number of points of
                the future of y to build the symbolic sequences.
                e.g.: symbolic-length=(3,2,1) means 3 points of x and
                    2 points of y might predict 1 point of the future
                    of y
    tau: int, None, optional
        Time-delay of reconstruction via method of embedding (Takens'), in number
        of steps. default=None (first zero correlation crossing).
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bit' or 'bits': log2 is adopted (default)
            - 'nat' or 'nats': ln is adopted
            - 'ban' or 'bans': log10 is adopted
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
        Font size for plot labels. Default: None (matplolib standard)
    ticksize: int, None, optional
        Font size for plot tick values. Default: None (matplolib standard)
    ticksize: int, None, optional
        Font size for subplot titles. Default: None (matplolib standard)
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
    tranfer_entropy_rate: calculates the Transfer Entropy
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
    from .get_prob import get_prob
    log = _core.log_function(units)
    normalized = method not in ('raw', 'regular', 'non-normalized', 'non-normalised')
    x, y, Sx, Sy, n_symbols, tau = _core.prepare_pair(x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay)
    lx, lyp, lyf = _core.parse_symbolic_length(symbolic_length)

    def words(Sa, Sb):
        return (_core.past_codes(Sa, lx, tau, n_symbols), _core.past_codes(Sb, lyp, tau, n_symbols),
                _core.future_codes(Sb, lyf, tau, n_symbols))

    def causal_tables(Sa, Sb):
        '''Pointwise CaMI and TE tables (direction a->b) and the global CaMI.'''
        p_xp, p_yp, __, p_ypf, p_xyp, p_xypf, __, __, __ = get_prob(Sa, Sb, n_symbols=n_symbols, symbolic_length=(lx, lyp, lyf), tau=tau)
        indep = p_xp[:, None, None] * p_ypf[None, :, :]
        mask = (indep > 0) & (p_xypf > 0)
        pcami, pte = np.zeros(p_xypf.shape), np.zeros(p_xypf.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            pcami[mask] = log(p_xypf / indep)[mask]
            pte[mask] = log(p_xypf * p_yp[None, :, None] / (p_xyp[:, :, None] * p_ypf[None, :, :]))[mask]
            total = float(np.sum(p_xypf[mask] * pcami[mask]))
            if normalized:
                pcami[mask] = -pcami[mask] / log(p_xypf[mask])
                pte[mask] = -pte[mask] / log(p_xypf[mask])
        return pcami, pte, total, (p_xp, p_yp, p_xyp)

    pcami_xy, pte_xy, cami_xy, (p_xp, p_yp, p_xyp) = causal_tables(Sx, Sy)
    #pointwise Mutual Information
    indep = p_xp[:, None] * p_yp[None, :]
    mask = (indep > 0) & (p_xyp > 0)
    pmi = np.zeros(p_xyp.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi[mask] = log(p_xyp / indep)[mask]
        mutual_info = float(np.sum(p_xyp[mask] * pmi[mask]))
        if normalized:
            pmi[mask] = -pmi[mask] / log(p_xyp[mask])
    te_xy = cami_xy - mutual_info
    if two_sided:
        pcami_yx, pte_yx, cami_yx, __ = causal_tables(Sy, Sx)
        te_yx = cami_yx - mutual_info

    #print overall information measures:
    print('Information values:')
    if two_sided:
        print('Mutual Information: ', mutual_info,
              '\nCausal Mutual Information X->Y: ', cami_xy,
              '\nCausal Mutual Information Y->X: ', cami_yx,
              '\n Transfer Entropy X->Y: ', te_xy,
              '\n Transfer Entropy Y->X: ', te_yx,
              '\n Directionality Index: ', cami_xy-cami_yx)
    else:
        print('Mutual Information: ', mutual_info,
              '\nCausal Mutual Information X->Y: ', cami_xy,
              '\n Transfer Entropy X->Y: ', te_xy)
    print('--------\nGenerating the time-series of pointwise information measures')

    #passing the results to time-series
    lmax = max([lx, lyp, lyf])

    def to_series(phis, tables):
        phi_a, phi_bp, phi_bf = phis
        out = [np.full(len(x), np.nan) for __ in tables]
        for i in range(len(Sx) - lmax):
            a, b, c = phi_a[i + lx], phi_bp[i + lyp], phi_bf[i + lyf]
            if a >= 0 and b >= 0 and c >= 0:
                for series, table in zip(out, tables):
                    series[i] = table[a, b] if table.ndim == 2 else table[a, b, c]
        return out

    point_mi, point_cami_xy, point_te_xy = to_series(words(Sx, Sy), (pmi, pcami_xy, pte_xy))
    if two_sided:
        point_cami_yx, point_te_yx = to_series(words(Sy, Sx), (pcami_yx, pte_yx))
        point_di = point_cami_xy - point_cami_yx
    print('done!')

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
            vlim = max(abs(min(values)), abs(max(values)))
            im = ax.scatter(x[:len(values)], y[:len(values)], c=values, cmap='bwr', marker='.', linewidths=0,
                            edgecolors='none', alpha=alpha, vmin=-vlim, vmax=vlim)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save == 'colab' or save == 'True':
            plt.savefig('fig.jpg')
        plt.show()
        if save == 'colab':
            from google.colab import files
            files.download('fig.jpg')

    #return results
    if two_sided:
        return point_mi, point_cami_xy, point_te_xy, point_cami_yx, point_te_yx, point_di
    return point_mi, point_cami_xy, point_te_xy
