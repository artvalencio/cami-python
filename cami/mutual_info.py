def mutual_info(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,delay=0,units='bits'):
    ''' Calculates the Mutual Information
        between two variables given their
        observable time-series.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
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
    symbolic_length: int, list (2 elements), tuple (2 elements), optional
        The symbolic length of the encoding, i.e., how many
        data points are used to build one symbol. Options:
            - int: same number of points in x, y and future-y points
                e.g.: symbolic-length=2 means 2 points of x and
                    2 points of y constitute a single meaningful symbol
                    (default: symbolic-length=1)
            - list or tuple (2 elements): the first element is the
                number of points of x and the second the number of
                points of y to build the symbolic sequences.
                e.g.: symbolic-length=(3,2) means 3 points of x and
                    2 points of y constitute a single meaningful symbol
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
    mi: float
        The value of the Mutual Information of X and Y

    See Also
    --------
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
    mi = mutual_info(x,y,symbolic_type='equal-points',n_symbols=10,
        symbolic_length=1)
    '''

    import numpy as np
    import pandas as pd
    import cami
    #checking units
    if units=='bit' or units=='bits' or units=='nat' or units=='nats' or units=='ban' or units=='bans':
        pass
    else:
        raise ValueError('Units must be bits or nat or ban. See help on function.')
    #interpolate missing data or trim if its in the edges
    x,y=pd.to_numeric(x,errors='coerce'),pd.to_numeric(y,errors='coerce')
    while np.isnan(x[0]) or np.isnan(y[0]):
        x,y=x[1:],y[1:]
    while np.isnan(x[-1]) or np.isnan(y[-1]):
        x,y=x[:-1],y[:-1]
    def interp_func(data):
        idx_bads=np.isnan(data)
        idx_goods=np.logical_not(idx_bads)
        data_good=data[idx_goods]
        interp_data=np.interp(idx_bads.nonzero()[0],idx_goods.nonzero()[0],data_good)
        data[idx_bads]=interp_data
        return data
    x,y=interp_func(x),interp_func(y)
    if delay>0:
        x,y=x[:-delay],y[delay:]
    elif delay<0:
        x,y=x[delay:],y[:-delay]
    
    #convert to symbolic sequence
    Sx,Sy=cami.symbolic_encoding(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols)
    #calculate tau
    if tau==None:
        def get_tau(data):
            old_corr_val=1
            tau=None
            for i in range(1,len(data)):
                new_corr_val=np.corrcoef(data[:-i],data[i:])[0,1]
                if old_corr_val>0 and new_corr_val<=0:
                    tau=i
                    break
                old_corr_val=new_corr_val
            if tau==None:
                tau=1
            return tau
        tau=max(get_tau(x),get_tau(y))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation',sep='') 
    #get box probabilities
    p_xp,p_yp,p_yf,__,p_xyp,__,lx,ly,__=cami.get_prob(Sx,Sy,n_symbols=n_symbols,symbolic_length=symbolic_length,tau=tau)
    #compute the Mutual Information
    mutual_info=0;
    pmi=np.zeros([n_symbols**lx,n_symbols**ly])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**ly):
            if (p_xp[i]*p_yp[j]>0) and (p_xyp[i,j]>0):
                if units=='nat' or units =='nats':
                    pmi[i,j]=p_xyp[i,j]*np.log(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                elif units=='ban' or units=='bans':
                    pmi[i,j]=p_xyp[i,j]*np.log10(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                else:
                    pmi[i,j]=p_xyp[i,j]*np.log2(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                mutual_info=mutual_info+pmi[i,j]
            else:
                pmi[i,j]=0
    #return results
    return mutual_info
