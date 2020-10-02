def cami(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,units='bits',two_sided=False):
    ''' Calculates the Causal Mutual Information
        between two variables given their observable
        time-series.

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
        Time-delay of reconstruction va method of embedding (Takens'), in number
        of steps. If None, calculates tau as the first zero of auto-correlation.
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    two_sided: bool, optional
        Whether to calculate CaMI X->Y and CaMI Y->X. Default: False (only X->Y)
    
    Returns
    -------
    cami_xy: float
        The value of the Causal Mutual Information
        in the direction of X->Y
    cami_yx: float (if two_sided=True)
        The value of the Causal Mutual Information
        in the direction of Y->X 

    See Also
    --------
    mutual_info: calculates the Mutual Information from
        time-series data
    transfer_entropy: calculates the Transfer Entropy
        from time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    cami_rate: calculates the Causal Mutual Information
        Rate with respect to the symbolic length
    directionality: calculates the net flow of causal
        information between variables X and Y

    Example
    -------
    cami_xy,cami_yx = cami(x,y,symbolic_type='equal-points',n_symbols=10,
        symbolic_length=1,two_sided=True)
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
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,lx,lyp,lyf=cami.get_prob(Sx,Sy,n_symbols=n_symbols,symbolic_length=symbolic_length,tau=tau)
    if two_sided==True:
        ip_x,ip_yp,ip_yf,ip_ypf,ip_xyp,ip_xypf,__,__,__=cami.get_prob(Sy,Sx,n_symbols=n_symbols,symbolic_length=symbolic_length,tau=tau)
    #calculate CaMI X->Y
    cami_xy=0
    pcami_xy=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**lyp):
            for k in range(n_symbols**lyf):
                if (p_xp[i]*p_ypf[j,k]>0) and (p_xypf[i,j,k]>0):
                    if units=='nat' or units=='nats':
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    elif units=='ban' or units=='bans':
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log10(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    else:
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log2(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    cami_xy=cami_xy+pcami_xy[i,j,k]
                else:
                    pcami_xy[i,j,k]=0
    #calculate CaMI Y->X
    if two_sided==True:
        pcami_yx=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
        cami_yx=0
        for i in range(n_symbols**lx):
            for j in range(n_symbols**lyp):
                for k in range(n_symbols**lyf):
                    if (ip_x[i]*ip_ypf[j,k]>0) and (ip_xypf[i,j,k]>0):
                        if units=='nat' or units=='nats':
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                        elif units=='ban' or units=='bans':
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log10(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                        else:
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log2(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                        cami_yx=cami_yx+pcami_yx[i,j,k]
                    else:
                        pcami_yx[i,j,k]=0
    #Return results
    if two_sided==True:
        return cami_xy,cami_yx
    else:
        return cami_xy
