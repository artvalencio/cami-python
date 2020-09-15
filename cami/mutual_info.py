def mutual_info(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,units='bits'):
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
    if units=='bits' or units=='nat' or units=='ban':
        pass
    else:
        raise ValueError('Units must be bits or nat or ban. See help on function.')
    #convert to symbolic sequence
    Sx,Sy=cami.symbolic_encoding(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols)
    #calculate tau
    if tau==None:
        xcorrel=np.correlate(x,x,mode='full')
        xcorrel=xcorrel[len(x)-1:]/xcorrel[len(x)-1]
        for i in range(1,len(xcorrel)):
            if xcorrel[i-1]>0 and xcorrel[i]<=0:
                tau=i
                break        
    #get box probabilities
    p_xp,p_yp,p_yf,__,p_xyp,__=cami.get_prob(Sx,Sy,n_symbols=n_symbols,symbolic_length=symbolic_length,tau=tau)
    #compute the Mutual Information
    mutual_info=0;
    pmi=np.zeros([n_symbols^lx,n_symbols^lx])
    for i in range(n_symbols^lx):
        for j in range(n_symbols^lx):
            if (p_xp[i]*p_yp[j]>1e-14) and (p_xyp[i,j]>1e-14):
                if units=='nat':
                    pmi[i,j]=p_xyp[i,j]*log(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                elif units=='ban':
                    pmi[i,j]=p_xyp[i,j]*log10(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                else:
                    pmi[i,j]=p_xyp[i,j]*log2(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                mutual_info=mutual_info+pmi[i,j]
            else:
                pmi[i,j]=0
    #return results
    return mutual_info
