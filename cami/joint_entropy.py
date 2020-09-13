def joint_entropy(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,units='bits'):
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
    '''
    
    import numpy as np
    import pandas as pd
    import cami
    #checking units
    if units=='bits' or units=='nat' or units=='ban':
        pass
    else
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
    #getting symbolic lengths (checking consistency)
    if type(symbolic_length)==int:
        lx,ly = symbolic_length, 2*symbolic_length
    elif type(symbolic_length)==tuple or type(symbolic_length)==list:
        if len(symbolic_length)==2:
            lx,ly = symbolic_length[0], symbolic_length[1]*2
        elif len(symbolic_length)==3:
            lx,ly = symbolic_length[0], symbolic_length[1]+symbolic_length[2]
        else:
            raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    else:
        raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    #initializing boxes
    phi_x=np.nan(tslen)
    phi_y=np.nan(tslen)
    #initializing probabilities of boxes
    p_x=np.zeros(n_symbols^lx+1)
    p_y=np.zeros(n_symbols^lx+1)
    p_xy=np.zeros([n_symbols^lx+1,n_symbols^lx+1])
    #calculating phi_x
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        phi_x[n]=0
        k=n-lx#running index for sum over tau-spaced elements
        for i in range(n-tau*lx,n-tau,tau):
            phi_x[n]=phi_x[n]+Sx[k]*n_symbols^((n-1)-k)
            k=k+1
        p_x[phi_x[n]]=p_xp[phi_x[n]]+1
    p_x=p_x/sum(p_x)
    #calculating phi_yp, about the past of y
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        phi_y[n]=0
        k=n-lx
        for i in range(n-tau*lx,n-tau,tau):
            phi_y[n]=phi_y[n]+Sy[k]*n_symbols^((n-1)-k)
            k=k+1
        p_y[phi_y[n]]=p_yp[phi_y[n]]+1
    p_y=p_y/sum(p_y)
    #calculating joint probability
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        p_xy[phi_x[n],phi_y[n]]=p_xy[phi_x[n],phi_y[n]]+1
    p_xy=p_xy/sum(sum(p_xy))
    
    #calculating entropy
    H=0;
    h=np.zeros([n_symbols^lx,n_symbols^ly])
    for i in range(n_symbols^lx):
        for j in range(n_symbols^ly):
            if p_xy[i,j]>1e-14:
                if units=='nat':
                    h[i,j]=-p_xy[i,j]*np.log(p_xy[i,j])
                elif units='ban':
                    h[i,j]=-p_xy[i,j]*np.log10(p_xy[i,j])
                else:
                    h[i,j]=-p_xy[i,j]*np.log2(p_xy[i,j])
                H=H+h[i,j];
            else:
                h[i,j]=0

    return H
