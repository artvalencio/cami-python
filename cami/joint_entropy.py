def joint_entropy(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,delay=0,units='bits'):
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
    #getting symbolic lengths (checking consistency)
    if type(symbolic_length)==int:
        lx,ly = symbolic_length, symbolic_length
    elif type(symbolic_length)==tuple or type(symbolic_length)==list:
        if len(symbolic_length)==2:
            lx,ly = symbolic_length[0], symbolic_length[1]
        else:
            raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    else:
        raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    #initializing boxes
    tslen=len(x)
    phi_x=np.full(tslen,np.nan)
    phi_y=np.full(tslen,np.nan)
    #initializing probabilities of boxes
    p_x=np.zeros(n_symbols**lx)
    p_y=np.zeros(n_symbols**ly)
    p_xy=np.zeros([n_symbols**lx,n_symbols**ly])
    #calculating phi_x
    for n in range(tau*lx,tslen):
        phi_x[n]=0
        k=0
        for i in range(n-tau*lx,n,tau):
            phi_x[n]=phi_x[n]+Sx[i]*n_symbols**(k)#phi is the partition box name of the sequence: e.g.: (|0|..tau..|1|..tau..|0|) => box phi=2
            k=k+1
        p_x[int(phi_x[n])]=p_x[int(phi_x[n])]+1
    p_x=p_x/sum(p_x)
    #calculating phi_y
    for n in range(tau*ly,tslen):
        phi_y[n]=0
        k=0
        for i in range(n-tau*ly,n,tau):
            phi_y[n]=phi_y[n]+Sy[i]*n_symbols**(k)
            k=k+1
        p_y[int(phi_y[n])]=p_y[int(phi_y[n])]+1
    p_y=p_y/sum(p_y)
    #calculating joint probability
    for n in range(tslen):
        if not(np.isnan(phi_x[n]) or np.isnan(phi_y[n])):
            p_xy[int(phi_x[n]),int(phi_y[n])]=p_xy[int(phi_x[n]),int(phi_y[n])]+1
    p_xy=p_xy/sum(sum(p_xy))
    
    #calculating joint entropy
    H=0
    h=np.zeros([n_symbols**lx,n_symbols**ly])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**ly):
            if p_xy[i,j]>0:
                if units=='nat' or units=='nats':
                    h[i,j]=-p_xy[i,j]*np.log(p_xy[i,j])
                elif units=='ban' or units=='bans':
                    h[i,j]=-p_xy[i,j]*np.log10(p_xy[i,j])
                else:
                    h[i,j]=-p_xy[i,j]*np.log2(p_xy[i,j])
                H=H+h[i,j]
            else:
                h[i,j]=0
    return H
