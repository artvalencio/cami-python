def total_correlation(x,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,units='bits'):
    ''' Calculates the Total Entropy
        of a multivariate systems from
        their time-series,
        allowing for different choices
        of binning/symbolic encoding.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The multivariate time-series, with each column
        representing one variable
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
    H=total_correlation(x,symbolic_type='equal-points',n_symbols=10,
    symbolic_length=2,tau=1)
    '''
    
    import numpy as np
    import pandas as pd

    #checking units
    if units=='bit' or units=='bits' or units=='nat' or units=='nats' or units=='ban' or units=='bans':
        pass
    else:
        raise ValueError('Units must be bits or nat or ban. See help on function.')
    #interpolate missing data or trim if its in the edges
    x=pd.DataFrame(x)
    x[:]=x[:].apply(pd.to_numeric,errors='coerce',axis=1)
    x=x.to_numpy()
    for i in range(len(x[0,:])):
        while np.isnan(x[0,i]):
            x[:-1,i]=x[1:,i]
            x[-1,i]=np.nan
    for i in range(len(x[0,:])):
        while np.isnan(x[-1,i]):
            x=x[:-1,:]
    def interp_func(data):
        idx_bads=np.isnan(data)
        idx_goods=np.logical_not(idx_bads)
        data_good=data[idx_goods]
        interp_data=np.interp(idx_bads.nonzero()[0],idx_goods.nonzero()[0],data_good)
        data[idx_bads]=interp_data
        return data
    for i in range(len(x[0,:])):
        x[:,i]=interp_func(x[:,i])

    #select tau
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
    if tau=='average' or tau=='mean' or tau==None:
        tau=int(np.mean([get_tau(x[:,var]) for var in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, average of all trials',sep='')
    elif tau=='max' or tau=='maximum':
        tau=int(max([get_tau(x[:,var]) for var in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, maximum of all trials',sep='')
    elif tau=='median' or tau=='med':
        tau=int(np.median([get_tau(x[:,var]) for var in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, median of all trials',sep='')

    #convert to symbolic sequence
    def multivar_symbolic_encoding(x,symbolic_type=symbolic_type,n_symbols=n_symbols):
        tslen=len(x[:,0])
        nvars=len(x[0,:])
        #generating partitions (checking consistency)
        if symbolic_type=='equal-divs':
            xmin,xmax = x.min(),x.max()
            xpart=[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
        elif symbolic_type=='equal-points':
            xtemp=x.copy()
            xtemp.reshape(-1)
            xsort = np.sort(xtemp)
            xpart = []
            for i in range(1,n_symbols):
                xpart.append(xsort[int(i*tslen/n_symbols)])
        elif symbolic_type=='equal-growth':
            xdiff = np.diff(x,axis=0)
            xmin,xmax = xdiff.min(),xdiff.max()
            xpart=[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
        elif symbolic_type=='equal-growth-points':
            xdiffsort = np.sort(np.diff(x,axis=0).reshape(-1))
            xpart = []
            for i in range(1,n_symbols):
                xpart.append(xdiffsort[int(i*(tslen-1)/n_symbols)])
        elif symbolic_type=='equal-concavity':
            xdiff2 = np.diff(np.diff(x,axis=0),axis=0)
            xmin,xmax = xdiff2.min(),xdiff2.max()
            xpart=[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
        elif symbolic_type=='equal-concavity-points':
            xdiff2sort = np.sort(np.diff(np.diff(x,axis=0),axis=0).reshape(-1))
            xpart = []
            for i in range(1,n_symbols):
                xpart.append(xdiff2sort[int(i*(tslen-2)/n_symbols)])
        else:
            raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')
        #Generating the symbolic sequences
        def getsequence(x,xpart):
            Sx=np.full_like(x,-1)
            for var in range(len(x[0,:])):
                for n in range(len(x[:,var])): #assign data points to partition symbols in x
                    for i in range(len(xpart)):
                        if x[n,var]<xpart[i]:
                            Sx[n,var]=i
                            break
                    if Sx[n,var]==-1:
                        Sx[n,var]=n_symbols-1
            return Sx      
        if symbolic_type=='equal-divs' or symbolic_type=='equal-points':
            Sx = getsequence(x,xpart)
        elif symbolic_type=='equal-growth':
            Sx = getsequence(xdiff,xpart)
        elif symbolic_type=='equal-concavity':
            Sx = getsequence(np.diff(x),xpart)
        elif symbolic_type=='equal-concavity':
            Sx = getsequence(xdiff2,xpart)
        elif symbolic_type=='equal-concavity-points':
            Sx = getsequence(np.diff(np.diff(x)),xpart)
        #Returning result
        return Sx

    Sx=multivar_symbolic_encoding(x,symbolic_type=symbolic_type,n_symbols=n_symbols)      

    tslen=len(Sx[:,0])
    nvars=len(Sx[0,:])
    #get probs
    #initializing symbolic box-counter
    phi=np.full([tslen,nvars],np.nan)
    #calculating individual probabilities
    pindiv=np.zeros([n_symbols**symbolic_length,nvars])
    for var in range(nvars):
        for n in range(tau*nvars,tslen):
            phi[n,var]=0
            k=0
            for i in range(n-tau*symbolic_length,n,tau):
                phi[n,var]=phi[n,var]+Sx[i,var]*n_symbols**(k)
                k=k+1
            pindiv[int(phi[n,var]),var]=pindiv[int(phi[n,var]),var]+1
        pindiv[:,var]=pindiv[:,var]/sum(pindiv[:,var])
    
    #calculating joint probabilities
    maxidx=int(np.nanmax(phi)+1)
    pjoint=np.zeros(maxidx)
    for n in range(tau*symbolic_length+1,tslen):
        for var in range(nvars):
            if not np.isnan(phi[n,var]):
                pjoint[int(phi[n,var])]=pjoint[int(phi[n,var])]+1
    pjoint=pjoint/sum(pjoint)

    #calculate the total correlation
    totalcorrel=0
    for i in range(len(pjoint)):
        #calc product term
        product=1
        for var in range(nvars):
            try:
                if pindiv[i,var]>0:
                    product=product*pindiv[i,var]
            except:
                print(i)
                raise ValueError('Error')
        #calc correl
        if product>0 and pjoint[i]>0:
            if units=='nat' or units=='nats':
                totalcorrel=totalcorrel+pjoint[i]*np.log(pjoint[i]/product)
            elif units=='ban' or units=='bans':
                totalcorrel=totalcorrel+pjoint[i]*np.log10(pjoint[i]/product)
            else:
                totalcorrel=totalcorrel+pjoint[i]*np.log2(pjoint[i]/product)

    return totalcorrel
