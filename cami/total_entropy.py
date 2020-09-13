def total_entropy(x,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,units='bits'):
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
    H=total_entropy(x,symbolic_type='equal-points',n_symbols=10,
    symbolic_length=2,tau=1)
    '''
    
    import numpy as np
    import pandas as pd
    x=np.array(x)
    #initial defs
    tslen=len(x[:,1])
    numelements=len(x[1,:])
    S=np.ones([tslen,numelements])*-1    

    #generating partitions (checking consistency)
    if symbolic_type=='equal-divs':
        xmin,xmax=np.zeros(numelements),np.zeros(numelements)
        xpart=[]
        for j in range(numelements):
            xmin[j],xmax[j] = x[:,j].min(),x[:,j].max()
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xmin[j]+i*(xmax[j]-xmin[j])/n_symbols)
    elif symbolic_type=='equal-points':
        xsort=np.zeros([tslen,numelements])
        xpart=[]
        for j in range(numelements):
            xsort[:,j] = np.sort(x[:,j])
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xsort[floor(i*tslen/n_symbols),j])
    elif symbolic_type=='equal-growth':
        xdiff=np.zeros([tslen-1,numelements])
        xmin,xmax=np.zeros(numelements),np.zeros(numelements)
        xpart=[]
        for j in range(numelements):
            xdiff[:,j]= np.diff(x[:,j])
            xmin[j],xmax[j] = min(xdiff[:,j]),max(xdiff[:,j])
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xmin[j]+i*(xmax[j]-xmin[j])/n_symbols)
    elif symbolic_type=='equal-growth-points':
        xdiffsort=np.zeros([tslen-1,numelements])
        xpart=[]
        for j in range(numelements):
            xdiffsort[:,j] = np.sort(np.diff(x[:,j]))
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xdiffsort[floor(i*(tslen-1)/n_symbols),j])
    elif symbolic_type=='equal-concavity':
        xdiff2=np.zeros([tslen-2,numelements])
        xmin,xmax=np.zeros(numelements),np.zeros(numelements)
        xpart=[]
        for j in range(numelements):
            xdiff2[:,j] = np.diff(np.diff(x[:,j]))
            xmin[j],xmax[j] = min(xdiff2[:,j]),max(xdiff2[:,j])
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xmin[j]+i*(xmax[j]-xmin[j])/n_symbols)
    elif symbolic_type=='equal-concavity-points':
        xdiff2sort=np.zeros([tslen-2,numelements])
        xpart=[]
        for j in range(numelements):
            xdiff2sort[:,j] = np.sort(np.diff(np.diff(x[:,j])))
            xpart.append([])
            for i in range(1,n_symbols):
                xpart[j].append(xdiff2sort[floor(i*(tslen-2)/n_symbols),j])
    else:
        raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')
    
    #calculating symbols
    S=np.ones([tslen,numelements])*-1
    for j in range(numelements):
        for n in range(tslen): #assign data points to partition symbols in x
            for i in range(len(xpart[:,j])):
                if x[n,j]<xpart[i,j]
                    S[n,j]=i
                    break
            if S[n,j]==-1
                S[n,j]=n_symbols-1
    
    #get probs
    #initializing symbolic box-counter
    phi=np.nan([tslen,numelements])
    #calculating individual probabilities
    pindiv=np.zeros([n_symbols^symbolic_length+1,numelements])
    for j in range(numelements):
        for n in range(tau*numelements,tslen):
            phi[n,j]=0
            k=n-l#running index for sum over tau-spaced elements
            for i in range(n-tau*symbolic_length,n-tau,tau):
                phi[n,j]=phi[n,j]+S[k,j]*n_symbols^((n-1)-k)
                k=k+1
            pindiv[phi[n,j],j]=pindiv[phi[n,j],j]+1
        pindiv[:,j]=pindiv[:,j]/sum(pindiv[:,j])
    
    #calculating joint probabilities
    maxidx=phi.max().max()+numelements*tslen
    pjoint=np.zeros(maxidx)
    for n in range(tau*symbolic_length+1,tslen):
        for j in range(numelements):
            pjoint[phi[:,j]]=pjoint[phi[:,j]+1]+1
    pjoint=pjoint/sum(pjoint)
   
    #calculate total entropy
    totalentropy=0;
    for i in range(len(pjoint)):
        if pjoint[i]>0:
            totalentropy=totalentropy+pjoint(i)*log(pjoint(i));

    return totalentropy


