def entropy(x,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,units='bits'):
    ''' Calculates the Shannon Entropy
        of a variable from its time-series,
        allowing for different choices
        of binning/symbolic encoding.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The time-series
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
        data points are used to build one symbol. Symbolic
        length = 1 means simple binning (default), whereas
        values greater than 1 mean that more than one data point
        is required to produce an unit of meaningful information.
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    
    Returns
    -------
    H: float
        The value of the Shannon Entropy of this variable encoded
        by the chosen parameters.

    See Also
    --------
    joint_entropy: calculates the Joint Entropy of two
        variables from their time-series, allowing for
        different choices of binning/symbolic encoding.
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
    H=entropy(x,symbolic_type='equal-points',n_symbols=10,symbolic_length=1,tau=1)
    '''

    #imports and time-series type convesion
    import numpy as np
    import pandas as pd
    if type(x)==pd.Series:
        x=x.values
    tslen=len(x)
    #generating partitions (checking consistency)
    if symbolic_type=='equal-divs':
        xmin,xmax = x.min(),x.max()
        xpart=[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
    elif symbolic_type=='equal-points':
        xsort= np.sort(x)
        xpart= []
        for i in range(1,n_symbols):
            xpart.append(xsort[floor(i*tslen/n_symbols)])
    elif symbolic_type=='equal-growth':
        xdiff= np.diff(x)
        xmin,xmax = min(xdiff),max(xdiff)
        xpart=[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
    elif symbolic_type=='equal-growth-points':
        xdiffsort = np.sort(np.diff(x))
        xpart = []
        for i in range(1,n_symbols):
            xpart.append(xdiffsort[floor(i*(tslen-1)/n_symbols)])
    elif symbolic_type=='equal-concavity':
        xdiff2 = np.diff(np.diff(x))
        xmin,xmax = min(xdiff2),max(xdiff2)
        xpart=[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
    elif symbolic_type=='equal-concavity-points':
        xdiff2sort = np.sort(np.diff(np.diff(x)))
        xpart = []
        for i in range(1,n_symbols):
            xpart.append(xdiff2sort[floor(i*(tslen-2)/n_symbols)])
    else:
        raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')

    #Generating the symbolic sequences
    def getsequence(x,xpart,tslen,partlen):
        Sx=np.ones(tslen)*-1
        for n in range(tslen): #assign data points to partition symbols in x
            for i in range(partlen):
                if x[n]<xpart[i]:
                    Sx[n]=i-1
                    break
            if Sx[n]==-1:
                Sx[n]=n_symbols-1
        return Sx     
    if symbolic_type=='equal-divs' or symbolic_type=='equal-points':
        Sx = getsequence(x,xpart,tslen,n_symbols-1)
    elif symbolic_type=='equal-growth':
        Sx = getsequence(xdiff,xpart,tslen-1,n_symbols-1)
    elif symbolic_type=='equal-concavity':
        Sx = getsequence(np.diff(x),xpart,tslen-1,n_symbols-1)
    elif symbolic_type=='equal-concavity':
        Sx = getsequence(xdiff2,xpart,tslen-2,n_symbols-1)
    elif symbolic_type=='equal-concavity-points':
        Sx = getsequence(np.diff(np.diff(x)),xpart,tslen-2,n_symbols-1)

    #getting tau if not given
    if tau==None:
        xcorrel=np.correlate(x,x,mode='full')
        xcorrel=xcorrel[len(x)-1:]/xcorrel[len(x)-1]
        for i in range(1,len(xcorrel)):
            if xcorrel[i-1]>0 and xcorrel[i]<=0:
                tau=i
                break  

    #getting symbolic lengths (checking consistency)
    if type(symbolic_length)==int:
        lx = symbolic_length
    else:
        raise TypeError('Error: Symbolic length must be int. See help on function')
    #initializing boxes
    phi_x=np.nan(tslen)
    #initializing probabilities of boxes
    p_x=np.zeros(n_symbols^lx+1)
    #calculating phi_x, about the past of x
    for n in range(tau*lx+1,tslen-tau*lx):
        phi_x[n]=0
        k=n-lx#running index for sum over tau-spaced elements
        for i in range(n-tau*lx,n-tau,tau):
            phi_x[n]=phi_x[n]+Sx[k]*n_symbols^((n-1)-k)
            k=k+1
        p_x[phi_x[n]]=p_x[phi_x[n]]+1
    p_x=p_x/sum(p_x)

    #calculating entropy
    H=0;
    h=np.zeros([n_symbols^lx])
    for i in range(n_symbols^lx):
        if p_x[i]>1e-14:
            if units=='nat':
                h[i]=-p_x[i]*np.log(p_x[i])
            elif units=='ban':
                h[i]=-p_x[i]*np.log10(p_x[i])
            else:
                h[i]=-p_x[i]*np.log2(p_x[i])
            H=H+h[i];
        else:
            h[i]=0

    return H
