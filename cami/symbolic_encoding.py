def symbolic_encoding(x,y,symbolic_type='equal-divs',n_symbols=2):
    ''' Generate a symbolic encoding from the given
        x,y time-series.

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
    Returns
    -------
    Sx,Sy: np.array
        The symbolic enconded time-series of x,y

    See Also
    --------
    get_prob: calculates the probabilities of symbolic sequences at each condition  
    state_space: reconstructs the state_space of the variables from the
        time-series data
    cami: calculates the Causal Mutual Information from time-series data 

    Example
    -------
    Sx,Sy = symbolic_encoding(x,y,symbolic_type='equal-points',n_symbols=10)
    '''
    #imports and time-series type convesion
    import numpy as np
    import pandas as pd
    if type(x)==pd.Series:
        x=x.values
    if type(y)==pd.Series:
        y=y.values
    #getting time-series length (checking consistency)
    if len(x)!=len(y):
        raise ValueError('Error: Length of x and y must be equal!')
    tslen=len(x)
    #generating partitions (checking consistency)
    if symbolic_type=='equal-divs':
        xmin,xmax,ymin,ymax = x.min(),x.max(),y.min(),y.max()
        xpart,ypart=[],[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
            ypart.append(ymin+i*(ymax-ymin)/n_symbols)
    elif symbolic_type=='equal-points':
        xsort,ysort = np.sort(x),np.sort(y)
        xpart,ypart = [],[]
        for i in range(1,n_symbols):
            xpart.append(xsort[floor(i*tslen/n_symbols)])
            ypart.append(ysort[floor(i*tslen/n_symbols)])
    elif symbolic_type=='equal-growth':
        xdiff,ydiff = np.diff(x),np.diff(y)
        xmin,xmax,ymin,ymax = min(xdiff),max(xdiff),min(ydiff),max(ydiff)
        xpart,ypart=[],[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
            ypart.append(ymin+i*(ymax-ymin)/n_symbols)
    elif symbolic_type=='equal-growth-points':
        xdiffsort,ydiffsort = np.sort(np.diff(x)),np.sort(np.diff(y))
        xpart,ypart = [],[]
        for i in range(1,n_symbols):
            xpart.append(xdiffsort[floor(i*(tslen-1)/n_symbols)])
            ypart.append(ydiffsort[floor(i*(tslen-1)/n_symbols)])
    elif symbolic_type=='equal-concavity':
        xdiff2,ydiff2 = np.diff(np.diff(x)),np.diff(np.diff(y))
        xmin,xmax,ymin,ymax = min(xdiff2),max(xdiff2),min(ydiff2),max(ydiff2)
        xpart,ypart=[],[]
        for i in range(1,n_symbols):
            xpart.append(xmin+i*(xmax-xmin)/n_symbols)
            ypart.append(ymin+i*(ymax-ymin)/n_symbols)
    elif symbolic_type=='equal-concavity-points':
        xdiff2sort,ydiff2sort = np.sort(np.diff(np.diff(x))),np.sort(np.diff(np.diff(y)))
        xpart,ypart = [],[]
        for i in range(1,n_symbols):
            xpart.append(xdiff2sort[floor(i*(tslen-2)/n_symbols)])
            ypart.append(ydiff2sort[floor(i*(tslen-2)/n_symbols)])
    else:
        raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')

    #Generating the symbolic sequences
    def getsequence(x,y,xpart,ypart,tslen,partlen):
        Sx=np.ones(tslen)*-1
        Sy=np.ones(tslen)*-1
        for n in range(tslen) #assign data points to partition symbols in x
            for i in range(partlen)
                if x[n]<xpart[i]
                    Sx[n]=i
                    break
            if Sx[n]==-1
                Sx[n]=n_symbols-1
        for n in range(tslen) #assign data points to partition symbols in y
            for i in range(partlen)
                if y[n]<ypart[i]
                    Sy[n]=i
                    break
            if Sy[n]==-1
                Sy[n]=n_symbols-1
        return Sx,Sy      
    if symbolic_type=='equal-divs' or symbolic_type=='equal-points':
        Sx,Sy = getsequence(x,y,xpart,ypart,tslen,n_symbols-1)
    elif symbolic_type=='equal-growth':
        Sx,Sy = getsequence(xdiff,ydiff,xpart,ypart,tslen-1,n_symbols-1)
    elif symbolic_type=='equal-concavity':
        Sx,Sy = getsequence(np.diff(x),np.diff(y),xpart,ypart,tslen-1,n_symbols-1)
    elif symbolic_type=='equal-concavity':
        Sx,Sy = getsequence(xdiff2,ydiff2,xpart,ypart,tslen-2,n_symbols-1)
    elif symbolic_type=='equal-concavity-points':
        Sx,Sy = getsequence(np.diff(np.diff(x)),np.diff(np.diff(y)),xpart,ypart,tslen-2,n_symbols-1)

    #Returning result
    return Sx,Sy
