def symbolic_encoding(x,y,x_divs=None,y_divs=None,symbolic_type='equal-divs',n_symbols=2):
    ''' Generate a symbolic encoding from the given
        x,y time-series.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series  
    x_divs: float,list,tuple, np.array, pd.Series, None, optional
            Partition divisions for the x variable. Select None for placing
            the divisions according to one of the symbolic-type options.
            Must have same length as y_divs.
            Default: None.
    y_divs: float,list,tuple, np.array, pd.Series, None, optional
            Partition divisions for the y variable. Select None for placing
            the divisions according to one of the symbolic-type options.
            Must have same length as x_div.
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
    Returns
    -------
    Sx,Sy: np.array
        The symbolic enconded time-series of x,y

    See Also
    --------
    get_prob: calculates the probabilities of symbolic sequences at each condition  
    cami: calculates the Causal Mutual Information from time-series data 

    Examples
    -------
    Sx,Sy = symbolic_encoding(x,y,symbolic_type='equal-points',n_symbols=10)
    Sx,Sy = symbolic_encoding(x,y,x_divs=[0.25,0.75],y_divs=[0.3,0.6])
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

    if x_divs==None and y_divs==None:
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
                xpart.append(xsort[int(i*tslen/n_symbols)])
                ypart.append(ysort[int(i*tslen/n_symbols)])
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
                xpart.append(xdiffsort[int(i*(tslen-1)/n_symbols)])
                ypart.append(ydiffsort[int(i*(tslen-1)/n_symbols)])
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
                xpart.append(xdiff2sort[int(i*(tslen-2)/n_symbols)])
                ypart.append(ydiff2sort[int(i*(tslen-2)/n_symbols)])
        else:
            raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')
    elif (x_divs==None and y_divs!=None) or (x_divs!=None and y_divs==None):
        raise ValueError("Inconsistent use of partition divisions: x_divs and y_divs must both be None or both be a float, list or tuple of same length")
    elif len(x_divs)!=len(y_divs):
        raise ValueError("x_divs and y_divs must have same length")
    elif len(x_divs)==1:
        if type(x_divs)==int or type(x_divs)==float:
            x_divs=[x_divs]
        if type(y_divs)==int or type(y_divs)==float:
            y_divs=[y_divs]
        xpart,ypart=x_divs,y_divs
    else:
        def convert_to_list(a):
            if type(a)==tuple:
                a=list(a)
            elif type(a)==np.ndarray:
                a=a.tolist()
            elif type(a)==pd.Series:
                a=a.to_list()
            elif type(a)!=list:
                raise ValueError("partition data type not supported. x_divs and y_divs must both be None or both be a float, list or tuple of same length")
            return a
        xpart,ypart=convert_to_list(x_divs),convert_to_list(y_divs)
                
    #Generating the symbolic sequences
    def getsequence(x,y,xpart,ypart,tslen,partlen):
        Sx=np.ones(tslen)*-1
        Sy=np.ones(tslen)*-1
        for n in range(tslen): #assign data points to partition symbols in x
            for i in range(partlen):
                if x[n]<xpart[i]:
                    Sx[n]=i
                    break
            if Sx[n]==-1:
                Sx[n]=partlen
        for n in range(tslen): #assign data points to partition symbols in y
            for i in range(partlen):
                if y[n]<ypart[i]:
                    Sy[n]=i
                    break
            if Sy[n]==-1:
                Sy[n]=partlen
        return Sx,Sy
    if x_divs==None:
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
    else:
        Sx,Sy = getsequence(x,y,xpart,ypart,tslen,len(xpart))

    #Returning result
    return Sx,Sy
