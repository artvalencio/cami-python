def getprob(Sx,Sy,n_symbols=2,symbolic_length=1,tau=1):
    ''' Calculates the probabilities of symbolic sequences at each condition  

    Parameters
    ----------
    Sx: list, tuple, ndarray, pd.Series
        The first symbolic time-series
    Sy: list, tuple, ndarray, pd.Series
        The second symbolic time-series
    lx: symbolic length in x
    ly: total symbolic length in y
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
    
    Returns
    -------
    p_xp,p_yp,p_yf: ndarray
        The marginal probabilities of (past of) x, past of y and future of y.
    p_ypf,p_xyp,p_xypf: ndarray
        The joint probabilities of past of y with future of y,
        (past of) x with past of y, and (past of) x with past of y with future of y.      

    See Also
    -------
    symbolic_encoding: generates a symbolic encoding from the time-series.
    state_space: reconstructs the state_space of the variables from the
        time-series data
    cami: calculates the Causal Mutual Information from time-series data 

    Example
    -------
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf=get_prob(Sx,Sy,n_symbols=10,symbolic_length=(2,2,1),tau=30)

    '''
    import numpy as np
    #getting time-series length and converting if given as pandas series (checking consistency)
    if len(Sx)!=len(Sy):
        raise ValueError('Error: Length of Sx and Sy must be equal!')
    tslen=len(Sx)
    if type(Sx)==pd.Series:
        Sx=Sx.values
    if type(Sy)==pd.Series:
        Sy=Sy.values
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
    phi_yp=np.nan(tslen)
    phi_yf=np.nan(tslen)
    #initializing probabilities of boxes
    #note: n_points=tslen-tau*ly-1;
    p_xp=np.zeros(n_symbols^lx+1)
    p_yp=np.zeros(n_symbols^lx+1)
    p_yf=np.zeros(n_symbols^(ly-lx)+1)
    p_ypf=np.zeros([n_symbols^lx+1,n_symbols^(ly-lx)+1])
    p_xyp=np.zeros([n_symbols^lx+1,n_symbols^lx+1])
    p_xypf=np.zeros([n_symbols^lx+1,n_symbols^lx+1,n_symbols^(ly-lx)+1])
    #calculating phi_x, about the past of x
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        phi_x[n]=0
        k=n-lx#running index for sum over tau-spaced elements
        for i in range(n-tau*lx,n-tau,tau):
            phi_x[n]=phi_x[n]+Sx[k]*n_symbols^((n-1)-k)
            k=k+1
        p_xp[phi_x[n]]=p_xp[phi_x[n]]+1
    p_xp=p_xp/sum(p_xp)
    #calculating phi_yp, about the past of y
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        phi_yp[n]=0
        k=n-lx
        for i in range(n-tau*lx,n-tau,tau):
            phi_yp[n]=phi_yp[n]+Sy[k]*n_symbols^((n-1)-k)
            k=k+1
        p_yp[phi_yp[n]]=p_yp[phi_yp[n]]+1
    p_yp=p_yp/sum(p_yp)
    #calculating phi_yf, about the future of y
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        phi_yf[n]=0
        k=n
        for i in range(n,n+tau*(ly-lx)-1,tau):
            phi_yf[n]=phi_yf[n]+Sy[k]*n_symbols^((n+(ly-lx)-1)-k)
            k=k+1
        p_yf[phi_yf[n]]=p_yf[phi_yf[n]]+1
    p_yf=p_yf/sum(p_yf);
    #calculating joint probabilities
    for n in range(tau*lx+1,tslen-tau*(ly-lx)):
        p_ypf[phi_yp[n],phi_yf[n]]=p_ypf[phi_yp[n],phi_yf[n]]+1
        p_xyp[phi_x[n],phi_yp[n]]=p_xyp[phi_x[n],phi_yp[n]]+1
        p_xypf[phi_x[n],phi_yp[n],phi_yf[n]]=p_xypf[phi_x[n],phi_yp[n],phi_yf[n]]+1
    p_ypf=p_ypf/sum(sum(p_ypf))
    p_xyp=p_xyp/sum(sum(p_xyp))
    p_xypf=p_xypf/sum(sum(sum(p_xypf)))

    #Returning result as list
    return p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf
