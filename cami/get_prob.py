def get_prob(Sx,Sy,n_symbols=2,symbolic_length=1,tau=1):
    ''' Calculates the probabilities of symbolic sequences at each condition  

    Parameters
    ----------
    Sx: list, tuple, ndarray, pd.Series
        The first symbolic time-series
    Sy: list, tuple, ndarray, pd.Series
        The second symbolic time-series
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
    lx,lyp,lyf: symbolic length in past of x, past of y and future of y

    See Also
    -------
    symbolic_encoding: generates a symbolic encoding from the time-series.
    state_space: reconstructs the state_space of the variables from the
        time-series data
    cami: calculates the Causal Mutual Information from time-series data 

    Example
    -------
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,lx,lyp,lyf=get_prob(Sx,Sy,n_symbols=10,symbolic_length=(2,2,1),tau=30)

    '''
    import numpy as np
    import pandas as pd
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
        lx,lyp,lyf = symbolic_length,symbolic_length,symbolic_length
    elif type(symbolic_length)==tuple or type(symbolic_length)==list:
        if len(symbolic_length)==2:
            lx,lyp,lyf = symbolic_length[0],symbolic_length[1],symbolic_length[1]
        elif len(symbolic_length)==3:
            lx,lyp,lyf = symbolic_length[0],symbolic_length[1],symbolic_length[2]
        else:
            raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    else:
        raise TypeError('Error: Symbolic length must be int or list/tuple with 2 or 3 elements. See help on function')
    #initializing boxes
    phi_x=np.full(tslen,np.nan)
    phi_yp=np.full(tslen,np.nan)
    phi_yf=np.full(tslen,np.nan)
    #initializing probabilities of boxes
    p_xp=np.zeros(n_symbols**lx)
    p_yp=np.zeros(n_symbols**lyp)
    p_yf=np.zeros(n_symbols**lyf)
    p_ypf=np.zeros([n_symbols**lyp,n_symbols**lyf])
    p_xyp=np.zeros([n_symbols**lx,n_symbols**lyp])
    p_xypf=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
    #calculating phi_x, about the past of x
    for n in range(tau*lx,tslen):
        phi_x[n]=0
        k=0
        for i in range(n-tau*lx,n,tau):
            phi_x[n]=phi_x[n]+Sx[i]*n_symbols**(k)#phi is the partition box name of the sequence: e.g.: (|0|..tau..|1|..tau..|0|) => box phi=2
            k=k+1
        p_xp[int(phi_x[n])]=p_xp[int(phi_x[n])]+1
    p_xp=p_xp/sum(p_xp)
    #calculating phi_yp, about the past of y
    for n in range(tau*lyp,tslen):
        phi_yp[n]=0
        k=0
        for i in range(n-tau*lyp,n,tau):
            phi_yp[n]=phi_yp[n]+Sy[i]*n_symbols**(k)
            k=k+1
        p_yp[int(phi_yp[n])]=p_yp[int(phi_yp[n])]+1
    p_yp=p_yp/sum(p_yp)
    #calculating phi_yf, about the future of y
    for n in range(0,tslen-tau*lyf):
        phi_yf[n]=0
        k=0
        for i in range(n,n+tau*lyf,tau):
            phi_yf[n]=phi_yf[n]+Sy[i]*n_symbols**(k)
            k=k+1
        p_yf[int(phi_yf[n])]=p_yf[int(phi_yf[n])]+1
    p_yf=p_yf/sum(p_yf)
    #calculating joint probabilities
    for n in range(tslen):
        if not(np.isnan(phi_x[n]) or np.isnan(phi_yp[n]) or np.isnan(phi_yf[n])):
            p_ypf[int(phi_yp[n]),int(phi_yf[n])]=p_ypf[int(phi_yp[n]),int(phi_yf[n])]+1
            p_xyp[int(phi_x[n]),int(phi_yp[n])]=p_xyp[int(phi_x[n]),int(phi_yp[n])]+1
            p_xypf[int(phi_x[n]),int(phi_yp[n]),int(phi_yf[n])]=p_xypf[int(phi_x[n]),int(phi_yp[n]),int(phi_yf[n])]+1
    p_ypf=p_ypf/sum(sum(p_ypf))
    p_xyp=p_xyp/sum(sum(p_xyp))
    p_xypf=p_xypf/sum(sum(sum(p_xypf)))

    #Returning result
    return p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,lx,lyp,lyf
