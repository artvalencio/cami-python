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
    from . import _core
    #getting time-series length and converting if given as pandas series (checking consistency)
    if len(Sx) != len(Sy):
        raise ValueError('Error: Length of Sx and Sy must be equal!')
    Sx = np.asarray(Sx.values if isinstance(Sx, pd.Series) else Sx).astype(np.int64)
    Sy = np.asarray(Sy.values if isinstance(Sy, pd.Series) else Sy).astype(np.int64)
    lx, lyp, lyf = _core.parse_symbolic_length(symbolic_length)
    n_symbols = _core.as_int(n_symbols, 'n_symbols')
    tau = _core.as_int(tau, 'tau')
    nx, nyp, nyf = n_symbols ** lx, n_symbols ** lyp, n_symbols ** lyf
    #phi is the partition box name of the sequence: e.g.: (|0|..tau..|1|..tau..|0|) => box phi=2
    phi_x = _core.past_codes(Sx, lx, tau, n_symbols)
    phi_yp = _core.past_codes(Sy, lyp, tau, n_symbols)
    phi_yf = _core.future_codes(Sy, lyf, tau, n_symbols)
    #marginal probabilities
    p_xp = _core.histogram(phi_x, nx)
    p_yp = _core.histogram(phi_yp, nyp)
    p_yf = _core.histogram(phi_yf, nyf)
    #joint probabilities, over the times where all words exist
    valid = (phi_x >= 0) & (phi_yp >= 0) & (phi_yf >= 0)
    a, b, c = phi_x[valid], phi_yp[valid], phi_yf[valid]
    p_ypf = _core.histogram(b * nyf + c, nyp * nyf).reshape(nyp, nyf)
    p_xyp = _core.histogram(a * nyp + b, nx * nyp).reshape(nx, nyp)
    p_xypf = _core.histogram((a * nyp + b) * nyf + c, nx * nyp * nyf).reshape(nx, nyp, nyf)
    return p_xp, p_yp, p_yf, p_ypf, p_xyp, p_xypf, lx, lyp, lyf
