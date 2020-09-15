def pointwise(x,y,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,units='bits',two_sided==False):
    ''' Calculates the Pointwise information
        measures between two variables given
        their observable time-series.

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
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    two_sided: bool, optional
        Whether to calculate in the directions X->Y and Y->X or only on X->Y.
        Default: False (only X->Y)
    
    Returns
    -------
    point_mi: ndarray
        The value of the Pointwise Mutual Information of X,Y
    point_cami_xy: ndarray
        The value of the Pointwise Causal Mutual Information
        in the direction of X->Y
    point_te_xy: ndarray
        The value of the Pointwise Transfer Entropy
        in the direction of X->Y
    point_cami_yx: ndarray (if two_sided=True)
        The value of the Pointwise Causal Mutual Information
        in the direction of Y->X
    point_te_yx: ndarray (if two_sided=True)
        The value of the Pointwise Transfer Entropy
        in the direction of Y->X
    point_di: ndarray (if two_sided=True)
        The value of the Pointwise Directionality Index

    See Also
    --------
    mutual_info: calculates the Mutual Information from
        time-series data
    cami: calculates the Causal Mutual Information from
        time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    tranfer_entropy_rate: calculates the Transfer Entropy
        Rate with respect to the symbolic length
    directionality: calculates the net flow of causal
        information between variables X and Y

    Example
    -------
    point_mi,point_cami_xy,point_te_xy,point_cami_yx,point_te_yx,point_di = pointwise(x,y,
        symbolic_type='equal-points',n_symbols=10,symbolic_length=1,two_sided=True)
    '''
    import numpy as np
    import pandas as pd
    import cami
    #checking units
    if units=='bits' or units=='nat' or units=='ban':
        pass
    else:
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
            
    #getting symbolic lengths
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

    #get box names and probabilities
    def full_get_prob(Sx,Sy,n_symbols=2,tau=1,lx,ly):
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
        return p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf
    
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf=full_get_prob(Sx,Sy,n_symbols=n_symbols,tau=tau,lx,ly)
    if two_sided==True:
        ip_xp,ip_yp,ip_yf,ip_ypf,ip_xyp,ip_xypf,iphi_x,iphi_yp,iphi_yf=full_get_prob(Sy,Sx,n_symbols=n_symbols,tau=tau,lx,ly)

    #calculate CaMI and TE X->Y
    cami_xy=0;
    pcami_xy=np.zeros([n_symbols^lx,n_symbols^lx,n_symbols^(ly-lx)])
    pte_xy=np.zeros([n_symbols^lx,n_symbols^lx,n_symbols^(ly-lx)])
    for i in range(n_symbols^lx):
        for j in range(n_symbols^lx):
            for k in range(n_symbols^(ly-lx)):
                if (p_xp[i]*p_ypf[j,k]>1e-14) and (p_xypf[i,j,k]>1e-14):
                    if units=='nat':
                        pcami_xy[i,j,k]=np.log(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    elif units=='ban':
                        pcami_xy[i,j,k]=np.log10(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log10((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    else:
                        pcami_xy[i,j,k]=np.log2(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log2((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    cami_xy=cami_xy+pcami_xy[i,j,k];
                else:
                    pcami_xy[i,j,k]=0
                    pte_xy[i,j,k]=0
    #calculate CaMI Y->X
    if two_sided==True:
        pcami_yx=np.zeros([n_symbols^lx,n_symbols^lx,n_symbols^(ly-lx)])
        pte_yx=np.zeros([n_symbols^lx,n_symbols^lx,n_symbols^(ly-lx)])
        cami_yx=0;
        for i in range(ns^lx):
            for j in range(ns^lx):
                for k in range(ns^(ly-lx)):
                    if (ip_x[i]*ip_ypf[j,k]>1e-14) and (ip_xypf[i,j,k]>1e-14):
                        if units=='nat':
                            pcami_yx[i,j,k]=np.log(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        elif units=='ban':
                            pcami_yx[i,j,k]=np.log10(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log10((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        else:
                            pcami_yx[i,j,k]=np.log2(ip_xypf[i,j,k]/(ip_x[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log2((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        cami_yx=cami_yx+pcami_yx[i,j,k]
                    else:
                        pcami_yx[i,j,k]=0
                        pte_yx[i,j,k]=0
    #calculate Mutual Information
    mutual_info=0;
    pmi=np.zeros([n_symbols^lx,n_symbols^lx])
    for i in range(n_symbols^lx):
        for j in range(n_symbols^lx):
            if (p_xp[i]*p_yp[j]>1e-14) and (p_xyp[i,j]>1e-14):
                if units=='nat':
                    pmi[i,j]=p_xyp[i,j]*log(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                elif units=='ban':
                    pmi[i,j]=p_xyp[i,j]*log10(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                else:
                    pmi[i,j]=p_xyp[i,j]*log2(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                mutual_info=mutual_info+pmi[i,j]
            else:
                pmi[i,j]=0
    #calculate TE X->Y
    te_xy=cami_xy-mutual_info
    #calculate TE Y->X
    if two_sided==True:
        te_yx=cami_yx-mutual_info

    #print overall information measures:
    print('Information values:')
    if two_sided==True:
        print('Mutual Information: ', mutual_info,
              '\nCausal Mutual Information X->Y: ', cami_xy,
              '\nCausal Mutual Information Y->X: ', cami_yx,
              '\n Transfer Entropy X->Y: ', te_xy,
              '\n Transfer Entropy Y->X: ', te_yx,
              '\n Directionality Index: ', cami_xy-cami_yx)
    else:
        print('Mutual Information: ', mutual_info,
              '\nCausal Mutual Information X->Y: ', cami_xy,
              '\n Transfer Entropy X->Y: ', te_xy)
    print('--------\nGenerating the time-series of pointwise information measures')

    #passing the results to time-series
    point_cami_xy=np.nan(len(x))
    point_mi=np.nan(len(x))
    point_te_xy=np.nan(len(x))
    if two_sided==True:
        point_cami_yx=np.nan(len(x))
        point_te_yx(len(x))
    for i in range(len(x)-lx):
        pos=i+lx
        if phi_x[pos]!=np.nan and phi_yp[pos]!=np.nan and phi_yf[pos]!=np.nan:
            point_cami_xy[i]=pcami_xy[phi_x[pos],phi_yp[pos],phi_yf[pos]]
            point_mi[i]=pmi[phi_x[pos],phi_yp[pos]]
            point_te_xy[i]=pte_xy[phi_x[pos],phi_yp[pos],phi_yf[pos]]
    if two_sided==True:
        for i in range(len(x)-lx):
              pos=i+lx
           if iphi_x[pos]!=np.nan and iphi_yp[pos]!=np.nan and iphi_yf[pos]!=np.nan: 
                point_cami_yx[i]=pcami_yx[iphi_x[pos],iphi_yp[pos],iphi_yf[pos]]
                point_te_yx[i]=pte_yx[iphi_x[pos],iphi_yp[pos],iphi_yf[pos]]    
        point_di=point_cami_xy-point_cami_yx
    print('done!')
    
    #return results
    if two_sided==True:
        return point_mi,point_cami_xy,point_te_xy,point_cami_yx,point_te_yx,point_di
    else:
        return point_mi,point_cami_xy,point_te_xy
