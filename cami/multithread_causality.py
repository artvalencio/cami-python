def multithread_causality(x,y,axis=1,symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,delay=0,units='bits',many_nans=False,two_sided=False):
    ''' Calculates Information Theory
        values based on many independent
        observation of two variables
        

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first variable time-series, where each
        independent observation is given as a column
    y: list, tuple, np.array, pd.Series
        The second variable time-series, where each
        independent observation is given as a column
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
    tau: int, str, None, optional
        Time-delay of reconstruction via method of embedding (Takens'), in number
        of steps. default=None (average of first zero correlation crossing in all trials).
        str options:
            - 'average' or 'mean' or None: average of first zero correlation crossing in all trials
            - 'maximum' or max: maximum of first zero correlation crossing in all trials
            - 'median' or 'med': median of first zero correlation crossing in all trials
    delay: int, optional
        Time-delay to be considered between cause and effect, in number of steps.
        Default: zero
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bit' or 'bits': log2 is adopted (default)
            - 'nat' or 'nats': ln is adopted
            - 'ban' or 'bans': log10 is adopted
    many_nans: bool, optional
        Whether the input data has too many nans or not. Default: False.
    two_sided: bool, optional
        Whether to calculate the quantitites in X->Y and Y->X directions
        or only on X->Y. Default: False (only X->Y)
    
    Returns
    -------
    te_xy: float
        The value of the Transfer Entropy
        in the direction of X->Y
    te_yx: float (if two_sided=True)
        The value of the Transfer Entropy
        in the direction of Y->X 

    See Also
    --------
    mutual_info: calculates the Mutual Information from
        time-series data
    cami: calculates the Causal Mutual Information from
        time-series data
    tranfer_entropy_rate: calculates the Transfer Entropy
        Rate with respect to the symbolic length
    directionality: calculates the net flow of causal
        information between variables X and Y

    Example
    -------
    mi,cami_xy,te_xy,cami_yx,te_yx = multithread_causality(x,y,symbolic_type='equal-points',n_symbols=10,
        symbolic_length=1,two_sided=True)
    '''

    import numpy as np
    import pandas as pd
    import cami
    #checking units
    if units=='bit' or units=='bits' or units=='nat' or units=='nats' or units=='ban' or units=='bans':
        pass
    else:
        raise ValueError('Units must be bits or nat or ban. See help on function.')
    #interpolate missing data or trim if its in the edges
    x,y=pd.DataFrame(x),pd.DataFrame(y)
    x[:],y[:]=x[:].apply(pd.to_numeric,errors='coerce',axis=1),y[:].apply(pd.to_numeric,errors='coerce',axis=1)
    x,y=x.to_numpy(dtype=np.float64),y.to_numpy(dtype=np.float64)
    #for data with many nans it is advisable to pad the start/finish with the first/last value, otherwise you will chop off too much data across threads
    #if you only have few nans (usual), the preferrable way is to chop down these nans at the edges, as this way captures best the true dynamics
    if not many_nans:
        for i in range(len(x[0,:])):
            while np.isnan(x[0,i]) or np.isnan(y[0,i]):
                x[:-1,i],y[:-1,i]=x[1:,i],y[1:,i]
                x[-1,i],y[-1,i]=np.nan,np.nan
        for i in range(len(x[0,:])):
            while np.isnan(x[-1,i]) or np.isnan(y[-1,i]):
                x,y=x[:-1,:],y[:-1,:]
    def interp_func(data):
        idx_bads=np.isnan(data)
        idx_goods=np.logical_not(idx_bads)
        data_good=data[idx_goods]
        interp_data=np.interp(idx_bads.nonzero()[0],idx_goods.nonzero()[0],data_good)
        data[idx_bads]=interp_data
        return data
    for i in range(len(x[0,:])):
        x[:,i],y[:,i]=interp_func(x[:,i]),interp_func(y[:,i])
    if delay>0:
        x,y=x[:-delay,:],y[delay:,:]
    elif delay<0:
        x,y=x[delay:,:],y[:-delay,:]

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
        tau=int(np.mean([max(get_tau(x[:,thread]),get_tau(y[:,thread])) for thread in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, average of all trials',sep='')
    elif tau=='max' or tau=='maximum':
        tau=int(max([max(get_tau(x[:,thread]),get_tau(y[:,thread])) for thread in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, maximum of all trials',sep='')
    elif tau=='median' or tau=='med':
        tau=int(np.median([max(get_tau(x[:,thread]),get_tau(y[:,thread])) for thread in range(len(x[0,:]))]))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation, median of all trials',sep='')

    #convert to symbolic sequence
    def multithread_symbolic_encoding(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols):
        tslen=len(x[:,0])
        nthreads=len(x[0,:])
        #generating partitions (checking consistency)
        if symbolic_type=='equal-divs':
            xmin,xmax,ymin,ymax = x.min(),x.max(),y.min(),y.max()
            xpart,ypart=[],[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
                ypart.append(ymin+i*(ymax-ymin)/n_symbols)
        elif symbolic_type=='equal-points':
            xtemp,ytemp=x.copy(),y.copy()
            xtemp=xtemp.reshape(-1)
            ytemp=ytemp.reshape(-1)
            xsort,ysort = np.sort(xtemp),np.sort(ytemp)
            xpart,ypart = [],[]
            for i in range(1,n_symbols):
                xpart.append(xsort[int(i*tslen/n_symbols)])
                ypart.append(ysort[int(i*tslen/n_symbols)])
        elif symbolic_type=='equal-growth':
            xdiff,ydiff = np.diff(x,axis=0),np.diff(y,axis=0)
            xmin,xmax,ymin,ymax = xdiff.min(),xdiff.max(),ydiff.min(),ydiff.max()
            xpart,ypart=[],[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
                ypart.append(ymin+i*(ymax-ymin)/n_symbols)
        elif symbolic_type=='equal-growth-points':
            xdiffsort,ydiffsort = np.sort(np.diff(x,axis=0).reshape(-1)),np.sort(np.diff(y,axis=0).reshape(-1))
            xpart,ypart = [],[]
            for i in range(1,n_symbols):
                xpart.append(xdiffsort[int(i*(tslen-1)/n_symbols)])
                ypart.append(ydiffsort[int(i*(tslen-1)/n_symbols)])
        elif symbolic_type=='equal-concavity':
            xdiff2,ydiff2 = np.diff(np.diff(x,axis=0),axis=0),np.diff(np.diff(y,axis=0),axis=0)
            xmin,xmax,ymin,ymax = xdiff2.min(),xdiff2.max(),ydiff2.min(),ydiff2.max()
            xpart,ypart=[],[]
            for i in range(1,n_symbols):
                xpart.append(xmin+i*(xmax-xmin)/n_symbols)
                ypart.append(ymin+i*(ymax-ymin)/n_symbols)
        elif symbolic_type=='equal-concavity-points':
            xdiff2sort,ydiff2sort = np.sort(np.diff(np.diff(x,axis=0),axis=0).reshape(-1)),np.sort(np.diff(np.diff(y,axis=0),axis=0).reshape(-1))
            xpart,ypart = [],[]
            for i in range(1,n_symbols):
                xpart.append(xdiff2sort[int(i*(tslen-2)/n_symbols)])
                ypart.append(ydiff2sort[int(i*(tslen-2)/n_symbols)])
        else:
            raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.')
        #Generating the symbolic sequences
        def getsequence(x,y,xpart,ypart):
            Sx=np.full_like(x,-1)
            Sy=np.full_like(y,-1)
            for thread in range(len(x[0,:])):
                for n in range(len(x[:,thread])): #assign data points to partition symbols in x
                    for i in range(len(xpart)):
                        if x[n,thread]<xpart[i]:
                            Sx[n,thread]=i
                            break
                    if Sx[n,thread]==-1:
                        Sx[n,thread]=n_symbols-1
            for thread in range(len(y[0,:])):
                for n in range(len(y[:,thread])): #assign data points to partition symbols in y
                    for i in range(len(ypart)):
                        if y[n,thread]<ypart[i]:
                            Sy[n,thread]=i
                            break
                    if Sy[n,thread]==-1:
                        Sy[n,thread]=n_symbols-1
            return Sx,Sy      
        if symbolic_type=='equal-divs' or symbolic_type=='equal-points':
            Sx,Sy = getsequence(x,y,xpart,ypart)
        elif symbolic_type=='equal-growth':
            Sx,Sy = getsequence(xdiff,ydiff,xpart,ypart)
        elif symbolic_type=='equal-concavity':
            Sx,Sy = getsequence(np.diff(x),np.diff(y),xpart,ypart)
        elif symbolic_type=='equal-concavity':
            Sx,Sy = getsequence(xdiff2,ydiff2,xpart,ypart)
        elif symbolic_type=='equal-concavity-points':
            Sx,Sy = getsequence(np.diff(np.diff(x)),np.diff(np.diff(y)),xpart,ypart)
        #Returning result
        return Sx,Sy

    Sx,Sy=multithread_symbolic_encoding(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols)      

    #get box probabilities
    #getting symbolic lengths
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
    #get box names and probabilities
    def multithread_get_prob(Sx,Sy,lx,lyp,lyf,n_symbols=2,tau=1):
        nthreads=len(Sx[0,:])
        tslen=len(Sx[:,0])
        #initializing boxes
        phi_x=np.full([tslen,nthreads],np.nan)
        phi_yp=np.full([tslen,nthreads],np.nan)
        phi_yf=np.full([tslen,nthreads],np.nan)
        #initializing probabilities of boxes
        p_xp=np.zeros(n_symbols**lx)
        p_yp=np.zeros(n_symbols**lyp)
        p_yf=np.zeros(n_symbols**lyf)
        p_ypf=np.zeros([n_symbols**lyp,n_symbols**lyf])
        p_xyp=np.zeros([n_symbols**lx,n_symbols**lyp])
        p_xypf=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
        #calculating phi_x, about the past of x
        for thread in range(nthreads):
            for n in range(tau*lx,tslen):
                phi_x[n,thread]=0
                k=0
                for i in range(n-tau*lx,n,tau):
                    phi_x[n,thread]=phi_x[n,thread]+Sx[i,thread]*n_symbols**(k)#phi is the partition box name of the sequence: e.g.: (|0|..tau..|1|..tau..|0|) => box phi=2
                    k=k+1
                p_xp[int(phi_x[n,thread])]=p_xp[int(phi_x[n,thread])]+1
        p_xp=p_xp/p_xp.sum()
        #calculating phi_yp, about the past of y
        for thread in range(nthreads):
            for n in range(tau*lyp,tslen):
                phi_yp[n,thread]=0
                k=0
                for i in range(n-tau*lyp,n,tau):
                    phi_yp[n,thread]=phi_yp[n,thread]+Sy[i,thread]*n_symbols**(k)
                    k=k+1
                p_yp[int(phi_yp[n,thread])]=p_yp[int(phi_yp[n,thread])]+1
        p_yp=p_yp/p_yp.sum()
        #calculating phi_yf, about the future of y
        for thread in range(nthreads):
            for n in range(0,tslen-tau*lyf):
                phi_yf[n,thread]=0
                k=0
                for i in range(n,n+tau*lyf,tau):
                    phi_yf[n,thread]=phi_yf[n,thread]+Sy[i,thread]*n_symbols**(k)
                    k=k+1
                p_yf[int(phi_yf[n,thread])]=p_yf[int(phi_yf[n,thread])]+1
        p_yf=p_yf/p_yf.sum()
        #calculating joint probabilities
        for thread in range(nthreads):
            for n in range(tslen):
                if not(np.isnan(phi_x[n,thread]) or np.isnan(phi_yp[n,thread]) or np.isnan(phi_yf[n,thread])):
                    p_ypf[int(phi_yp[n,thread]),int(phi_yf[n,thread])]=p_ypf[int(phi_yp[n,thread]),int(phi_yf[n,thread])]+1
                    p_xyp[int(phi_x[n,thread]),int(phi_yp[n,thread])]=p_xyp[int(phi_x[n,thread]),int(phi_yp[n,thread])]+1
                    p_xypf[int(phi_x[n,thread]),int(phi_yp[n,thread]),int(phi_yf[n,thread])]=p_xypf[int(phi_x[n,thread]),int(phi_yp[n,thread]),int(phi_yf[n,thread])]+1
        p_ypf=p_ypf/p_ypf.sum()
        p_xyp=p_xyp/p_xyp.sum()
        p_xypf=p_xypf/p_xypf.sum()
        return p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf
    
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf=multithread_get_prob(Sx,Sy,lx,lyp,lyf,n_symbols=n_symbols,tau=tau)
    if two_sided==True:
        ip_xp,ip_yp,ip_yf,ip_ypf,ip_xyp,ip_xypf,__,__,__=multithread_get_prob(Sy,Sx,lx,lyp,lyf,n_symbols=n_symbols,tau=tau)
        
    #calculate CaMI X->Y
    cami_xy=0
    pcami_xy=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**lyp):
            for k in range(n_symbols**lyf):
                if (p_xp[i]*p_ypf[j,k]>0) and (p_xypf[i,j,k]>0):
                    if units=='nat' or units=='nats':
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    elif units=='ban' or units=='bans':
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log10(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    else:
                        pcami_xy[i,j,k]=p_xypf[i,j,k]*np.log2(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                    cami_xy=cami_xy+pcami_xy[i,j,k]
                else:
                    pcami_xy[i,j,k]=0
    #calculate CaMI Y->X
    if two_sided==True:
        pcami_yx=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
        cami_yx=0
        for i in range(n_symbols**lx):
            for j in range(n_symbols**lyp):
                for k in range(n_symbols**lyf):
                    if (ip_xp[i]*ip_ypf[j,k]>0) and (ip_xypf[i,j,k]>0):
                        if units=='nat' or units=='nats':
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                        elif units=='ban' or units=='bans':
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log10(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                        else:
                            pcami_yx[i,j,k]=ip_xypf[i,j,k]*np.log2(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                        cami_yx=cami_yx+pcami_yx[i,j,k]
                    else:
                        pcami_yx[i,j,k]=0
    #calculate Mutual Information
    mi=0
    pmi=np.zeros([n_symbols**lx,n_symbols**lyp])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**lyp):
            if (p_xp[i]*p_yp[j]>0) and (p_xyp[i,j]>0):
                if units=='nat' or units=='nats':
                    pmi[i,j]=p_xyp[i,j]*np.log(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                elif units=='ban' or units=='bans':
                    pmi[i,j]=p_xyp[i,j]*np.log10(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                else:
                    pmi[i,j]=p_xyp[i,j]*np.log2(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                mi=mi+pmi[i,j]
            else:
                pmi[i,j]=0
    #calculate TE X->Y
    te_xy=cami_xy-mi
    #calculate TE Y->X
    if two_sided==True:
        te_yx=cami_yx-mi
        
    #return results
    if two_sided==True:
        return mi,cami_xy,te_xy,cami_yx,te_yx
    else:
        return mi,cami_xy,te_xy
