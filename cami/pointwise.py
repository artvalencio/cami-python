def pointwise(x,y,method='normalized',symbolic_type='equal-divs',n_symbols=2,symbolic_length=1,tau=None,units='bits',two_sided=False,make_plot=True):
    ''' Calculates the Pointwise information
        measures between two variables given
        their observable time-series.

    Parameters
    ----------
    x: list, tuple, np.array, pd.Series
        The first time-series
    y: list, tuple, np.array, pd.Series
        The second time-series
    method: str, optional
        Whether to calculate the (regular) pointwise information measures
        or the normalized information measures (default). Options:
            - 'normalized' or 'normalised' (default): calculates the
                pointwise information measures normalized by -1/log(joint_prob)
            - 'raw' or 'regular' or 'non-normalized' or 'non-normalised': does not
                perfom any normalization procedure
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
    tau: int, None, optional
        Time-delay of reconstruction via method of embedding (Takens'), in number
        of steps. default=None (first zero correlation crossing).
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bit' or 'bits': log2 is adopted (default)
            - 'nat' or 'nats': ln is adopted
            - 'ban' or 'bans': log10 is adopted
    two_sided: bool, optional
        Whether to calculate in the directions X->Y and Y->X or only on X->Y.
        Default: False (only X->Y)
    make_plot: bool,optional
        Whether to make scatter plot with the results or not. Default: True.
    
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
        symbolic_type='equal-points',n_symbols=10,symbolic_length=1,two_sided=True,make_plot=True)
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
    x,y=pd.to_numeric(x,errors='coerce'),pd.to_numeric(y,errors='coerce')
    while np.isnan(x[0]) or np.isnan(y[0]):
        x,y=x[1:],y[1:]
    while np.isnan(x[-1]) or np.isnan(y[-1]):
        x,y=x[:-1],y[:-1]
    def interp_func(data):
        idx_bads=np.isnan(data)
        idx_goods=np.logical_not(idx_bads)
        data_good=data[idx_goods]
        interp_data=np.interp(idx_bads.nonzero()[0],idx_goods.nonzero()[0],data_good)
        data[idx_bads]=interp_data
        return data
    x,y=interp_func(x),interp_func(y)
    #convert to symbolic sequence
    Sx,Sy=cami.symbolic_encoding(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols)
    #calculate tau
    if tau==None:
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
        tau=max(get_tau(x),get_tau(y))
        print('Selected tau=',tau,' by the method of first zero of auto-correlation',sep='')
            
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
    
    #get box names and probabilities
    def full_get_prob(Sx,Sy,lx,lyp,lyf,n_symbols=2,tau=1):
        tslen=len(Sx)
        #initializing boxes
        phi_x=np.full(tslen,np.nan)
        phi_yp=np.full(tslen,np.nan)
        phi_yf=np.full(tslen,np.nan)        
        #initializing probabilities of boxes
        #note: n_points=tslen-tau*ly-1;
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
        return p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf
    
    p_xp,p_yp,p_yf,p_ypf,p_xyp,p_xypf,phi_x,phi_yp,phi_yf=full_get_prob(Sx,Sy,lx,lyp,lyf,n_symbols=n_symbols,tau=tau)
    if two_sided==True:
        ip_xp,ip_yp,ip_yf,ip_ypf,ip_xyp,ip_xypf,iphi_x,iphi_yp,iphi_yf=full_get_prob(Sy,Sx,lx,lyp,lyf,n_symbols=n_symbols,tau=tau)

    

    #calculate CaMI X->Y
    cami_xy=0
    pcami_xy=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
    pte_xy=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**lyp):
            for k in range(n_symbols**lyf):
                if (p_xp[i]*p_ypf[j,k]>0) and (p_xypf[i,j,k]>0):
                    if units=='nat' or units=='nats':
                        pcami_xy[i,j,k]=np.log(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    elif units=='ban' or units=='bans':
                        pcami_xy[i,j,k]=np.log10(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log10((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    else:
                        pcami_xy[i,j,k]=np.log2(p_xypf[i,j,k]/(p_xp[i]*p_ypf[j,k]))
                        pte_xy[i,j,k]=np.log2((p_xypf[i,j,k]*p_yp[j])/(p_xyp[i,j]*p_ypf[j,k]))
                    cami_xy=cami_xy+p_xypf[i,j,k]*pcami_xy[i,j,k]
                    if not(method=='raw' or method=='regular' or method=='non-normalized' or method=='non-normalised'):
                        if units=='nat' or units=='nats':
                            pcami_xy[i,j,k]=-pcami_xy[i,j,k]/np.log(p_xypf[i,j,k])
                            pte_xy[i,j,k]=-pte_xy[i,j,k]/np.log(p_xypf[i,j,k])
                        elif units=='ban' or units=='bans':
                            pcami_xy[i,j,k]=-pcami_xy[i,j,k]/np.log10(p_xypf[i,j,k])
                            pte_xy[i,j,k]=-pte_xy[i,j,k]/np.log10(p_xypf[i,j,k])
                        else:
                            pcami_xy[i,j,k]=-pcami_xy[i,j,k]/np.log2(p_xypf[i,j,k])
                            pte_xy[i,j,k]=-pte_xy[i,j,k]/np.log2(p_xypf[i,j,k])
                else:
                    pcami_xy[i,j,k]=0
    #calculate CaMI Y->X
    if two_sided==True:
        pcami_yx=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
        pte_yx=np.zeros([n_symbols**lx,n_symbols**lyp,n_symbols**lyf])
        cami_yx=0
        for i in range(n_symbols**lx):
            for j in range(n_symbols**lyp):
                for k in range(n_symbols**lyf):
                    if (ip_xp[i]*ip_ypf[j,k]>0) and (ip_xypf[i,j,k]>0):
                        if units=='nat' or units=='nats':
                            pcami_yx[i,j,k]=np.log(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        elif units=='ban' or units=='bans':
                            pcami_yx[i,j,k]=np.log10(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log10((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        else:
                            pcami_yx[i,j,k]=np.log2(ip_xypf[i,j,k]/(ip_xp[i]*ip_ypf[j,k]))
                            pte_yx[i,j,k]=np.log2((ip_xypf[i,j,k]*ip_yp[j])/(ip_xyp[i,j]*ip_ypf[j,k]))
                        cami_yx=cami_yx+ip_xypf[i,j,k]*pcami_yx[i,j,k]
                        if not(method=='raw' or method=='regular' or method=='non-normalized' or method=='non-normalised'):
                            if units=='nat' or units=='nats':
                                pcami_yx[i,j,k]=-pcami_yx[i,j,k]/np.log(ip_xypf[i,j,k])
                                pte_yx[i,j,k]=-pte_yx[i,j,k]/np.log(ip_xypf[i,j,k])
                            elif units=='ban' or units=='bans':
                                pcami_yx[i,j,k]=-pcami_yx[i,j,k]/np.log10(ip_xypf[i,j,k])
                                pte_yx[i,j,k]=-pte_yx[i,j,k]/np.log10(ip_xypf[i,j,k])
                            else:
                                pcami_yx[i,j,k]=-pcami_yx[i,j,k]/np.log2(ip_xypf[i,j,k])
                                pte_yx[i,j,k]=-pte_yx[i,j,k]/np.log2(ip_xypf[i,j,k])
                    else:
                        pcami_yx[i,j,k]=0
    #calculate Mutual Information
    mutual_info=0
    pmi=np.zeros([n_symbols**lx,n_symbols**lyp])
    for i in range(n_symbols**lx):
        for j in range(n_symbols**lyp):
            if (p_xp[i]*p_yp[j]>0) and (p_xyp[i,j]>0):
                if units=='nat' or units=='nats':
                    pmi[i,j]=np.log(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                elif units=='ban' or units=='bans':
                    pmi[i,j]=np.log10(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                else:
                    pmi[i,j]=np.log2(p_xyp[i,j]/(p_xp[i]*p_yp[j]))
                mutual_info=mutual_info+p_xyp[i,j]*pmi[i,j]
                if not(method=='raw' or method=='regular' or method=='non-normalized' or method=='non-normalised'):
                    if units=='nat' or units=='nats':
                        pmi[i,j]=-pmi[i,j]/np.log(p_xyp[i,j])
                    elif units=='ban' or units=='bans':
                        pmi[i,j]=-pmi[i,j]/np.log10(p_xyp[i,j])
                    else:
                        pmi[i,j]=-pmi[i,j]/np.log2(p_xyp[i,j])
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
    point_cami_xy=np.full(len(x),np.nan)
    point_mi=np.full(len(x),np.nan)
    point_te_xy=np.full(len(x),np.nan)
    if two_sided==True:
        point_cami_yx=np.full(len(x),np.nan)
        point_te_yx=np.full(len(x),np.nan)
    lmax=max([lx,lyp,lyf])
    for i in range(len(x)-lmax):
        if not (np.isnan(phi_x[i+lx]) or np.isnan(phi_yp[i+lyp]) or np.isnan(phi_yf[i+lyf])):
            point_cami_xy[i]=pcami_xy[int(phi_x[i+lx]),int(phi_yp[i+lyp]),int(phi_yf[i+lyf])]
            point_mi[i]=pmi[int(phi_x[i+lx]),int(phi_yp[i+lyp])]
            point_te_xy[i]=pte_xy[int(phi_x[i+lx]),int(phi_yp[i+lyp]),int(phi_yf[i+lyf])]
    if two_sided==True:
        for i in range(len(x)-lmax):
            if not(np.isnan(iphi_x[i+lx]) or np.isnan(iphi_yp[i+lyp]) or np.isnan(iphi_yf[i+lyf])): 
                point_cami_yx[i]=pcami_yx[int(iphi_x[i+lx]),int(iphi_yp[i+lyp]),int(iphi_yf[i+lyf])]
                point_te_yx[i]=pte_yx[int(iphi_x[i+lx]),int(iphi_yp[i+lyp]),int(iphi_yf[i+lyf])]    
        point_di=point_cami_xy-point_cami_yx
    print('done!')

    #plotting
    if make_plot==True:
        import matplotlib.pyplot as plt
        if two_sided==True:
            fig,ax=plt.subplots(3,2)
            im1=ax[0,0].scatter(x[:len(point_cami_xy)],y[:len(point_cami_xy)],c=point_cami_xy,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_cami_xy)),abs(max(point_cami_xy))),vmax=max(abs(min(point_cami_xy)),abs(max(point_cami_xy))))
            ax[0,0].set_title(r'$CaMI_{X\rightarrow Y}$')
            fig.colorbar(im1,ax=ax[0,0])
            im2=ax[0,1].scatter(x[:len(point_cami_yx)],y[:len(point_cami_yx)],c=point_cami_yx,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_cami_yx)),abs(max(point_cami_yx))),vmax=max(abs(min(point_cami_yx)),abs(max(point_cami_yx))))
            ax[0,1].set_title(r'$CaMI_{Y\rightarrow X}$')
            fig.colorbar(im2,ax=ax[0,1])
            im3=ax[1,0].scatter(x[:len(point_te_xy)],y[:len(point_te_xy)],c=point_te_xy,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_te_xy)),abs(max(point_te_xy))),vmax=max(abs(min(point_te_xy)),abs(max(point_te_xy))))
            ax[1,0].set_title(r'$TE_{X\rightarrow Y}$')
            fig.colorbar(im3,ax=ax[1,0])
            im4=ax[1,1].scatter(x[:len(point_te_yx)],y[:len(point_te_yx)],c=point_te_yx,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_te_yx)),abs(max(point_te_yx))),vmax=max(abs(min(point_te_yx)),abs(max(point_te_yx))))
            ax[1,1].set_title(r'$TE_{Y\rightarrow X}$')
            fig.colorbar(im4,ax=ax[1,1])
            im5=ax[2,0].scatter(x[:len(point_mi)],y[:len(point_mi)],c=point_mi,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_mi)),abs(max(point_mi))),vmax=max(abs(min(point_mi)),abs(max(point_mi))))
            ax[2,0].set_title(r'Mutual Information')
            fig.colorbar(im5,ax=ax[2,0])
            im6=ax[2,1].scatter(x[:len(point_di)],y[:len(point_di)],c=point_di,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_di)),abs(max(point_di))),vmax=max(abs(min(point_di)),abs(max(point_di))))
            ax[2,1].set_title(r'Directionality Index')
            fig.colorbar(im6,ax=ax[2,1])
        else:
            fig,ax=plt.subplots(3,1)
            im1=ax[0,0].scatter(x[:len(point_cami_xy)],y[:len(point_cami_xy)],c=point_cami_xy,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_cami_xy)),abs(max(point_cami_xy))),vmax=max(abs(min(point_cami_xy)),abs(max(point_cami_xy))))
            ax[0,0].set_title(r'$CaMI_{X\rightarrow Y}$')
            fig.colorbar(im1,ax=ax[0,0])
            im2=ax[1,0].scatter(x[:len(point_te_xy)],y[:len(point_te_xy)],c=point_te_xy,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_te_xy)),abs(max(point_te_xy))),vmax=max(abs(min(point_te_xy)),abs(max(point_te_xy))))
            ax[1,0].set_title(r'$TE_{X\rightarrow Y}$')
            fig.colorbar(im2,ax=ax[1,0])
            im3=ax[2,0].scatter(x[:len(point_mi)],y[:len(point_mi)],c=point_mi,cmap='bwr',marker='.',linewidths=0,edgecolors='none',alpha=0.5,vmin=-max(abs(min(point_mi)),abs(max(point_mi))),vmax=max(abs(min(point_mi)),abs(max(point_mi))))
            ax[2,0].set_title(r'Mutual Information')
            fig.colorbar(im3,ax=ax[2,0])
        fig.tight_layout()
        plt.show()
    
    #return results
    if two_sided==True:
        return point_mi,point_cami_xy,point_te_xy,point_cami_yx,point_te_yx,point_di
    else:
        return point_mi,point_cami_xy,point_te_xy
