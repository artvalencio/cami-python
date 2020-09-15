def mutual_info_rate(x,y,symbolic_type='equal-divs',n_symbols=2,tau=1,units='bits'):
    ''' Calculates the Mutual Information Rate
        between two variables given their
        observable time-series.

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
        binary (0,1) (default). If n_symbols=3, it is ternary
        (0,1,2), and so on.
    units: str, optional
        Units to be used (base of the logarithm). Options:
            - 'bits': log2 is adopted (default)
            - 'nat': ln is adopted
            - 'ban': log10 is adopted
    
    Returns
    -------
    Prints the value of the Mutual Information Rate on the shell

    See Also
    --------
    mutual_info: calculates the Mutual Information between two
        variables from time-series data
    cami_rate: calculates the Causal Mutual Information Rate
        from time-series data
    transfer_entropy: calculates the Transfer Entropy
        from time-series data
    multithread_causality: calculates information-theory
        measures from many individual short time-series
        of the same phenomena instead of using a single
        long time-series
    pointwise: calculates the pointwise contributions to
        the information measurements
    total_correlation: calculates the ammount of mutual
        of a set of variables

    Example
    -------
    mutual_info_rate(x,y,symbolic_type='equal-points',n_symbols=10)
    '''
    #import libraries
    import matplotlib.pyplot as plt
    from tkinter import Tk,Label,Button,Entry
    import numpy as np
    import pandas as pd
    import cami
    from scipy.stats import linregress

    #calculate mutual info as a function of symbolic length
    mi=np.zeros(10)
    for L in range(1,11):
        mi[L-1]=cami.mutual_info(x,y,symbolic_type=symbolic_type,n_symbols=n_symbols,symbolic_length=L,tau=tau,units=units)

    #plot MI vs L
    plt.ion()
    plt.plot(np.arange(10)+1,mi)
    plt.xlabel('Symbolic length (L)')
    plt.ylabel('Mutual Information [',units,']')
    plt.title('Check for a linear part')
    plt.draw()

    #create dialog box asking where the linear portion begins and ends
    master = Tk()
    label=Label(master,text="Inform the linear part")
    label.pack()
    e1 = Entry(master)
    msg1 = "Type L_min here (int)"
    e1.delete(0, END)
    e1.insert(0, msg1)
    e1.pack()
    e1.focus_set()
    e2 = Entry(master)
    msg2 = "Type L_max here (int)"
    e2.delete(0, END)
    e2.insert(0, msg2)
    e2.pack()
    e2.focus_set()

    #callback to dialog button performs the linear regression which prints mir in turn
    def callback():
        #get the informed Lmin and Lmax
        Lmin=int(e1.get())
        Lmax=int(e2.get())
        #calculates MIR
        mir=linregress(np.arange(Lmax-Lmin+1)+Lmin,mi[Lmin+1:Lmax+1]).slope
        #print results
        print('Mutual Information Rate: ', mir)
        #close figures and dialog box
        plt.close()
        master.destroy()

    #creates the dialog box button and run the dialog box
    b = Button(master, text = "OK", width = 10, command = callback)
    b.pack()
    mainloop()
