
import numpy as np

def Jacobian(func,y,fev=None, args=()):
    '''
    Forward differentiation formula to estimate the jacobian.
    func: R^n --> R^n
    '''    
    fe_tot = 0
    if(fev is None):
        fev=func(y,args)
        fe_tot +=1
    
    n=len(y)
    J= np.zeros((n,n))
    # The step is the machine precision * 10^2
    # It is problem dependent... (add as ~tolerance)
    # Even better, copy how h is chosen by scipy/integrate/odepack/prja.f (lines 80-100)
    # Explained in section 3.4.5 of Description and Use of LSODE, the Livermore Solver for ODEs
    h = np.finfo(np.dtype(type(y[0]))).eps*1e2
    for i in range(n):
        yi = y[i]
        y[i] += h
        J[:,i] = 1/h*(func(y,args)-fev)
        y[i]=yi
        fe_tot +=1
        
    return (J,fe_tot)