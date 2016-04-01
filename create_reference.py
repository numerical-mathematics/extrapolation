import numpy as np
from compare_test import kdv_init, kdv_func, kdv_solout, burgers_init, burgers_func, burgers_solout
import time
from scipy.integrate import ode, complex_ode
import ex_parallel as ex_p

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

# reference for kdv problem  
def kdv_reference():
    t0 = 0
    tf = 0.003
    y0 = kdv_init(t0)
    tol = 1.e-15
    nsteps = 10.e5
    def func2(t,y):
        return kdv_func(y,t)
    print 'computing reference solution to kdv problem'
    start_time = time.time()
    y = ex_p.ex_midpoint_explicit_parallel(kdv_func, None, y0, [t0, tf], atol=tol, rtol=tol, mxstep=nsteps)
    y = kdv_solout(y[-1])

    np.savetxt("reference_kdv.txt", y)
    print 'reference solution saved in reference_kdv.txt , it took', time.time() - start_time, ' s'
    return y

# reference for burgers problem  
def burgers_reference():
    t0 = 0
    tf = 3.
    y0 = burgers_init(t0)
    tol = 1.e-10
    nsteps = 10.e4
    def func2(t,y):
        return burgers_func(y,t)
    print 'computing reference solution to burgers problem'
    start_time = time.time()
    y = ex_p.ex_midpoint_explicit_parallel(burgers_func, None, y0, [t0, tf], atol=tol, rtol=tol, mxstep=nsteps)
    y = burgers_solout(y[-1])
    
    np.savetxt("reference_burgers.txt", y)
    print 'reference solution saved in reference_burgers.txt , it took', time.time() - start_time, ' s'	
    return y    

if __name__ == "__main__":
    kdv_reference() 
    burgers_reference() 
