import numpy as np
from compare_test import nbod_func, kdv_init, kdv_func, burgers_init, burgers_func
import time
from scipy.integrate import ode, complex_ode
import ex_parallel as ex_p
import fnbod


# reference for nbod problem  
def nbod_dense_reference(num):
    t0 = 0
    tf = 0.08
    y0 = fnbod.init_fnbod(2400)
    t = np.linspace(t0, tf, num=num, endpoint=True)
    tol = 1.e-15
    nsteps = 10.e5
    print 'computing reference solution to nbod problem (dense)'
    start_time = time.time()
    y = ex_p.ex_midpoint_explicit_parallel(nbod_func, None, y0, t, atol=tol, rtol=tol, mxstep=nsteps)
    y = y[1:]

    np.savetxt("reference_nbod_dense.txt", y.real)
    print 'reference solution saved in reference_nbod_dense.txt , it took', time.time() - start_time, ' s'
    return y

# reference for kdv problem  
def kdv_solout_dense(U_hat, ts):
    y = np.zeros_like(U_hat)
    N = 256
    for i in range(len(ts)): 
        t = ts[i]
        k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
        E = np.exp(1j * k**3 * t)
        y[i] = np.squeeze(np.real(np.fft.ifft(E*U_hat[i])))
    return y

def kdv_dense_reference(num):
    t0 = 0
    tf = 0.003
    t = np.linspace(t0, tf, num=num, endpoint=True)
    y0 = kdv_init(t0)
    tol = 1.e-15
    nsteps = 10.e5
    print 'computing reference solution to kdv problem (dense)'
    start_time = time.time()
    y = ex_p.ex_midpoint_explicit_parallel(kdv_func, None, y0, t, atol=tol, rtol=tol, mxstep=nsteps)
    y = kdv_solout_dense(y[1:], t[1:])

    np.savetxt("reference_kdv_dense.txt", y.real)
    print 'reference solution saved in reference_kdv_dense.txt , it took', time.time() - start_time, ' s'
    return y


# reference for burgers problem  
def burgers_solout_dense(U_hat, ts):
    y = np.zeros_like(U_hat)
    epslison = 0.1
    N = 64
    for i in range(len(ts)): 
        t = ts[i]
        k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
        E_ = np.exp(-epslison * k**2 * t)
        y[i] = np.squeeze(np.real(np.fft.ifft(E_*U_hat[i])))
    return y

def burgers_dense_reference(num):
    t0 = 0
    tf = 3.
    t = np.linspace(t0, tf, num=num, endpoint=True)
    y0 = burgers_init(t0)
    tol = 1.e-10
    nsteps = 10.e4
    print 'computing reference solution to burgers problem (dense)'
    start_time = time.time()
    y = ex_p.ex_midpoint_explicit_parallel(burgers_func, None, y0, t, atol=tol, rtol=tol, mxstep=nsteps)
    y = burgers_solout_dense(y[1:], t[1:])
    
    np.savetxt("reference_burgers_dense.txt", y.real)
    print 'reference solution saved in reference_burgers_dense.txt , it took', time.time() - start_time, ' s'	
    return y    

if __name__ == "__main__":
    num = 50
    nbod_dense_reference(num)
    kdv_dense_reference(num)
    burgers_dense_reference(num) 
