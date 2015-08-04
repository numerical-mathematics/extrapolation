import numpy as np
from compare_test import KdV_init, KdV_func, burgers_init, burgers_func
import time
from scipy.integrate import ode

def relative_error(y, y_ref):
    return np.linalg.norm((y-y_ref)/y_ref)/(len(y)**0.5)

# reference for KdV problem  
def KdV_reference():
    t0 = 0
    tf = 0.003
    y0 = KdV_init(t0)
    tol = 1.e-15
    nsteps = 10.e5
    def func2(t,y):
        return KdV_func(y,t)
    print 'computing reference solution to KdV problem'
    start_time = time.time()
    r = ode(func2, jac=None).set_integrator('zvode', atol=tol, rtol=tol, method='adams', nsteps=nsteps)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    np.savetxt("reference_KdV.txt", r.y.view('float'))
    print 'reference solution saved in reference_KdV.txt , it took', time.time() - start_time, ' s'

# reference for burgers problem  
def burgers_reference():
    t0 = 0
    tf = 3
    y0 = burgers_init(t0)
    tol = 1.e-11
    nsteps = 10.e5
    def func2(t,y):
        return burgers_func(y,t)
    print 'computing reference solution to burgers problem'
    start_time = time.time()
    r = ode(func2, jac=None).set_integrator('zvode', atol=tol, rtol=tol, method='adams', nsteps=nsteps)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    np.savetxt("reference_burgers.txt", r.y.view('float'))
    print 'reference solution saved in reference_burgers.txt , it took', time.time() - start_time, ' s'	

# KdV_reference() 
# burgers_reference() 
