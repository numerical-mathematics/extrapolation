import numpy as np
from compare_test import kdv_init, kdv_func, kdv_solout, burgers_init, burgers_func, burgers_solout
import time
from scipy.integrate import ode
from scipy.integrate import complex_ode

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref*len(y_ref))

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
    # r = ode(func2, jac=None).set_integrator('zvode', atol=tol, rtol=tol, method='adams', nsteps=nsteps)
    r = complex_ode(func2).set_integrator('dop853', atol=tol, rtol=tol, verbosity=10, nsteps=nsteps)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    y = kdv_solout(r.y)
    np.savetxt("reference_kdv.txt", y)
    print 'reference solution saved in reference_kdv.txt , it took', time.time() - start_time, ' s'
    return y

# reference for burgers problem  
def burgers_reference():
    t0 = 0
    tf = 0.18
    y0 = burgers_init(t0)
    tol = 1.e-15
    nsteps = 10.e6
    def func2(t,y):
        return burgers_func(y,t)
    print 'computing reference solution to burgers problem'
    start_time = time.time()
    r = ode(func2, jac=None).set_integrator('zvode', atol=tol, rtol=tol, method='adams', nsteps=nsteps)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    y = burgers_solout(r.y)
    np.savetxt("reference_burgers.txt", y)
    print 'reference solution saved in reference_burgers.txt , it took', time.time() - start_time, ' s'	
    return y    

if __name__ == "__main__":
    kdv_reference() 
    # burgers_reference() 
