'''
Runs a dense output performance test comparing ParEx with 
DOPRI5 and DOP853 integrators from scipy.integrate.ode 
Result graphs are saved in the folder ./images

In this file only, we use a version of scipy.integrate.ode due to 
James D. D. Martin (2015) that exposes the dense output option from the 
Fortran code of DOPRI5 and DOP853 integrators. Please install 
this version of scipy first before running this file.

James D. D. Martin version of scipy can be downloaded from 
https://github.com/jddmartin/scipy/tree/dense_output_from_dopri5_and_dop853
Examples of the usage of dense output in scipy are in 
https://github.com/jddmartin/dense_output_example_usage

you essentially need to run these commands: 
$ sudo pip install cython numpy
$ sudo pip install git+https://github.com/jddmartin/scipy.git@dense_output_from_dopri5_and_dop853

if you want to revert back to the most recent version of scipy, run these commands
$ sudo pip uninstall scipy
$ sudo pip install scipy

'''

from __future__ import division
import numpy as np
import time
from scipy.integrate import ode, complex_ode, dense_dop

import ex_parallel as ex_p
import fnbod

class DenseSolout(object):
    def __init__(self, ts):
        self.dense_output = []
        self.tindex = 0
        self.ts = ts

    def solout(self, nr, told, t, y, con):
        if nr == 1:  # initial conditions:
            self.dense_output.append(y)
            self.tindex += 1
        else:  # subsequent step positions (after starting point):
            while self.tindex < len(self.ts) and t >= self.ts[self.tindex]:
                yd = dense_dop(self.ts[self.tindex], told, t, con)
                self.dense_output.append(yd)
                self.tindex += 1

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def compare_performance_dense(func, y0, t, y_ref, problem_name, tol_boundary=(0,6), is_complex=False, nsteps=10e5, solout=(lambda y,t: y)):
    print 'RUNNING COMPARISON TEST FOR ' + problem_name
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]
    t0, tf = t[0], t[-1]

    py_runtime = np.zeros(len(tol))
    py_fe_seq = np.zeros(len(tol))
    py_fe_tot = np.zeros(len(tol))
    py_yerr = np.zeros(len(tol))
    py_nstp = np.zeros(len(tol))

    dopri5_runtime = np.zeros(len(tol))
    dopri5_fe_seq = np.zeros(len(tol))
    dopri5_fe_tot = np.zeros(len(tol))
    dopri5_yerr = np.zeros(len(tol))
    dopri5_nstp = np.zeros(len(tol))

    dop853_runtime = np.zeros(len(tol))
    dop853_fe_seq = np.zeros(len(tol))
    dop853_fe_tot = np.zeros(len(tol))
    dop853_yerr = np.zeros(len(tol))
    dop853_nstp = np.zeros(len(tol))
    
    # This is necessary because multiprocessing uses pickle, which can't handle
    # Fortran function pointers
    def func2(t,y):
        return func(y,t)

    for i in range(len(tol)):
        print 'Tolerance: ', tol[i]

        # run Python extrapolation code 
        print 'running ParEx'
        start_time = time.time()
        y, infodict = ex_p.ex_midpoint_explicit_parallel(func, None, y0, t, atol=tol[i], rtol=tol[i], mxstep=nsteps, adaptive="order", full_output=True)
        py_runtime[i] = time.time() - start_time
        py_fe_seq[i], py_fe_tot[i], py_nstp[i] = infodict['fe_seq'], infodict['nfe'], infodict['nst']
        y[1:] = solout(y[1:],t[1:])
        py_yerr[i] = relative_error(y[1:], y_ref)
        print 'Runtime: ', py_runtime[i], ' s   Error: ', py_yerr[i], '   fe_seq: ', py_fe_seq[i], '   fe_tot: ', py_fe_tot[i], '   nstp: ', py_nstp[i]
        print ''
        
        # run DOPRI5 (scipy)
        print 'running DOPRI5 (scipy)'
        dopri5_d_solout = DenseSolout(t)
        start_time = time.time()
        if is_complex:
            r = complex_ode(func2).set_integrator('dopri5', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        else:
            r = ode(func2).set_integrator('dopri5', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        r.set_solout(dopri5_d_solout.solout, dense_components=tuple(range(len(y0))))
        r.set_initial_value(y0, t0)
        r.integrate(r.t+(tf-t0))
        assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
        dopri5_runtime[i] = time.time() - start_time
        y = np.array(dopri5_d_solout.dense_output)
        y[1:] = solout(y[1:],t[1:])
        dopri5_yerr[i] = relative_error(y[1:], y_ref)
        print 'Runtime: ', dopri5_runtime[i], ' s   Error: ', dopri5_yerr[i], '   fe_seq: ', dopri5_fe_seq[i], '   fe_tot: ', dopri5_fe_tot[i], '   nstp: ', dopri5_nstp[i]
        print ''

        # run DOP853 (scipy)
        print 'running DOP853 (scipy)'
        dop853_d_solout = DenseSolout(t)
        start_time = time.time()
        if is_complex:
            r = complex_ode(func2, jac=None).set_integrator('dop853', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        else:
            r = ode(func2, jac=None).set_integrator('dop853', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        r.set_solout(dop853_d_solout.solout, dense_components=tuple(range(len(y0))))
        r.set_initial_value(y0, t0)
        r.integrate(r.t+(tf-t0))
        assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
        dop853_runtime[i] = time.time() - start_time
        y = np.array(dop853_d_solout.dense_output)
        y[1:] = solout(y[1:],t[1:])
        dop853_yerr[i] = relative_error(y[1:], y_ref)
        print 'Runtime: ', dop853_runtime[i], ' s   Error: ', dop853_yerr[i], '   fe_seq: ', dop853_fe_seq[i], '   fe_tot: ', dop853_fe_tot[i], '   nstp: ', dop853_nstp[i]
        print ''

        print ''
        
    print "Final data: ParEx"
    print py_runtime, py_fe_seq, py_fe_tot, py_yerr, py_nstp
    print "Final data: DOPRI5 (scipy)"
    print dopri5_runtime, dopri5_fe_seq, dopri5_fe_tot, dopri5_yerr, dopri5_nstp
    print "Final data: DOP853 (scipy)"
    print dop853_runtime, dop853_fe_seq, dop853_fe_tot, dop853_yerr, dop853_nstp
    print ''
    print ''

    # plot performance graphs
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.hold('true')

    py_line,   = plt.loglog(py_yerr, py_runtime, "s-")
    dopri5_line, = plt.loglog(dopri5_yerr, dopri5_runtime, "s-")
    dop853_line, = plt.loglog(dop853_yerr, dop853_runtime, "s-")
    plt.legend([py_line, dopri5_line, dop853_line], ["ParEx", "DOPRI5 (scipy)", "DOP853 (scipy)"], loc=1)
    plt.xlabel('Error')
    plt.ylabel('Wall clock time (seconds)')
    plt.title(problem_name)
    plt.show()
    plt.savefig('images/' + problem_name + '_err_vs_time.png')
    plt.close()


###############################################################
###################### TEST PROBLEMS ##########################
###############################################################

###### N-Body Problem ######
def nbod_func(y,t):
    return fnbod.fnbod(y,t)

def nbod_problem_dense(num):
    t0 = 0
    tf = 0.08
    t = np.linspace(t0, tf, num=num, endpoint=True)
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference_nbod_dense.txt")
    compare_performance_dense(nbod_func, y0, t, y_ref, "nbod_problem_dense")

###### kdv Problem ######
def kdv_init(t0):
    N = 256
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E_ = np.exp(-1j * k**3 * t0)
    x = (2*np.pi/N)*np.arange(-int(N/2),int(N/2))
    A = 25; B = 16;
    u = 3*A**2/np.cosh(0.5*(A*(x+2.)))**2 + 3*B**2/np.cosh(0.5*(B*(x+1)))**2
    U_hat = E_*np.fft.fft(u)
    return U_hat

def kdv_func(U_hat, t):
    # U_hat := exp(-i*k^3*t)*u_hat
    N = 256
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E = np.exp(1j * k**3 * t)
    E_ = np.exp(-1j * k**3 * t)
    g = -0.5j * E_ * k
    return g*np.fft.fft(np.real(np.fft.ifft(E*U_hat))**2)

def kdv_solout_dense(U_hat, ts):
    y = np.zeros_like(U_hat)
    N = 256
    for i in range(len(ts)): 
        t = ts[i]
        k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
        E = np.exp(1j * k**3 * t)
        y[i] = np.squeeze(np.real(np.fft.ifft(E*U_hat[i])))
    return y

def kdv_problem_dense(num):
    t0 = 0.
    tf = 0.003
    t = np.linspace(t0, tf, num=num, endpoint=True)
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv_dense.txt")
    compare_performance_dense(kdv_func, y0, t, y_ref, "kdv_problem_dense", is_complex=True, solout=kdv_solout_dense)

###### Burgers' Problem ######
def burgers_init(t0):
    epslison = 0.1
    N = 64
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E = np.exp(epslison * k**2 * t0)
    x = (2*np.pi/N)*np.arange(-int(N/2),int(N/2))
    u = np.sin(x)**2 * (x<0.)
    # u = np.sin(x)**2 
    U_hat = E*np.fft.fft(u)
    return U_hat

def burgers_func(U_hat, t):
    # U_hat := exp(epslison*k^2*t)*u_hat
    epslison = 0.1
    N = 64
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E = np.exp(epslison * k**2 * t)
    E_ = np.exp(-epslison * k**2 * t)
    g = -0.5j * E * k
    return g*np.fft.fft(np.real(np.fft.ifft(E_*U_hat))**2)

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

def burgers_problem_dense(num):
    t0 = 0.
    tf = 3.
    t = np.linspace(t0, tf, num=num, endpoint=True)
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers_dense.txt")
    compare_performance_dense(burgers_func, y0, t, y_ref, "burgers_problem_dense", tol_boundary=(0,4), nsteps=10e4, is_complex=True, solout=burgers_solout_dense)


########### RUN TESTS ###########
if __name__ == "__main__":
    num = 50
    nbod_problem_dense(num)
    kdv_problem_dense(num)
    burgers_problem_dense(num)


