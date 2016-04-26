'''
Runs a performance test comparing ParEx with DOPRI5 and DOP853  
integrators from scipy.integrate.ode
Result graphs are saved in the folder ./images
'''

from __future__ import division
import numpy as np
import time
from scipy.integrate import ode, complex_ode

import parex
import fnbod

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def compare_performance(func, y0, t0, tf, y_ref, problem_name, tol_boundary=(0,6), is_complex=False, nsteps=10e5, solout=(lambda t: t)):
    print 'RUNNING COMPARISON TEST FOR ' + problem_name
    tol = [1.e-3,1.e-5,1.e-7]#,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]

    parex = {}
    dopri5 = {}
    dop853 = {}

    for method in [parex, dopri5, dop853]:
        for diagnostic in ['runtime','fe_seq','fe_tot','yerr','nstp']:
            method[diagnostic] = np.zeros(len(tol))

    def func2(t,y):
        return func(y,t)

    for i in range(len(tol)):
        print 'Tolerance: ', tol[i]

        for method, name in [(parex,'ParEx'), (dopri5,'DOPRI5'), (dop853,'DOP853')]:
            print 'running ' + name
            start_time = time.time()
            if name == 'ParEx':
                y, infodict = parex.solve(func, [t0, tf], y0, solver=parex.Solvers.EXPLICIT_MIDPOINT, atol=tol[i], rtol=tol[i], mxstep=nsteps, adaptive="order", full_output=True)
                y[-1] = solout(y[-1])
                method['yerr'][i] = relative_error(y[-1], y_ref)
            else: # scipy solvers DOPRI5 and DOP853
                if is_complex:
                    r = complex_ode(func2, jac=None).set_integrator(name.lower(), atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
                else:
                    r = ode(func2, jac=None).set_integrator(name.lower(), atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
                r.set_initial_value(y0, t0)
                r.integrate(r.t+(tf-t0))
                assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
                y = solout(r.y)
                method['yerr'][i] = relative_error(y, y_ref)

            method['runtime'][i] = time.time() - start_time
            method['fe_seq'][i], method['fe_tot'][i], method['nstp'][i] = infodict['fe_seq'], infodict['nfe'], infodict['nst']
            print 'Runtime: ', method['runtime'][i], ' s   Error: ', method['yerr'][i], '   fe_seq: ', method['fe_seq'][i], '   fe_tot: ', method['fe_tot'][i], '   nstp: ', method['nstp'][i]
            print ''
        
        print ''

    for method, name in [(parex,'ParEx'), (dopri5,'DOPRI5'), (dop853,'DOP853')]:
        print "Final data: " + name
        print method['runtime'], method['fe_seq'], method['fe_tot'], method['yerr'], method['nstp']
    print ''
    print ''

    return (parex, dopri5, dop853)

def plot_results(methods,problem_name):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.hold('true')

    parex, dopri5, dop853 = methods

    for method in [parex, dopri5, dop853]:
        method['line'],   = plt.loglog(method['yerr'], method['runtime'], "s-")
    plt.legend([parex['line'], dopri5['line'], dop853['line']], ["ParEx", "DOPRI5 (scipy)", "DOP853 (scipy)"], loc=1)
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

def nbod_problem():
    t0 = 0
    tf = 0.08
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference.txt")
    results = compare_performance(nbod_func, y0, t0, tf, y_ref, "nbod_problem")
    return results

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

def kdv_solout(U_hat):
    t = 0.003
    N = 256
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E = np.exp(1j * k**3 * t)
    return np.squeeze(np.real(np.fft.ifft(E*U_hat)))

def kdv_problem():
    t0 = 0.
    tf = 0.003
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv.txt")
    results = compare_performance(kdv_func, y0, t0, tf, y_ref, "kdv_problem", is_complex=True, solout=kdv_solout)
    return results

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

def burgers_solout(U_hat):
    t = 3.
    epslison = 0.1
    N = 64
    k = np.array(range(0,int(N/2)) + [0] + range(-int(N/2)+1,0))
    E_ = np.exp(-epslison * k**2 * t)
    return np.squeeze(np.real(np.fft.ifft(E_*U_hat)))

def burgers_problem():
    t0 = 0.
    tf = 3.
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers.txt")
    results = compare_performance(burgers_func, y0, t0, tf, y_ref, "burgers_problem", tol_boundary=(0,4), nsteps=10e4, is_complex=True, solout=burgers_solout)
    return results


########### RUN TESTS ###########
if __name__ == "__main__":
    import pickle
    for problem, name in ( (nbod_problem, 'N-body'), (kdv_problem, 'KdV'), (burgers_problem, 'Burgers') ):
        results = problem()
        f = open(name+'.data','w')
        pickle.dump(results,f)
        f.close()
        plot_results(results, name + ' problem')
