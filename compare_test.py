'''
Runs a performance test comparing ParEx, Fortran DOP853, ODEX-P(12),
and scipy.integrate.ode for integrators DOPRI5 and DOP853 
Result graphs are saved in the folder ./images
'''

from __future__ import division
import numpy as np
import time
import subprocess
from scipy.integrate import ode
from scipy.integrate import complex_ode

import ex_parallel as ex_p
import fnbod

def replace_in_file(infile, outfile, oldstring, newstring):
    f_in = open(infile,'r')
    f_out = open(outfile,'w')
    for line in f_in:
        f_out.write(line.replace(oldstring,newstring))
    f_out.close()
    f_in.close()

def get_fe(out):
    fe_total = float(out[out.find("fcn=")+4:out.find("step=")])
    step = float(out[out.find("step=")+5:out.find("accpt=")])
    return (fe_total, step)

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def compare_performance(func, y0, t0, tf, y_ref, problem_name, tol_boundary=(0,6), is_complex=False, nsteps=10e5, solout=(lambda t: t), run_odex_code=False):
    print 'RUNNING COMPARISON TEST FOR ' + problem_name
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]
    extrap_order = 12
    num_threads = 4

    py_runtime = np.zeros(len(tol))
    py_fe_seq = np.zeros(len(tol))
    py_fe_tot = np.zeros(len(tol))
    py_yerr = np.zeros(len(tol))
    py_nstp = np.zeros(len(tol))

    if run_odex_code:
        f_dop853_runtime = np.zeros(len(tol))
        f_dop853_fe_seq = np.zeros(len(tol))
        f_dop853_fe_tot = np.zeros(len(tol))
        f_dop853_yerr = np.zeros(len(tol))
        f_dop853_nstp = np.zeros(len(tol))

        odex_runtime = np.zeros(len(tol))
        odex_fe_seq = np.zeros(len(tol))
        odex_fe_tot = np.zeros(len(tol))
        odex_yerr = np.zeros(len(tol))
        odex_nstp = np.zeros(len(tol))

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
    
    adams_runtime = np.zeros(len(tol))
    adams_fe_seq = np.zeros(len(tol))
    adams_fe_tot = np.zeros(len(tol))
    adams_yerr = np.zeros(len(tol))
    adams_nstp = np.zeros(len(tol))

    # This is necessary because multiprocessing uses pickle, which can't handle
    # Fortran function pointers
    def func2(t,y):
        return func(y,t)

    for i in range(len(tol)):
        print 'Tolerance: ', tol[i]

        # run Python extrapolation code 
        print 'running ParEx'
        start_time = time.time()
        y, infodict = ex_p.ex_midpoint_explicit_parallel(func, None, y0, [t0, tf], atol=tol[i], rtol=tol[i], mxstep=nsteps, adaptive="order", full_output=True)
        py_runtime[i] = time.time() - start_time
        py_fe_seq[i], py_fe_tot[i], py_nstp[i] = infodict['fe_seq'], infodict['nfe'], infodict['nst']
        py_yerr[i] = relative_error(y[-1], y_ref)
        print 'Runtime: ', py_runtime[i], ' s   Error: ', py_yerr[i], '   fe_seq: ', py_fe_seq[i], '   fe_tot: ', py_fe_tot[i], '   nstp: ', py_nstp[i]
        print ''
        
        # run DOPRI5 (scipy)
        print 'running DOPRI5 (scipy)'
        start_time = time.time()
        if is_complex:
            r = complex_ode(func2).set_integrator('dopri5', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        else:
            r = ode(func2).set_integrator('dopri5', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        r.set_initial_value(y0, t0)
        r.integrate(r.t+(tf-t0))
        assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
        dopri5_runtime[i] = time.time() - start_time
        y = solout(r.y)
        dopri5_yerr[i] = relative_error(y, y_ref)
        print 'Runtime: ', dopri5_runtime[i], ' s   Error: ', dopri5_yerr[i], '   fe_seq: ', dopri5_fe_seq[i], '   fe_tot: ', dopri5_fe_tot[i], '   nstp: ', dopri5_nstp[i]
        print ''

        # run DOP853 (scipy)
        print 'running DOP853 (scipy)'
        start_time = time.time()
        if is_complex:
            r = complex_ode(func2, jac=None).set_integrator('dop853', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        else:
            r = ode(func2, jac=None).set_integrator('dop853', atol=tol[i], rtol=tol[i], verbosity=10, nsteps=nsteps)
        r.set_initial_value(y0, t0)
        r.integrate(r.t+(tf-t0))
        assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
        dop853_runtime[i] = time.time() - start_time
        y = solout(r.y)
        dop853_yerr[i] = relative_error(y, y_ref)
        print 'Runtime: ', dop853_runtime[i], ' s   Error: ', dop853_yerr[i], '   fe_seq: ', dop853_fe_seq[i], '   fe_tot: ', dop853_fe_tot[i], '   nstp: ', dop853_nstp[i]
        print ''
        
        # run adams extrapolation code 
        #Can only run for real problems (non complex values)
#         print 'running adams Extrap'
#         start_time = time.time()
#         y, infodict = scipy.integrate.odeint(func, y0, [t0, tf],Dfun=None,mxstep=10000000, atol=tol[i], rtol=tol[i], full_output=True)
#         adams_runtime[i] = time.time() - start_time
#         print"y"+str(y[-1])
#         y[-1] = solout(y[-1])
#         print"ysol"+str(y[-1])
#         adams_fe_tot[i], adams_nstp[i] = infodict['nfe'], infodict['nst']
#         print"yref"+str(y_ref)
#         adams_yerr[i] = relative_error(y[-1], y_ref)
#         print 'Runtime: ', adams_runtime[i], ' s   Error: ', adams_yerr[i], '   fe_seq: ', adams_fe_seq[i], '   fe_tot: ', adams_fe_tot[i], '   nstp: ', adams_nstp[i]
#         print ''

        if run_odex_code:
            # run Fortran DOP853
            replace_in_file('odex/dr_dop853.f','odex/driver.f','relative_tolerance',str(tol[i]))
            subprocess.call('gfortran -O3 odex/driver.f',shell=True)
            print 'running Fortran DOP853'
            start_time = time.time()
            proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out, err = proc.communicate()
            f_dop853_runtime[i] = time.time() - start_time
            f_dop853_yerr[i] = float(out.split()[2])
            f_dop853_fe_seq[i], step = get_fe(out)
            f_dop853_fe_tot[i] = f_dop853_fe_seq[i]
            f_dop853_nstp[i] = step
            print 'Runtime: ', f_dop853_runtime[i], ' s   Error: ', f_dop853_yerr[i], '   fe_seq: ', f_dop853_fe_seq[i], '   fe_tot: ', f_dop853_fe_tot[i], '   nstp: ', f_dop853_nstp[i]
            print ''

            # run ODEX-P
            replace_in_file('odex/dr_odex.f','odex/driver.f','relative_tolerance',str(tol[i]))
            replace_in_file('odex/odex_template.f','odex/odex_load_balanced.f','half_method_order',str(extrap_order/2))
            subprocess.call('gfortran -O3 -fopenmp odex/driver.f',shell=True)
            print 'running ODEX with p =', extrap_order
            start_time = time.time()
            proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env = {'OMP_NUM_THREADS': str(num_threads)})
            out, err = proc.communicate()
            odex_runtime[i] = time.time() - start_time
            odex_yerr[i] = float(out.split()[2])
            odex_fe_tot[i], step = get_fe(out)    
            odex_fe_seq[i] = step*extrap_order
            odex_nstp[i] = step
            print 'Runtime: ', odex_runtime[i], ' s   Error: ', odex_yerr[i], '   fe_seq: ', odex_fe_seq[i], '   fe_tot: ', odex_fe_tot[i], '   nstp: ', odex_nstp[i]
            print ''

        print ''

    print "Final data: ParEx"
    print py_runtime, py_fe_seq, py_fe_tot, py_yerr, py_nstp
    print "Final data: DOPRI5 (scipy)"
    print dopri5_runtime, dopri5_fe_seq, dopri5_fe_tot, dopri5_yerr, dopri5_nstp
    print "Final data: DOP853 (scipy)"
    print dop853_runtime, dop853_fe_seq, dop853_fe_tot, dop853_yerr, dop853_nstp
    if run_odex_code:
        print "Final data: Fortran DOP853"
        print f_dop853_runtime, f_dop853_fe_seq, f_dop853_fe_tot, f_dop853_yerr, f_dop853_nstp
        print "Final data: ODEX-P"
        print odex_runtime, odex_fe_seq, odex_fe_tot, odex_yerr, odex_nstp

    # plot performance graphs
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.hold('true')

    py_line,   = plt.loglog(py_yerr, py_runtime, "s-")
    dopri5_line, = plt.loglog(dopri5_yerr, dopri5_runtime, "s-")
    dop853_line, = plt.loglog(dop853_yerr, dop853_runtime, "s-")
#     adams_line,   = plt.loglog(adams_yerr, adams_runtime, "s-")
    if run_odex_code:
        f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_runtime, "s-")
        odex_line, = plt.loglog(odex_yerr, odex_runtime, "s-")
        plt.legend([py_line, f_dop853_line, odex_line, dopri5_line, dop853_line], ["ParEx", "Fortran DOP853", "ODEX-P(12)", "DOPRI5 (scipy)", "DOP853 (scipy)"], loc=1)
    else:
        plt.legend([py_line, dopri5_line, dop853_line], ["ParEx", "DOPRI5 (scipy)", "DOP853 (scipy)"], loc=1)
    plt.xlabel('Error')
    plt.ylabel('Wall clock time (seconds)')
    plt.title(problem_name)
    plt.show()
    plt.savefig('images/' + problem_name + '_err_vs_time.png')
    plt.close()

    if run_odex_code:
        py_line,   = plt.loglog(py_yerr, py_fe_seq, "s-")
        f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_fe_seq, "s-")
        odex_line, = plt.loglog(odex_yerr, odex_fe_seq, "s-")
        plt.legend([py_line, f_dop853_line, odex_line], ["ParEx", "Fortran DOP853", "ODEX-P(12)"], loc=1)
        plt.xlabel('Error')
        plt.ylabel('Sequential derivative evaluations')
        plt.title(problem_name)
        plt.savefig('images/' + problem_name + '_err_vs_fe_seq.png')
        plt.close()

        py_line,   = plt.loglog(py_yerr, py_fe_tot, "s-")
        f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_fe_tot, "s-")
        odex_line, = plt.loglog(odex_yerr, odex_fe_tot, "s-")
        plt.legend([py_line, f_dop853_line, odex_line], ["ParEx", "Fortran DOP853", "ODEX-P(12)"], loc=1)
        plt.xlabel('Error')
        plt.ylabel('Total derivative evaluations')  
        plt.title(problem_name)
        plt.savefig('images/' + problem_name + '_err_vs_fe_tot.png')
        plt.close()

        py_line,   = plt.loglog(tol, py_nstp, "s-")
        f_dop853_line, = plt.loglog(tol, f_dop853_nstp, "s-")
        odex_line, = plt.loglog(tol, odex_nstp, "s-")
        plt.legend([py_line, f_dop853_line, odex_line], ["ParEx", "Fortran DOP853", "ODEX-P(12)"], loc=1)
        plt.xlabel('tol')
        plt.ylabel('Total number of steps')  
        plt.title(problem_name)
        plt.savefig('images/' + problem_name + '_tol_vs_nstp.png')
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
    compare_performance(nbod_func, y0, t0, tf, y_ref, "nbod_problem")

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
    compare_performance(kdv_func, y0, t0, tf, y_ref, "kdv_problem", is_complex=True, solout=kdv_solout)

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
    compare_performance(burgers_func, y0, t0, tf, y_ref, "burgers_problem", tol_boundary=(0,4), nsteps=10e4, is_complex=True, solout=burgers_solout)


########### RUN TESTS ###########
if __name__ == "__main__":
    nbod_problem()
    kdv_problem()
    burgers_problem()


