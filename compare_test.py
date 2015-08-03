'''
Runs a performance test comparing Python Extrap, Fortran DOP853, ODEX-P(12),
and scipy.integrate.ode for integrators DOPRI5 and DOP853 
Result graphs are saved in the folder ./images
'''

from __future__ import division
import numpy as np
import math 
import time
import subprocess
from scipy.integrate import ode

import ex_parallel as ex_p
import fnbod

atol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
t0 = 0
tf = 0.08
y0 = fnbod.init_fnbod(2400)
y_ref = np.loadtxt("reference.txt")

extrap_order = 12
num_threads = 4

py_runtime = np.zeros(len(atol))
py_fe_seq = np.zeros(len(atol))
py_fe_tot = np.zeros(len(atol))
py_yerr = np.zeros(len(atol))
py_nstp = np.zeros(len(atol))

f_dop853_runtime = np.zeros(len(atol))
f_dop853_fe_seq = np.zeros(len(atol))
f_dop853_fe_tot = np.zeros(len(atol))
f_dop853_yerr = np.zeros(len(atol))
f_dop853_nstp = np.zeros(len(atol))

odex_runtime = np.zeros(len(atol))
odex_fe_seq = np.zeros(len(atol))
odex_fe_tot = np.zeros(len(atol))
odex_yerr = np.zeros(len(atol))
odex_nstp = np.zeros(len(atol))

dopri5_runtime = np.zeros(len(atol))
dopri5_fe_seq = np.zeros(len(atol))
dopri5_fe_tot = np.zeros(len(atol))
dopri5_yerr = np.zeros(len(atol))
dopri5_nstp = np.zeros(len(atol))

dop853_runtime = np.zeros(len(atol))
dop853_fe_seq = np.zeros(len(atol))
dop853_fe_tot = np.zeros(len(atol))
dop853_yerr = np.zeros(len(atol))
dop853_nstp = np.zeros(len(atol))

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
    return np.linalg.norm((y-y_ref)/y_ref)/(len(y)**0.5)

def f(y,t):
    return fnbod.fnbod(y,t)

def f2(t,y):
    return fnbod.fnbod(y,t)

def f_autonomous(y):
    return fnbod.fnbod(y, 0)

for i in range(len(atol)):
    print 'Tolerance: ', atol[i]

    # run Python extrapolation code 
    print 'running Python Extrap'
    start_time = time.time()
    y, infodict = ex_p.ex_midpoint_parallel(f, y0, [t0, tf], atol=(atol[i]), adaptive="order", full_output=True)
    py_runtime[i] = time.time() - start_time
    py_fe_seq[i], py_fe_tot[i], py_nstp[i] = infodict['fe_seq'], infodict['fe_tot'], infodict['nstp']
    py_yerr[i] = relative_error(y[-1], y_ref)
    print 'Runtime: ', py_runtime[i], ' s   Error: ', py_yerr[i], '   fe_seq: ', py_fe_seq[i], '   fe_tot: ', py_fe_tot[i], '   nstp: ', py_nstp[i]
    print ''
    
    # run Fortran DOP853
    replace_in_file('odex/dr_dop853.f','odex/driver.f','relative_tolerance',str(atol[i]))
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
    replace_in_file('odex/dr_odex.f','odex/driver.f','relative_tolerance',str(atol[i]))
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

    # run DOPRI5
    print 'running DOPRI5'
    start_time = time.time()
    r = ode(f2).set_integrator('dopri5', atol=atol[i], rtol=0, verbosity=10, nsteps= 10e5)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    dopri5_runtime[i] = time.time() - start_time
    dopri5_yerr[i] = relative_error(r.y, y_ref)
    print 'Runtime: ', dopri5_runtime[i], ' s   Error: ', dopri5_yerr[i], '   fe_seq: ', dopri5_fe_seq[i], '   fe_tot: ', dopri5_fe_tot[i], '   nstp: ', dopri5_nstp[i]
    print ''

    # run Python DOP853
    print 'running Python DOP853'
    start_time = time.time()
    r = ode(f2, jac=None).set_integrator('dop853', atol=atol[i], rtol=0, verbosity=10, nsteps= 10e5)
    r.set_initial_value(y0, t0)
    r.integrate(r.t+(tf-t0))
    assert r.t == tf, "Integration did not converge. Try increasing the max number of steps"
    dop853_runtime[i] = time.time() - start_time
    dop853_yerr[i] = relative_error(r.y, y_ref)
    print 'Runtime: ', dop853_runtime[i], ' s   Error: ', dop853_yerr[i], '   fe_seq: ', dop853_fe_seq[i], '   fe_tot: ', dop853_fe_tot[i], '   nstp: ', dop853_nstp[i]
    print ''

    print ''

print "Final data: Python Extrap"
print py_runtime, py_fe_seq, py_fe_tot, py_yerr, py_nstp
print "Final data: Fortran DOP853"
print f_dop853_runtime, f_dop853_fe_seq, f_dop853_fe_tot, f_dop853_yerr, f_dop853_nstp
print "Final data: ODEX-P"
print odex_runtime, odex_fe_seq, odex_fe_tot, odex_yerr, odex_nstp
print "Final data: DOPRI5"
print dopri5_runtime, dopri5_fe_seq, dopri5_fe_tot, dopri5_yerr, dopri5_nstp
print "Final data: Python DOP853"
print dop853_runtime, dop853_fe_seq, dop853_fe_tot, dop853_yerr, dop853_nstp

# plot performance graphs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.hold('true')
py_line,   = plt.loglog(py_yerr, py_runtime, "s-")
f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_runtime, "s-")
odex_line, = plt.loglog(odex_yerr, odex_runtime, "s-")
dopri5_line, = plt.loglog(dopri5_yerr, dopri5_runtime, "s-")
dop853_line, = plt.loglog(dop853_yerr, dop853_runtime, "s-")
plt.legend([py_line, f_dop853_line, odex_line, dopri5_line, dop853_line], ["Python Extrap", "Fortran DOP853", "ODEX-P(12)", "DOPRI5", "Python DOP853"], loc=1)
plt.xlabel('Error')
plt.ylabel('Wall clock time (seconds)')
plt.savefig('images/err_vs_time.png')
plt.close()

py_line,   = plt.loglog(py_yerr, py_fe_seq, "s-")
f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_fe_seq, "s-")
odex_line, = plt.loglog(odex_yerr, odex_fe_seq, "s-")
plt.legend([py_line, f_dop853_line, odex_line], ["Python Extrap", "Fortran DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('Error')
plt.ylabel('Sequential derivative evaluations')
plt.savefig('images/err_vs_fe_seq.png')
plt.close()

py_line,   = plt.loglog(py_yerr, py_fe_tot, "s-")
f_dop853_line, = plt.loglog(f_dop853_yerr, f_dop853_fe_tot, "s-")
odex_line, = plt.loglog(odex_yerr, odex_fe_tot, "s-")
plt.legend([py_line, f_dop853_line, odex_line], ["Python Extrap", "Fortran DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('Error')
plt.ylabel('Total derivative evaluations')  
plt.savefig('images/err_vs_fe_tot.png')
plt.close()

py_line,   = plt.loglog(atol, py_nstp, "s-")
f_dop853_line, = plt.loglog(atol, f_dop853_nstp, "s-")
odex_line, = plt.loglog(atol, odex_nstp, "s-")
plt.legend([py_line, f_dop853_line, odex_line], ["Python Extrap", "Fortran DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('tol')
plt.ylabel('Total number of steps')  
plt.savefig('images/tol_vs_nstp.png')
plt.close()
