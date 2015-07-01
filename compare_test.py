'''
Runs a performance test comparing Python Extrap, DOP853, and ODEX-P(12). 
Result graphs are saved in the folder ./images
'''

from __future__ import division
import numpy as np
import math 
import time
import subprocess

import ex_parallel as ex_p
import fnbod

Atol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
t0 = 0
tf = 0.08
y0 = fnbod.init_fnbod(2400)
y_ref = np.loadtxt("reference.txt")

extrap_order = 12
num_threads = 4

py_runtime = np.zeros(len(Atol))
py_fe_seq = np.zeros(len(Atol))
py_fe_tot = np.zeros(len(Atol))
py_yerr = np.zeros(len(Atol))

dop_runtime = np.zeros(len(Atol))
dop_fe_seq = np.zeros(len(Atol))
dop_fe_tot = np.zeros(len(Atol))
dop_yerr = np.zeros(len(Atol))

odex_runtime = np.zeros(len(Atol))
odex_fe_seq = np.zeros(len(Atol))
odex_fe_tot = np.zeros(len(Atol))
odex_yerr = np.zeros(len(Atol))

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

for i in range(len(Atol)):
    print 'Tolerance: ', Atol[i]

    # run Python extrapolation code 
    print 'running Python Extrap'
    start_time = time.time()
    y, (py_fe_seq[i], py_fe_tot[i]), _, _ = ex_p.ex_midpoint_parallel(f, t0, tf, y0, Atol=(Atol[i]), adaptive="order")
    py_runtime[i] = time.time() - start_time
    py_yerr[i] = relative_error(y, y_ref)
    print 'Runtime: ', py_runtime[i], ' s   Error: ', py_yerr[i], '   fe_seq: ', py_fe_seq[i], '   fe_tot: ', py_fe_tot[i]

    # run DOP853
    replace_in_file('odex/dr_dop853.f','odex/driver.f','relative_tolerance',str(Atol[i]))
    subprocess.call('gfortran -O3 odex/driver.f',shell=True)
    print 'running DOP853'
    start_time = time.time()
    proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out, err = proc.communicate()
    print out
    dop_runtime[i] = time.time() - start_time
    dop_yerr[i] = float(out.split()[2])
    dop_fe_seq[i], _ = get_fe(out)
    dop_fe_tot[i], _ = get_fe(out)
    print 'Runtime: ', dop_runtime[i], ' s   Error: ', dop_yerr[i], '   fe_seq: ', dop_fe_seq[i], '   fe_tot: ', dop_fe_tot[i]

    # run ODEX-P
    replace_in_file('odex/dr_odex.f','odex/driver.f','relative_tolerance',str(Atol[i]))
    replace_in_file('odex/odex_template.f','odex/odex_load_balanced.f','half_method_order',str(extrap_order/2))
    subprocess.call('gfortran -O3 -fopenmp odex/driver.f',shell=True)
    print 'running ODEX with p =', extrap_order
    start_time = time.time()
    proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env = {'OMP_NUM_THREADS': str(num_threads)})
    out, err = proc.communicate()
    print out
    odex_runtime[i] = time.time() - start_time
    odex_yerr[i] = float(out.split()[2])
    odex_fe_tot[i], step = get_fe(out)    
    odex_fe_seq[i] = step*extrap_order
    print 'Runtime: ', odex_runtime[i], ' s   Error: ', odex_yerr[i], '   fe_seq: ', odex_fe_seq[i], '   fe_tot: ', odex_fe_tot[i]

    print ''

print "Final data: Python Extrap"
print py_runtime, py_fe_seq, py_fe_tot, py_yerr
print "Final data: DOP853"
print dop_runtime, dop_fe_seq, dop_fe_tot, dop_yerr
print "Final data: ODEX-P"
print odex_runtime, odex_fe_seq, odex_fe_tot, odex_yerr

# plot performance graphs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.hold('true')
py_line,   = plt.loglog(py_yerr, py_runtime, "s-")
dop_line, = plt.loglog(dop_yerr, dop_runtime, "s-")
odex_line, = plt.loglog(odex_yerr, odex_runtime, "s-")
plt.legend([py_line, dop_line, odex_line], ["Python Extrap", "DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('Error')
plt.ylabel('Wall clock time (seconds)')
plt.savefig('images/err_vs_time.png')
plt.close()

py_line,   = plt.loglog(py_yerr, py_fe_seq, "s-")
dop_line, = plt.loglog(dop_yerr, dop_fe_seq, "s-")
odex_line, = plt.loglog(odex_yerr, odex_fe_seq, "s-")
plt.legend([py_line, dop_line, odex_line], ["Python Extrap", "DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('Error')
plt.ylabel('Sequential derivative evaluations')
plt.savefig('images/err_vs_fe_seq.png')
plt.close()

py_line,   = plt.loglog(py_yerr, py_fe_tot, "s-")
dop_line, = plt.loglog(dop_yerr, dop_fe_tot, "s-")
odex_line, = plt.loglog(odex_yerr, odex_fe_tot, "s-")
plt.legend([py_line, dop_line, odex_line], ["Python Extrap", "DOP853", "ODEX-P(12)"], loc=1)
plt.xlabel('Error')
plt.ylabel('Total derivative evaluations')  
plt.savefig('images/err_vs_fe_tot.png')
plt.close()
