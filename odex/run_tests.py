"""This script produces the results in the final section of the
manuscript."""

import numpy as np
import subprocess

tolerances = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
extrap_order = 12
num_threads = 4
doptimes   = np.zeros(len(tolerances))
doperrors  = np.zeros(len(tolerances))
odextimes  = np.zeros(len(tolerances))
odexerrors = np.zeros(len(tolerances))


def replace_in_file(infile, outfile, oldstring, newstring):
    f_in = open(infile,'r')
    f_out = open(outfile,'w')
    for line in f_in:
        f_out.write(line.replace(oldstring,newstring))
    f_out.close()
    f_in.close()

def get_runtime(stderr):
    l = stderr.split()
    ind = l.index('real')
    return float(l[ind-1])


#Main script starts here
for itol,tol in enumerate(tolerances):
    print 'Tolerance: ', tol
    #First run DOP853
    replace_in_file('dr_dop853.f','driver.f','relative_tolerance',str(tol))
    #Now run extrapolation at different orders and with different #'s of threads
    subprocess.call('gfortran driver.f',shell=True)
    print 'running DOP853'
    proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out, err = proc.communicate()
    yerr = float(out.split()[2])
    doperrors[itol] = yerr
    doptimes[itol] = get_runtime(err)
    print 'Runtime: ', doptimes[itol], ' s   Error: ', doperrors[itol]

    #Now run ODEX
    replace_in_file('dr_odex.f','driver.f','relative_tolerance',str(tol))
    replace_in_file('odex_template.f','odex_load_balanced.f','half_method_order',str(extrap_order/2))
    subprocess.call('gfortran -fopenmp driver.f',shell=True)
    print 'running ODEX with p =', extrap_order
    proc = subprocess.Popen(['time', './a.out'],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env = {'OMP_NUM_THREADS': str(num_threads)})
    out, err = proc.communicate()
    yerr = float(out.split()[2])
    odexerrors[itol] = yerr
    odextimes[itol] = get_runtime(err)
    print 'Runtime: ', odextimes[itol], ' s   Error: ', odexerrors[itol]

    print ''

print doptimes, odextimes
print doperrors, odexerrors

import matplotlib.pyplot as plt
plt.loglog(doperrors, doptimes,'k-s',linewidth=3,markersize=14)
plt.hold(True)
plt.loglog(odexerrors, odextimes,'g-D',linewidth=3,markersize=14)
plt.hold(False)
plt.grid()
plt.xlabel(r'RMS Error at $t_{final}$')
plt.ylabel('Wall clock time (seconds)')
plt.legend( ('DOP853', 'ODEX-P(12)') )
