'''
Runs a performance test comparing the parallel speedup achieved by ParEx 
for nworkers = 2, 4, 6, 8, and the theoretical expected speedup.
Problem used for testing are N-body, KDV equation, and Burgers equation.
It also compares the average extrapolation order and step size for each nworkers.

Resulting graphs are saved in the `./images` folder
'''

from __future__ import division
import numpy as np
import time

import ex_parallel as ex_p
import fnbod

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def compare_speedup(func, y0, t0, tf, y_ref, problem_name, tol = 1e-9, nsteps=10e5, solout=(lambda t: t)):
    print 'RUNNING PARALLEL SPEEDUP TEST FOR ' + problem_name
    nworkers = [1, 2, 4, 6, 8]

    runtime = np.zeros(len(nworkers))
    fe_seq = np.zeros(len(nworkers))
    fe_tot = np.zeros(len(nworkers))
    yerr = np.zeros(len(nworkers))
    nstp = np.zeros(len(nworkers))
    h_avg = np.zeros(len(nworkers))
    k_avg = np.zeros(len(nworkers))
    speedup = np.zeros(len(nworkers))

    for i in range(len(nworkers)):
        print 'nworkers: ', nworkers[i]
        start_time = time.time()
        y, infodict = ex_p.ex_midpoint_explicit_parallel(func, None, y0, [t0, tf], atol=tol, 
                                                         rtol=tol, mxstep=nsteps, nworkers=nworkers[i], 
                                                         adaptive="order", full_output=True)
        runtime[i] = time.time() - start_time
        # print"y"+str(y[-1])
        y[-1] = solout(y[-1])
        # print"ysol"+str(y[-1])
        fe_seq[i], fe_tot[i], nstp[i], h_avg[i], k_avg[i] = infodict['fe_seq'], infodict['nfe'], infodict['nst'], infodict['h_avg'], infodict['k_avg']
        # print"yref"+str(y_ref)
        yerr[i] = relative_error(y[-1], y_ref)
        print 'Runtime: ', runtime[i], ' s   Error: ', yerr[i], '   fe_seq: ', fe_seq[i], '   fe_tot: ', fe_tot[i], '   nstp: ', nstp[i], '   h_avg: ', h_avg[i], '   k_avg: ', k_avg[i], 
        speedup[i]= runtime[0]/runtime[i]
        print '\nSpeedup: ', speedup[i]
        print ''       
 
    # plot performance graphs
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.hold('true')

    runtime_line,   = plt.semilogy(nworkers, runtime, "s-")
    plt.xlabel('nworkers')
    plt.ylabel('Wall clock time (seconds)')
    plt.title('ParEx: ' + problem_name + ' at tolerance = ' + str(tol))
    plt.show()
    plt.savefig('images/' + problem_name + '_nworkers_vs_time.png')
    plt.close()

    speedup_line,   = plt.plot(nworkers, speedup, "s-")
    plt.xlabel('nworkers')
    plt.ylabel('Speedup (time with nworkers=1 / time with nworkers=n)')
    plt.title('ParEx: ' + problem_name + ' at tolerance = ' + str(tol))
    plt.show()
    plt.savefig('images/' + problem_name + '_nworkers_vs_speedup.png')
    plt.close()

    fig, plot1 = plt.subplots()
    plot2 = plot1.twinx()
    colors = ('Blue', 'Red')
    plot1_line, = plot1.semilogy(nworkers, k_avg, '-s', color=colors[0])
    plot2_line, = plot2.semilogy(nworkers, h_avg, '-s', color=colors[1])
    plot1.set_xlabel("nworkers")
    plot1.set_ylabel("Average Extrapolation Order", color= colors[0])
    plot2.set_ylabel("Average Step Size", color= colors[1])
    plot1.tick_params(axis='y', colors=colors[0])
    plot2.tick_params(axis='y', colors=colors[1])
    plt.title('ParEx: ' + problem_name + ' at tolerance = ' + str(tol))
    plt.show()
    plt.savefig('images/' + problem_name + '_nworkers_vs_k_avg_and_h_avg.png')
    plt.close()

    print "FINISHED! Images were saved in ./images folder"


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
    # compare_speedup(nbod_func, y0, t0, tf, y_ref, "nbod_problem", run_odex_code=True)
    compare_speedup(nbod_func, y0, t0, tf, y_ref, "nbod_problem")

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
    compare_speedup(kdv_func, y0, t0, tf, y_ref, "kdv_problem", solout=kdv_solout)

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
    compare_speedup(burgers_func, y0, t0, tf, y_ref, "burgers_problem", nsteps=10e4, solout=burgers_solout)

########### RUN TESTS ###########
if __name__ == "__main__":
    nbod_problem()
    kdv_problem()
    burgers_problem()


