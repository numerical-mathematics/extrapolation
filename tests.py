from __future__ import division
import numpy as np
import math 
import time
import cProfile

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import ex_serial as ex_s
import ex_parallel as ex_p
import fnbod

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def tst_convergence(f, t0, tf, y0, order, exact, method, title="tst_convergence"):
    '''
        Runs a convergence test, integrating the system of initial value problems
        y'(t) = f(y, t) using a sequence of fixed step sizes with the provided
        extrapolation method.
        Creates a plot of the resulting errors versus step size with a reference
        line with the given order to compare with the method error 

        **Inputs**:
            - f         -- the right hand side function of the IVP.
                        Must output a non-scalar numpy.ndarray
            - [t0, tf]  -- the interval of integration
            - y0        -- the value of y(t0). Must be a non-scalar numpy.ndarray
            - order     -- the order of extrapolation
            - exact     -- the exact solution to the IVP.
                        Must output a non-scalar numpy.ndarray
            - method    -- the extrapolation method function
            - title     -- the title of the graph produced (optional)
    '''
    hs = np.asarray([2**(-k) for k in range(10)])
    err = np.zeros(len(hs))

    for i in range(len(hs)):
        y, _ = method(f, t0, tf, y0, p=order, step_size=(hs[i]), adaptive="fixed")
        err[i] = np.linalg.norm(y - exact(tf))

    plt.hold('true')
    method_err,  = plt.loglog(hs, err, 's-')
    order_line,  = plt.loglog(hs, (hs**order)*(err[5]/hs[5]))
    plt.legend([method_err, order_line], ['Method Error', 'order ' + str(order)], loc=2)
    plt.title(title)
    plt.ylabel('||error||')
    plt.xlabel('time step size')
    plt.show()
    return err

def tst_adaptive(f, t0, tf, y0, order, exact, method, title="tst_adaptive"):
    '''
        Runs a test, integrating a system of initial value problems y'(t) = f(y, t)
        with the given adaptive step size and adaptive order extrapolation method
        using a sequence of tolerances for local error.
        Creates a plot of the number of f evaluations versus the global error.

        **Inputs**:
            - f         -- the right hand side function of the IVP.
                        Must output a non-scalar numpy.ndarray
            - [t0, tf]  -- the interval of integration
            - y0        -- the value of y(t0). Must be a non-scalar numpy.ndarray
            - order     -- the order of extrapolation. 
            - exact     -- the exact solution to the IVP. 
                        Must output a non-scalar numpy.ndarray
            - method    -- the extrapolation method function
            - title     -- the title of the graph produced (optional)
    '''
    tol = np.asarray([2**(-k) for k in range(16)])
    err_step = np.zeros(len(tol))
    fe_step = np.zeros(len(tol))
    err_order = np.zeros(len(tol))
    fe_order = np.zeros(len(tol))

    for i in range(len(tol)):
        y, fe_step[i] = method(f, t0, tf, y0, p=order, atol=(tol[i]), rtol=(tol[i]), exact=exact, adaptive="step")
        err_step[i] = np.linalg.norm(y - exact(tf))
        y, fe_order[i], _, _ = method(f, t0, tf, y0, p=order, atol=(tol[i]), rtol=(tol[i]), exact=exact, adaptive="order")
        err_order[i] = np.linalg.norm(y - exact(tf))

    plt.hold('true')
    line_step, =plt.loglog(err_step, fe_step, 's-')
    line_order, =plt.loglog(err_order, fe_order, 's-')
    plt.legend([line_step, line_order], ["adaptive step", "adaptive order"], loc=2)

    plt.title(title + ' [order ' + str(order) + ']')
    plt.xlabel('||error||')
    plt.ylabel('fe')
    plt.show()
    return (err_step, err_order)


def tst_parallel_vs_serial(f, t0, tf, y0, title="tst_parallel_vs_serial"):
    tol = np.asarray([10**(-k) for k in range(3, 14)])
    time_ratio = np.zeros(len(tol))
    fe_seq = np.zeros(len(tol))
    fe_tot = np.zeros(len(tol))
    fe_diff = np.zeros(len(tol))
    h_avg_diff = np.zeros(len(tol))
    k_avg_diff = np.zeros(len(tol))
    err_p = np.zeros(len(tol))
    err_s = np.zeros(len(tol))
    y_ref = np.loadtxt("reference.txt")
    for i in range(len(tol)):
        print("tol = " + str(tol[i]))
        time_ = time.time()

        y_, infodict = ex_p.ex_midpoint_parallel(f, y0, [t0, tf], atol=(tol[i]), rtol=(tol[i]), adaptive="order", full_output=True)
        parallel_time = time.time() - time_
        fe_seq[i], fe_tot[i], h_avg_, k_avg_ = infodict['fe_seq'], infodict['fe_tot'], infodict['h_avg'], infodict['k_avg']
        err_p[i] = relative_error(y_[-1], y_ref)
        print("parallel = " + '%g' % (parallel_time) + "\terr  = " + '%e' % err_p[i] + "\th_avg = " + '%e' % h_avg_ + "\tk_avg = " + '%e' % k_avg_ + "\tfe_s1 = " + str(fe_seq[i]) + "\tfe_t1 = " + str(fe_tot[i]) + "\tfe_t1/fe_s1 = " + '%g' % (fe_tot[i]/fe_seq[i]))
        time_ = time.time()
        y, fe, h_avg, k_avg = ex_s.ex_midpoint_serial(f, t0, tf, y0, atol=(tol[i]), rtol=(tol[i]), adaptive="order")
        serial_time = time.time() - time_
        err_s[i] = relative_error(y, y_ref)
        print("serial   = " + '%g' % (serial_time)   + "\terr  = " + '%e' % err_s[i] + "\th_avg = " + '%e' % h_avg + "\tk_avg = " + '%e' % k_avg + "\tfe_s2 = " + str(fe) + "\tfe_t2 = " + str(fe) + "\tfe_s2/fe_s1 = " + '%g' % (fe/fe_seq[i]))
        time_ratio[i] = serial_time / parallel_time
        fe_diff[i] = fe_seq[i] - fe
        h_avg_diff[i] = h_avg_ - h_avg
        k_avg_diff[i] = k_avg_ - k_avg
        print("ratio    = " + '%g' % (time_ratio[i]) + "\tdiff = " + '%e' %(err_p[i] - err_s[i]) + "\tdiff  = " + '%e' %(h_avg_diff[i]) + "\tdiff  = " + '%e' %(k_avg_diff[i]) + "\tdiff  = " + str(fe_diff[i]) + "\tdiff  = " + str(fe_tot[i] - fe) + "\ttime_ratio/(fe_s2/fe_s1) = " + '%g' % (time_ratio[i]/(fe/fe_seq[i])))
        print('')

    return (err_p, err_s)


def tst(f, t0, tf, exact, test_name):

    # tst_parallel_vs_serial(f, t0, tf, exact(t0), 4, exact, ex_s.ex_midpoint_serial,
    #     title="tst_parallel_vs_serial")
    
    for i in range(4, 6):    
        tst_convergence(f, t0, tf, exact(t0), i, exact, ex_s.ex_euler_serial,
            title=(test_name + ": fixed step (Euler)"))
        # Not yet implemented:
        #tst_adaptive(f, t0, tf, exact(t0), i, exact, ex_s.ex_euler_serial,
        #     title=(test_name + ": adaptive Euler"))
    for i in range(4, 12, 2):    
        tst_convergence(f, t0, tf, exact(t0), i, exact, ex_s.ex_midpoint_serial,
            title=(test_name + ": fixed step (midpoint)"))
        tst_adaptive(f, t0, tf, exact(t0), i, exact, ex_s.ex_midpoint_serial,
             title=(test_name + ": adaptive Midpoint"))

################
delay = 0

def f_1(y,t):
    lam = -1j
    y0 = np.array([1 + 0j])
    time.sleep(delay)
    return lam*y

def exact_1(t):
    lam = -1j
    y0 = np.array([1 + 0j])
    return y0*np.exp(lam*t)

def test1():
    t0 = 0
    tf = 10
    tst(f_1, t0, tf, exact_1, "TEST 1")


################

def f_2(y,t):
    time.sleep(delay)
    return 4.*y*float(np.sin(t))**3*np.cos(t)
    
def exact_2(t):
    y0 = np.array([1])
    return y0*np.exp((np.sin(t))**4)
    
def test2():
    t0 = 0
    tf = 10
    tst(f_2, t0, tf, exact_2, "TEST 2")

################

def f_3(y,t):
    time.sleep(delay)
    return 4.*t*np.sqrt(y)
    
def exact_3(t):
    return np.array([(1.+t**2)**2])

def test3():
    t0 = 0
    tf = 10
    tst(f_3, t0, tf, exact_3, "TEST 3")

################

def f_4(y,t):
    time.sleep(delay)
    return y/t*np.log(y)
    
def exact_4(t):
    return np.array([np.exp(2.*t)])

def test4():
    t0 = 0.5
    tf = 10
    tst(f_4, t0, tf, exact_4, "TEST 4")

###############

def f_5(y,t):
    return fnbod.fnbod(y,t)

def test5():
    bodys = 400
    n = 6*bodys
    print("n = " + str(n))
    t0 = 0
    tf = 0.08
    y0 = fnbod.init_fnbod(n)
    return tst_parallel_vs_serial(f_5, t0, tf, y0, title="tst_parallel_vs_serial w/ N = " + str(n))

def cprofile_tst():
    bodys = 200
    n = 6*bodys
    tol = 10**(-6)
    print("n = " + str(n))
    print("tol = " + str(tol))
    t0 = 0
    tf = 3
    y0 = fnbod.init_fnbod(n)
    # print("serial")
    # _, _, h_avg, k_avg = ex_s.ex_midpoint_serial(f_5, t0, tf, y0, atol=tol, rtol=tol, adaptive="order")
    print("parallel")
    _, infodict = ex_p.ex_midpoint_parallel(f_5, y0, [t0, tf], atol=tol, rtol=tol, adaptive="order", full_output=True)
    h_avg, k_avg = infodict['h_avg'], infodict['k_avg']
    print("h_avg = " + str(h_avg) + "\tk_avg = " + str(k_avg))


import multiprocessing as mp
import random

def f_6(y,t):
    return -np.array([math.sin(t)])

def exact_6(t):
    return np.array([math.cos(t)])

def test_interpolation():
    f = f_6
    exact = lambda t: np.array([math.cos(t)])
    t0 = 0
    # t0 = random.random()*math.pi
    h = t0+2*math.pi
    pool = mp.Pool(mp.cpu_count())    
    seq = lambda t: 4*t -2
    dense = True
    ts = np.linspace(t0,t0+h,500)
    ts_poly = [(t - t0)/h for t in ts]
    ys_exact = np.array([exact(t) for t in ts])
    max_k = 10
    min_k = 3
    err = np.zeros(max_k + 1)
    for k in range(min_k,max_k):
        _, _, _, y0, Tkk, f_Tkk, y_half, f_yj, hs = ex_p.compute_ex_table(f, t0, exact(t0), (), h, k, pool, seq=seq, dense=dense)
        poly = ex_p.interpolate(y0, Tkk, f_Tkk, y_half, f_yj, hs, h, k, 0, 0)
        ys_poly = np.array([poly(t)[0] for t in ts_poly])
        
        plt.hold(True)
        line_exact, = plt.plot(ts, ys_exact)
        line_poly, = plt.plot(ts, ys_poly)
        plt.legend([line_exact, line_poly], ["exact", "poly w/ k=" + str(k)], loc=4)
        plt.show()
        
        err[k] = relative_error(ys_poly, ys_exact)
        line_error, = plt.semilogy(ts, abs(ys_exact - ys_poly))
        plt.legend([line_error], ["error w/ k=" + str(k)], loc=4)
        plt.show()

    plt.semilogy(range(min_k,max_k+1), err[min_k:], "s-")
    plt.show()


def test_dense():
    f = f_6
    exact = exact_6
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    t0 = 0
    # t0 = random.random()*math.pi
    t_max = t0 + 2*math.pi
    y0 = exact(t0)
    t = np.linspace(t0,t_max,100)
    ys_exact = np.array([exact(t_) for t_ in t])
    err = np.zeros(len(tol))
    
    for i in range(len(tol)):
        print("tol = ", tol[i])
        ys = ex_p.ex_midpoint_parallel(f, y0, t, atol=tol[i], rtol=tol[i], adaptive="order")
        plt.hold(True)
        line_exact, = plt.plot(t, ys_exact, "s-")
        line_sol, = plt.plot(t, ys, "s-")
        plt.legend([line_exact, line_sol], ["exact", "sol w/ tol =" + str(tol[i])], loc=4)
        plt.show()
        err[i] = relative_error(ys, ys_exact)
        line_error, = plt.semilogy(t, abs(ys_exact - ys), "s-")
        plt.legend([line_error], ["error"], loc=4)
        plt.show()

    plt.loglog(tol, err, "s-")
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    test5()
    test_interpolation()
    test_dense()
    pass
