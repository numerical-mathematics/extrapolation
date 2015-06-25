from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math 
import time
import cProfile

import ex_serial as ex_s
import ex_parallel as ex_p
import fnbod

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

def tst_adaptive(f, t0, tf, y0, order, exact, method, title="tst_adaptive"):
    '''
        Runs a test, integrating a system of initial value problems y'(t) = f(y, t)
        with the given adaptive step size and adaptive order extrapolation method
        using a sequence of absolute tolerance of local error.
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
    Atol = np.asarray([2**(-k) for k in range(16)])
    err_step = np.zeros(len(Atol))
    fe_step = np.zeros(len(Atol))
    err_order = np.zeros(len(Atol))
    fe_order = np.zeros(len(Atol))

    for i in range(len(Atol)):
        y, fe_step[i] = method(f, t0, tf, y0, p=order, Atol=(Atol[i]), exact=exact, adaptive="step")
        err_step[i] = np.linalg.norm(y - exact(tf))
        y, fe_order[i] = method(f, t0, tf, y0, p=order, Atol=(Atol[i]), exact=exact, adaptive="order")
        err_order[i] = np.linalg.norm(y - exact(tf))

    plt.hold('true')
    line_step, =plt.loglog(err_step, fe_step, 's-')
    line_order, =plt.loglog(err_order, fe_order, 's-')
    plt.legend([line_step, line_order], ["adaptive step", "adaptive order"], loc=2)

    plt.title(title + ' [order ' + str(order) + ']')
    plt.xlabel('||error||')
    plt.ylabel('fe')
    plt.show()


def tst_parallel_vs_serial(f, t0, tf, y0, title="tst_parallel_vs_serial"):
    Atol = np.asarray([10**(-k) for k in range(1, 10)])
    time_ratio = np.zeros(len(Atol))
    fe_diff = np.zeros(len(Atol))
    h_avg_diff = np.zeros(len(Atol))
    k_avg_diff = np.zeros(len(Atol))
    for i in range(len(Atol)):
        print "Atol = " + str(Atol[i])
        time_ = time.time()
        y_, fe_, h_avg_, k_avg_ = ex_p.ex_midpoint_parallel(f, t0, tf, y0, Atol=(Atol[i]), adaptive="order")
        parallel_time = time.time() - time_
        print "parallel = " + str(parallel_time) + " \tfe = " + str(fe_) + " \th_avg = " + str(h_avg_) + " \tk_avg = " + str(k_avg_)
        time_ = time.time()
        y, fe, h_avg, k_avg = ex_s.ex_midpoint_serial(f, t0, tf, y0, Atol=(Atol[i]), adaptive="order")
        serial_time = time.time() - time_
        print "serial   = " + str(serial_time) + " \tfe = " + str(fe) + " \th_avg = " + str(h_avg) + " \tk_avg = " + str(k_avg)
        time_ratio[i] = serial_time / parallel_time
        fe_diff[i] = fe_ - fe
        h_avg_diff[i] = h_avg_ - h_avg
        k_avg_diff[i] = k_avg_ - k_avg
        print "ratio    = " + str(time_ratio[i]) + " \tdiff  = " + str(fe_diff[i]) + " \tdiff  = " + str(h_avg_diff[i]) + " \tdiff  = " + str(k_avg_diff[i])
        if time_ratio[i] > 1: print "[[Speedup]]"
        print

    plt.hold('true')
    plt.semilogx(Atol, time_ratio, "s-")
    plt.semilogx(Atol, [1]*len(Atol))
    plt.title(title)
    plt.xlabel('Atol')
    plt.ylabel('serial time / parallel time')
    plt.show()

    plt.hold('true')
    plt.semilogx(Atol, fe_diff, "s-")
    plt.semilogx(Atol, [0]*len(Atol))
    plt.title(title)
    plt.xlabel('Atol')
    plt.ylabel('fe_parallel - fe_serial')
    plt.show()


def tst(f, t0, tf, exact, test_name):

    # tst_parallel_vs_serial(f, t0, tf, exact(t0), 4, exact, ex_s.ex_midpoint_serial,
    #     title="tst_parallel_vs_serial")

    for i in range(4, 6):    
        tst_convergence(f, t0, tf, exact(t0), i, exact, ex_s.ex_euler_serial,
            title=(test_name + ": fixed step (Euler)"))
        tst_adaptive(f, t0, tf, exact(t0), i, exact, ex_s.ex_euler_serial,
             title=(test_name + ": adaptive Euler"))
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
    print "n = " + str(n)
    t0 = 0
    tf = 0.5
    y0 = fnbod.init_fnbod(n)
    tst_parallel_vs_serial(f_5, t0, tf, y0, title="tst_parallel_vs_serial w/ N = " + str(n))

test5()

def cprofile_tst():
    bodys = 200
    n = 6*bodys
    Atol = 10**(-6)
    print "n = " + str(n)
    print "Atol = " + str(Atol)
    t0 = 0
    tf = 3
    y0 = fnbod.init_fnbod(n)
    # print "serial"
    # _, _, h_avg, k_avg = ex_s.ex_midpoint_serial(f_5, t0, tf, y0, Atol=Atol, adaptive="order")
    print "parallel"
    _, _, h_avg, k_avg = ex_p.ex_midpoint_parallel(f_5, t0, tf, y0, Atol=Atol, adaptive="order")

    print "h_avg = " + str(h_avg) + "\tk_avg = " + str(k_avg)

# t0 = 0
# tf = 10
# Atol = 10**(-9)
# print ex_p.ex_midpoint_parallel(f_2, t0, tf, exact_2(t0), Atol=Atol, adaptive="order")
# print ex_p.ex_midpoint_parallel(f_2, t0, tf, exact_2(t0), Atol=Atol, adaptive="order")
# print ex_p.ex_midpoint_parallel(f_3, t0, tf, exact_3(t0), Atol=Atol, adaptive="order")
# print ex_p.ex_midpoint_parallel(f_4, t0, tf, exact_4(t0), Atol=Atol, adaptive="order")
# print ex_p.ex_midpoint_parallel(f_5, t0, tf, exact_5(t0), Atol=Atol, adaptive="order")


if __name__ == "__main__":
    import doctest
    doctest.testmod()

