
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math 

debugging = False

def ex_one_step(f, yn, tn, h, p):
    Y = np.zeros((p+1,p+1), dtype=complex)
    T = np.zeros((p+1,p+1), dtype=complex)
    fe = 0
    for k in range(1,p+1):
        Y[k,0] = yn
        for j in range(1,k+1):
            Y[k,j] = Y[k,j-1] + (h/k)*f(Y[k,j-1], tn + j*(h/k))
            fe = fe + 1
        T[k,1] = Y[k,k]

    for k in range(2, p+1):
        for j in range(k, p+1):
            T[j,k] = T[j,k-1] + (T[j,k-1] - T[j-1,k-1])/((j/(j-k+1)) - 1)

    return (T[p,p], T[p-1,p-1], fe)

def adjust_step(f, yn_1, tn_1, h, p, y, y_hat, Atol, Rtol):
    fe = 0
    facmax = 1.5
    facmin = 0.8
    fac = 0.8
    tol = Atol + max(y,y_hat)*Rtol
    # err = np.linalg.norm((y-y_hat)/tol)*sqrt(len(y))    # if y and y_hat are arrays
    err = abs(y-y_hat)/tol # if y and y_hat are scalars
    h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    while err > 1:
        h = h_new
        y, y_hat, fe_ = ex_one_step(f, yn_1, tn_1, h, p)
        fe = fe + fe_
        # err = np.linalg.norm((y-y_hat)/tol)*sqrt(len(y))    # if y and y_hat are arrays
        err = (abs(y-y_hat))/tol # if y and y_hat are scalars
        h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

        assert abs(err) < abs(complex('inf'))

    return (y, y_hat, h, h_new, fe)

# solve the first order ODE y' = f(y) by extrapolation based on Euler's method
# on the interval [t0, tf] with the intial value y(t0) = y0 and time step size
# h and order p
#
#
#
# TODO: change the function comment to explain the new behavior
def ex_euler_serial(f, y0, t0, tf, p, step_size=0.05, Atol=0, Rtol=0, fixed_step=False):
    fe = 0
    if fixed_step:
        ts, h = np.linspace(t0, tf, (tf-t0)/step_size + 1, retstep=True)
        ys = np.zeros(len(ts), dtype=complex)
        ys_hat = np.zeros(len(ts), dtype=complex)
        ys[0] = y0

        for i in range(len(ts) - 1):
            ys[i+1], ys_hat[i+1], fe_ = ex_one_step(f, ys[i], ts[i], h, p)
            fe = fe + fe_
    else:
        assert p > 1, "order of method must be greater than 1 if fixed_step is True"
        ts = np.zeros(1, dtype=complex)
        ys = np.zeros(1, dtype=complex)
        ys_hat = np.zeros(1)
        ts[0] = t0
        ys[0] = y0        
        h = min(step_size, tf-t0)
        
        t, i = t0, 0
        while t < tf:
            y, y_hat, fe_ = ex_one_step(f, ys[i], ts[i], h, p)
            fe = fe + fe_
            y, y_hat, h, h_new, fe_ = adjust_step(f, ys[i], ts[i], h, p, y, y_hat, Atol, Rtol)
            t, i, fe = t + h, i+1, fe + fe_
            ts = np.append(ts, t)
            ys = np.append(ys, y)
            ys_hat = np.append(ys, y_hat)
            h = min(h_new, tf - t)

    return (ts, ys, fe)

# preforms a convergence test solving the IVP y' = f(y) and y(t0) = y0 with
# the [method] provided, where the function exact is the solution the IVP.
# The function plots the ||error|| vs time step size, and a reference line
# of the given order to compare. The graph is titled with graph_title.
def test_convergence(f, exact, t0, tf, y0, method, order, graph_title):
    hs = np.asarray([2**(-k) for k in range(1, 10)])
    err = np.zeros(len(hs))

    for i in range(len(hs)):
        _, ys, _ = method(f, y0, t0, tf, order, step_size=(hs[i]), fixed_step=True)
        err[i] = abs(ys[-1] - exact(tf))

    plt.hold('true')
    method_err,  = plt.loglog(hs, err, 's-')
    order_line,  = plt.loglog(hs, (hs**order)*(err[5]/hs[5]))
    plt.legend([method_err, order_line], ['Method Error', 'order ' + str(order)], loc=2)
    plt.title(graph_title)
    plt.ylabel('||error||')
    plt.xlabel('time step size')
    plt.show()

def test_adaptive_step(f, exact, t0, tf, y0, method, order, graph_title):
    Atol = np.asarray([2**(-k) for k in range(1, 16)])
    err = np.zeros( len(Atol))
    fe = np.zeros(len(Atol))

    for i in range(len(Atol)):
        _, ys, fe[i] = method(f, y0, t0, tf, order, Atol=(Atol[i]))
        err[i] = abs(ys[-1] - exact(tf))

    plt.hold('true')
    line, =plt.loglog(err, fe, 's-')
    plt.legend([line], ['order ' + str(order)], loc=2)

    plt.title(graph_title)
    plt.xlabel('||error||')
    plt.ylabel('fe')
    plt.show()


# Convergence Tests
if debugging:
    #### test 1 ####
    lam = -1j
    y0 = 1
    f = lambda y,t: lam*y
    exact = lambda t: y0*np.exp(lam*t)
    t0 = 0
    tf = 5
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 1: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 1: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 1: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 1: adaptive step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 1, "TEST 1: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 1: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 1: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 1: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 1: fixed step")

    #### test 2 ####
    f = lambda y,t: 4.*y*float(np.sin(t))**3*np.cos(t)
    y0 = 1
    exact = lambda t: y0*np.exp((np.sin(t))**4)
    t0 = 0
    tf = 5
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 2: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 2: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 2: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 2: adaptive step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 1, "TEST 2: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 2: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 2: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 2: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 2: fixed step")

    #### test 3 ####
    f = lambda y,t: 4.*t*np.sqrt(y)
    y0 = 1
    exact = lambda t: (1.+t**2)**2
    t0 = 0
    tf = 5
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 3: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 3: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 3: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 3: adaptive step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 1, "TEST 3: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 3: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 3: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 3: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 3: fixed step")

    #### test 4 ####
    f = lambda y,t:  y/t*np.log(y)
    y0 = np.exp(1.)
    exact = lambda t: np.exp(2.*t)
    t0 = 0.5
    tf = 5
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 4: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 4: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 4: adaptive step")
    test_adaptive_step(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 4: adaptive step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 1, "TEST 4: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "TEST 4: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "TEST 4: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "TEST 4: fixed step")
    test_convergence(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "TEST 4: fixed step")
