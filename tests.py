from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math 

import ex_serial as exs

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
        _, ys, _ = method(f, t0, tf, y0, p=order, step_size=(hs[i]), adaptive="fixed")
        err[i] = np.linalg.norm(ys[-1] - exact(tf))

    plt.hold('true')
    method_err,  = plt.loglog(hs, err, 's-')
    order_line,  = plt.loglog(hs, (hs**order)*(err[5]/hs[5]))
    plt.legend([method_err, order_line], ['Method Error', 'order ' + str(order)], loc=2)
    plt.title(title)
    plt.ylabel('||error||')
    plt.xlabel('time step size')
    plt.show()

def tst_adaptive_step(f, t0, tf, y0, order, exact, method, title="tst_adaptive_step"):
    '''
        Runs a test, integrating a system of initial value problems y'(t) = f(y, t)
        with the given adaptive step size extrapolation method using a sequence
        of absolute tolerance of local error.
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
    err = np.zeros(len(Atol))
    fe = np.zeros(len(Atol))

    for i in range(len(Atol)):
        _, ys, fe[i] = method(f, t0, tf, y0, p=order, Atol=(Atol[i]), exact=exact, adaptive="step")
        err[i] = np.linalg.norm(ys[-1] - exact(tf))

    plt.hold('true')
    line, =plt.loglog(err, fe, 's-')
    plt.legend([line], ['order ' + str(order)], loc=2)

    plt.title(title)
    plt.xlabel('||error||')
    plt.ylabel('fe')
    plt.show()

def tst_adaptive_order(f, t0, tf, y0, order, exact, method, title="tst_adaptive_order"):

    Atol = np.asarray([10**(-k) for k in range(3, 12, 3)])
    err = np.zeros(len(Atol))
    fe = np.zeros(len(Atol))

    for i in range(len(Atol)):
        ts, ys, fe[i], h_acc, k_acc, h_rej_, k_rej_ = method(f, t0, tf, y0, 
            p=order, Atol=(Atol[i]), exact=exact, adaptive="order", step_size=2)
        err[i] = np.linalg.norm(ys[-1] - exact(tf))

        k_acc = [2*k for k in k_acc]
        h_rej = []
        k_rej = []
        ts_hk = []

        for j in range(len(h_rej_)):
            if not (h_rej_[j] == []):
                h_rej += h_rej_[j]
                k_rej += [2*k for k in k_rej_[j]]
                ts_hk += len(h_rej_[j])*[ts[j]]
        
        plt.hold('true')

        plt.subplot(311)
        ts_ = np.linspace(t0, tf, 100)
        plt.plot(ts, ys, 's-')
        ys_ = np.hstack(exact(ts_))
        plt.plot(ts_, ys_)
        plt.xlabel('t')
        plt.ylabel('Calculated y(t)')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,y1 - 0.1*abs((y2- y1)/2),y2 + 0.1*abs((y2- y1)/2)))
        
        plt.title(title + " with Atol = " + str(Atol[i]))

        plt.subplot(312)
        plt.semilogy(ts[:-1], h_acc, 's-')
        plt.semilogy(ts_hk, h_rej, 'ro')
        plt.ylabel('Step size')
        _,_,y1,y2 = plt.axis()
        plt.axis((x1,x2,y1 - 0.1*max(abs(y1), 1),y2 + 0.1*max(abs(y1), 1)))
        
        plt.subplot(313)
        plt.plot(ts[:-1], k_acc, 's-')
        plt.plot(ts_hk, k_rej, 'ro')
        plt.ylabel('order')
        _,_,y1,y2 = plt.axis()
        plt.axis((x1,x2,y1 - 0.1*max(abs(y1), 1),y2 + 0.1*max(abs(y1), 1)))

        plt.show()


def tst(f, t0, tf, exact, test_name):
    tst_adaptive_order(f, t0, tf, exact(t0), 4, exact, exs.ex_midpoint_serial,
        title=(test_name + ": adaptive order (Midpoint)"))

    for i in range(2, 6):    
        tst_convergence(f, t0, tf, exact(t0), i, exact, exs.ex_euler_serial,
            title=(test_name + ": fixed step (Euler)"))
        tst_adaptive_step(f, t0, tf, exact(t0), i, exact, exs.ex_euler_serial,
            title=(test_name + ": adaptive step (Euler)"))
    for i in range(4, 12, 2):    
        tst_convergence(f, t0, tf, exact(t0), i, exact, exs.ex_midpoint_serial,
            title=(test_name + ": fixed step (Midpoint)"))
        tst_adaptive_step(f, t0, tf, exact(t0), i, exact, exs.ex_midpoint_serial,
            title=(test_name + ": adaptive step (Midpoint)"))


def test1():
    lam = -1j
    y0 = np.array([1 + 0j])
    f = lambda y,t: lam*y
    exact = lambda t: y0*np.exp(lam*t)
    t0 = 0
    tf = 10
    tst(f, t0, tf, exact, "TEST 1")

def test2():
    f = lambda y,t: 4.*y*float(np.sin(t))**3*np.cos(t)
    y0 = np.array([1])
    exact = lambda t: y0*np.exp((np.sin(t))**4)
    t0 = 0
    tf = 10
    tst(f, t0, tf, exact, "TEST 2")

def test3():
    f = lambda y,t: 4.*t*np.sqrt(y)
    y0 = np.array([1])
    exact = lambda t: np.array([(1.+t**2)**2])
    t0 = 0
    tf = 10
    tst(f, t0, tf, exact, "TEST 3")

def test4():
    f = lambda y,t:  y/t*np.log(y)
    y0 = np.array([np.exp(1.)])
    exact = lambda t: np.array([np.exp(2.*t)])
    t0 = 0.5
    tf = 10
    tst(f, t0, tf, exact, "TEST 4")



if __name__ == "__main__":
    import doctest
    doctest.testmod()

