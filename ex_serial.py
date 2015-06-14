from __future__ import division
import numpy as np
import math 

def ex_one_step(f, tn, yn, h, p):
    Y = np.zeros((p+1,p+1, len(yn)), dtype=(type(yn[0])))
    T = np.zeros((p+1,p+1, len(yn)), dtype=(type(yn[0])))
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

def adjust_step(f, tn_1, yn_1, y, y_hat, h, p, Atol, Rtol):
    '''
        Checks if the step size is accepted. If not, computes a new step size
        and checks again. Repeats until step size is accepted 

        **Inputs**:
            - f         -- the right hand side function of the IVP.
                        Must output a non-scalar numpy.ndarray
            - tn_1, yn_1 -- y(tn_1) = yn_1 is the last accepted value of the 
                        computed solution 
            - y, y_hat  -- the computed values of y(tn_1 + h) of order p and 
                        (p-1), respectively
            - h         -- the step size taken and to be tested
            - p         -- the order of higher extrapolation method
                        Assumed to be greater than 1.
            - Atol, Rtol -- the absolute and relative tolerance of the local 
                         error.

        **Outputs**:
            - y, y_hat  -- the computed solution of orders p and (p-1) at the
                        accepted step size
            - h         -- the accepted step taken to compute y and y_hat
            - h_new     -- the proposed next step size
            - fe        -- the number of f evaluations 

    ''' 

    fe = 0
    facmax = 5
    facmin = 0.2
    fac = 0.8
    tol = Atol + np.maximum(y,y_hat)*Rtol
    err = np.linalg.norm((y-y_hat)/tol)/(len(y)**0.5)
    h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    while err > 1:
        h = h_new
        y, y_hat, fe_ = ex_one_step(f, tn_1, yn_1, h, p)
        fe = fe + fe_
        err = np.linalg.norm((y-y_hat)/tol)/(len(y)**0.5)
        h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    return (y, y_hat, h, h_new, fe)

def ex_euler_serial(f, t0, tf, y0, p, adaptive=True, step_size=0.5, Atol=0, 
        Rtol=0, exact=(lambda t: t)):
    '''
        Solves the system of IVPs y'(t) = f(y, t) with extrapolation of order 
        p based on Euler's method. The implementation is serial.

        **Inputs**:
            - f         -- the right hand side function of the IVP
                        Must output a non-scalar numpy.ndarray
            - [t0, tf]  -- the interval of integration
            - y0        -- the value of y(t0). Must be a non-scalar numpy.ndarray
            - p         -- the order of extrapolation. 
                        Must be greater than 1 when adaptive=True
            - adaptive  -- to use adaptive or fixed step size.
                        optional; defaults to True
            - step_size -- the starting step size when adaptive=True, or the 
                        fixed step size otherwise. optional; defaults to 0.5
            - Atol, Rtol -- the absolute and relative tolerance of the local 
                         error. optional; both defaults to 0
            - exact     -- the exact solution to the IVP. Only used for 
                        debugging. optional; defaults to (lambda t: t).
                        Must output a non-scalar numpy.ndarray

        **Outputs**:
            - ts, ys    -- the computed solution y for the IVP. y(ts[i]) = ys[i]
            - fe        -- the number of f evaluations 

    '''
    fe = 0
    if not adaptive:
        ts, h = np.linspace(t0, tf, (tf-t0)/step_size + 1, retstep=True)
        ys = np.zeros((len(ts), len(y0)), dtype=(type(y0[0])))
        ys_hat = np.zeros((len(ts), len(y0)), dtype=(type(y0[0])))
        ys[0] = y0

        for i in range(len(ts) - 1):
            ys[i+1], ys_hat[i+1], fe_ = ex_one_step(f, ts[i], ys[i], h, p)
            fe = fe + fe_
    else:
        assert p > 1, "order of method must be greater than 1 if adaptive=True"
        ts = np.zeros((1, len(y0)), dtype=(type(y0[0])))
        ys = np.zeros((1, len(y0)), dtype=(type(y0[0])))
        ys_hat = np.zeros((1, len(y0)), dtype=(type(y0[0])))
        ts[0] = t0
        ys[0] = y0
        ys_hat[0] = y0
        h = min(step_size, tf-t0)

        t, i = t0, 0
        while t < tf:
            y, y_hat, fe_ = ex_one_step(f, ts[i], ys[i], h, p)
            fe = fe + fe_
            y, y_hat, h, h_new, fe_ = adjust_step(f, ts[i], ys[i], y, y_hat, h, p, Atol, Rtol)
            t, i, fe = t + h, i+1, fe + fe_
            ts = np.append(ts, t)
            ys = np.vstack((ys, y))
            ys_hat =np.vstack((ys_hat, y_hat))
            h = min(h_new, tf - t)

    return (ts, ys, fe)



if __name__ == "__main__":
    import doctest
    doctest.testmod()

