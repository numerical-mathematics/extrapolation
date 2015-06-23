from __future__ import division
import numpy as np
import multiprocessing as mp
import math

def error_norm(y1, y2, Atol, Rtol):
    tol = Atol + np.maximum(y1,y2)*Rtol
    return np.linalg.norm((y1-y2)/tol)/(len(y1)**0.5)

def adapt_step(method, f, tn_1, yn_1, y, y_hat, h, p, Atol, Rtol, pool):
    '''
        Checks if the step size is accepted. If not, computes a new step size
        and checks again. Repeats until step size is accepted 

        **Inputs**:
            - method    -- the method on which the extrapolation is based
            - f         -- the right hand side function of the IVP.
                        Must output a non-scalar numpy.ndarray
            - tn_1, yn_1 -- y(tn_1) = yn_1 is the last accepted value of the 
                        computed solution 
            - y, y_hat  -- the computed values of y(tn_1 + h) of order p and 
                        (p-1), respectively
            - h         -- the step size taken and to be tested
            - p         -- the order of the higher extrapolation method
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
    err = error_norm(y, y_hat, Atol, Rtol)
    h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    while err > 1:
        h = h_new
        y, y_hat, fe_ = method(f, tn_1, yn_1, h, p, pool)
        fe += fe_
        err = error_norm(y, y_hat, Atol, Rtol)
        h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    return (y, y_hat, h, h_new, fe)

def extrapolation_parallel(method, f, t0, tf, y0, adaptive="order", p=4, 
        step_size=0.5, Atol=0, Rtol=0, exact=(lambda t: t)):
    '''
        Solves the system of IVPs y'(t) = f(y, t) with extrapolation of order 
        p based on method provided. The implementation is parallel.

        **Inputs**:
            - method    -- the method on which the extrapolation is based
            - f         -- the right hand side function of the IVP
                        Must output a non-scalar numpy.ndarray
            - [t0, tf]  -- the interval of integration
            - y0        -- the value of y(t0). Must be a non-scalar numpy.ndarray
            - adaptive  -- can be either of three values:
                            "fixed" = use fixed step size and order.
                            "step"  = use adaptive step size but fixed order.
                            "order" = use adaptive step size and adaptive order.
                        optional; defaults to "order"
            - p         -- the order of extrapolation if adaptive is not "order",
                        and the starting order otherwise. optional; defaults to 4
            - step_size -- the fixed step size when adaptive="fixed", and the
                        starting step size otherwise. optional; defaults to 0.5
            - Atol, Rtol -- the absolute and relative tolerance of the local 
                         error. optional; both default to 0
            - exact     -- the exact solution to the IVP. Only used for 
                        debugging. optional; defaults to (lambda t: t).
                        Must output a non-scalar numpy.ndarray

        **Outputs**:
            - y         -- the computed solution for y(tf)
            - fe        -- the number of f evaluations 
    '''

    try:
        processes = mp.cpu_count()
    except NotImplementedError:
        processes = 4

    pool = mp.Pool(processes)

    fe = 0
    if adaptive == "fixed":
        ts, h = np.linspace(t0, tf, (tf-t0)/step_size + 1, retstep=True)
        y = y0

        for i in range(len(ts) - 1):
            y, _, fe_ = method(f, ts[i], y, h, p, pool)
            fe += fe_
    
    elif adaptive == "step":
        assert p > 1, "order of method must be greater than 1 if adaptive=True"
        y, t = y0, t0
        h = min(step_size, tf-t0)

        while t < tf:
            y_, y_hat, fe_ = method(f, t, y, h, p, pool)
            fe += fe_
            y, _, h, h_new, fe_ = adapt_step(method, f, t, y, y_, y_hat, h, p, 
                Atol, Rtol, pool)
            t, fe = t + h, fe + fe_
            h = min(h_new, tf - t)
    
    elif adaptive == "order":
        y, t, k = y0, t0, p
        h = min(step_size, tf-t0)

        while t < tf:
            y, h, k, h_new, k_new, _, _, fe_ = method(f, t, y, h, k, Atol, Rtol, pool)
            t, fe = t + h, fe + fe_
            h = min(h_new, tf - t)
            k = k_new

    else:
        raise Exception("\'" + str(adaptive) + 
            "\' is not a valid value for the argument \'adaptive\'")

    pool.close()
    return (y, fe)

def compute_stages((f, tn, yn, h, k_lst)):
    res = []
    for k in k_lst:
        k = int(k)
        Y = np.zeros((2*k+1, len(yn)), dtype=(type(yn[0])))
        fe = 0
        Y[0] = yn
        Y[1] = Y[0] + h/(2*k)*f(Y[0], tn)
        for j in range(2,2*k+1):
            Y[j] = Y[j-2] + (h/k)*f(Y[j-1], tn + (j-1)*(h/(2*k)))
            fe += 1
        res += [(k, Y[2*k], fe)]

    return res

def compute_ex_table(f, tn, yn, h, k, pool):
    T = np.zeros((k+1,k+1, len(yn)), dtype=(type(yn[0])))
    fe = 0

    if k % 2 == 1:
        ks = [[k]] + [[i, k - i] for i in range(1, int(k/2)+1)]
    else:
        ks = [[k]] + [[i, k - i] for i in range(1, int(k/2))] + [[k/2]]

    jobs = [(f, tn, yn, h, k_lst) for k_lst in ks]

    results = pool.map(compute_stages, jobs, chunksize=1)

    # process the returned results from the pool 
    for res in results:
        for (k_, Tk_, fe_) in res:
            fe += fe_
            T[k_, 1] = Tk_

    # compute extrapolation table 
    for i in range(2, k+1):
        for j in range(i, k+1):
            T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((j/(j-i+1))**2 - 1)

    return (T, fe)

def midpoint_fixed_step(f, tn, yn, h, p, pool):
    r = int(round(p/2))
    T, fe = compute_ex_table(f, tn, yn, h, r, pool)
    return (T[r,r], T[r-1,r-1], fe)

def midpoint_adapt_order(f, tn, yn, h, k, Atol, Rtol, pool):
    k_max = 10
    k_min = 3
    k = min(k_max, max(k_min, k))
    A_k = lambda k: k*(k+1)
    H_k = lambda h, k, err_k: h*0.94*(0.65/err_k)**(1/(2*k-1)) 
    W_k = lambda Ak,Hk: Ak/Hk
    h_rej = []
    k_rej = []

    T, fe = compute_ex_table(f, tn, yn, h, k+1, pool)

    # compute the error and work function for the stages (k-2) to (k+1)
    err_k_2 = error_norm(T[k-2,k-3], T[k-2,k-2], Atol, Rtol)
    err_k_1 = error_norm(T[k-1,k-2], T[k-1,k-1], Atol, Rtol)
    err_k   = error_norm(T[k,k-1],   T[k,k],     Atol, Rtol)
    err_k1  = error_norm(T[k+1,k],   T[k+1,k+1], Atol, Rtol)
    h_k_2   = H_k(h, k-2, err_k_2)
    h_k_1   = H_k(h, k-1, err_k_1)
    h_k     = H_k(h, k,   err_k)
    h_k1    = H_k(h, k+1, err_k1)
    w_k_2   = W_k(A_k(k-2), h_k_2)
    w_k_1   = W_k(A_k(k-1), h_k_1)
    w_k     = W_k(A_k(k),   h_k)
    w_k1    = W_k(A_k(k+1), h_k1)

    if err_k_1 <= 1:
        # convergence in line k-1
        y = T[k-1,k-1]
        k_new = k if w_k_1 < 0.9*w_k_2 else k-1
        h_new = h_k_1 if k_new <= k-1 else h_k_1*A_k(k)/A_k(k-1)

    elif err_k <= 1:
        # convergence in line k
        y = T[k,k]
        k_new = k-1 if w_k_1 < 0.9*w_k else (
                k+1 if w_k < 0.9*w_k_1 else k)
        h_new = h_k_1 if k_new == k-1 else (
                h_k if k_new == k else h_k*A_k(k+1)/A_k(k))

    elif err_k1 <= 1:
        # convergence in line k+1
        y = T[k+1,k+1]
        if w_k_1 < 0.9*w_k:
            k_new = k+1 if w_k1 < 0.9*w_k_1 else k-1
        else:
            k_new = k+1 if w_k1 < 0.9*w_k else k

        h_new = h_k_1 if k_new == k-1 else (
                h_k if k_new == k else h_k1)

    elif err_k_1 > ((k+1)*k)**2:
        # convergence monitor
        # reject (h, k) and restart with new values accordingly
        k_new = k-1
        h_new = min(h_k_1, h)
        h_rej.append(h)
        k_rej.append(k)
        y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, tn, 
            yn, h_new, k_new, Atol, Rtol, pool)
        fe += fe_
        h_rej += h_rej_
        k_rej += k_rej_

    elif err_k > (k+1)**2:
        # second convergence monitor 
        # reject (h, k) and restart with new values accordingly
        k_new = k-1 if w_k_1 < 0.9*w_k else k
        h_new = min(h_k_1 if k_new == k-1 else h_k, h)
        h_rej.append(h)
        k_rej.append(k)
        y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, 
            tn, yn, h_new, k_new, Atol, Rtol, pool)
        fe += fe_
        h_rej += h_rej_
        k_rej += k_rej_

    else: 
        # no convergence
        # reject (h, k) and restart with new values accordingly
        k_new = k-1 if w_k_1 < 0.9*w_k else k
        h_new = min(h_k_1 if k_new == k-1 else h_k, h)
        h_rej.append(h)
        k_rej.append(k)
        y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, 
            tn, yn, h_new, k_new, Atol, Rtol, pool)
        fe += fe_
        h_rej += h_rej_
        k_rej += k_rej_

    return (y, h, k, h_new, k_new, h_rej, k_rej, fe)

def ex_midpoint_parallel(f, t0, tf, y0, adaptive="order", p=4, step_size=0.5, Atol=0, 
        Rtol=0, exact=(lambda t: t)):
    '''
        An instantiation of extrapolation_serial() function with the midpoint method.
        For more details, refer to extrapolation_serial() function.
    '''

    method = midpoint_adapt_order if adaptive == "order" else midpoint_fixed_step

    return extrapolation_parallel(method, f, t0, tf, y0,
        adaptive=adaptive, p=p, step_size=step_size, Atol=Atol, Rtol=Rtol, 
        exact=exact)