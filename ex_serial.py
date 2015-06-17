from __future__ import division
import numpy as np
import math 

def error_norm(y1, y2, Atol, Rtol):
    tol = Atol + np.maximum(y1,y2)*Rtol
    return np.linalg.norm((y1-y2)/tol)/(len(y1)**0.5)

def adapt_step(method, f, tn_1, yn_1, y, y_hat, h, p, Atol, Rtol):
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
        y, y_hat, fe_ = method(f, tn_1, yn_1, h, p)
        fe = fe + fe_
        err = error_norm(y, y_hat, Atol, Rtol)
        h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    return (y, y_hat, h, h_new, fe)

def extrapolation_serial(method, f, t0, tf, y0, adaptive="order", p=4, 
        step_size=0.5, Atol=0, Rtol=0, exact=(lambda t: t)):
    '''
        Solves the system of IVPs y'(t) = f(y, t) with extrapolation of order 
        p based on method provided. The implementation is serial.

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
            - ts, ys    -- the computed solution y for the IVP. y(ts[i]) = ys[i]
            - fe        -- the number of f evaluations 
    '''

    fe = 0
    if adaptive == "fixed":
        ts, h = np.linspace(t0, tf, (tf-t0)/step_size + 1, retstep=True)
        ys = np.zeros((len(ts), len(y0)), dtype=(type(y0[0])))
        ys_hat = np.zeros((len(ts), len(y0)), dtype=(type(y0[0])))
        ys[0] = y0

        for i in range(len(ts) - 1):
            ys[i+1], ys_hat[i+1], fe_ = method(f, ts[i], ys[i], h, p)
            fe = fe + fe_
    
    elif adaptive == "step":
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
            y, y_hat, fe_ = method(f, ts[i], ys[i], h, p)
            fe = fe + fe_
            y, y_hat, h, h_new, fe_ = adapt_step(method, f, ts[i], ys[i], 
                y, y_hat, h, p, Atol, Rtol)
            t, i, fe = t + h, i+1, fe + fe_
            ts = np.append(ts, t)
            ys = np.vstack((ys, y))
            ys_hat =np.vstack((ys_hat, y_hat))
            h = min(h_new, tf - t)
    
    elif adaptive == "order":
        ts = np.zeros((1, len(y0)), dtype=(type(y0[0])))
        ys = np.zeros((1, len(y0)), dtype=(type(y0[0])))
        ts[0] = t0
        ys[0] = y0
        h = min(step_size, tf-t0)

        h_acc = []
        k_acc = []
        h_rej = []
        k_rej = []

        t, i, k = t0, 0, p
        while t < tf:
            y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = method(f, ts[i], ys[i], 
                h, k, Atol, Rtol)
            t, i, fe = t + h, i+1, fe + fe_
            ts = np.append(ts, t)
            ys = np.vstack((ys, y))
            h_acc.append(h)
            k_acc.append(k)
            h_rej.append(h_rej_)
            k_rej.append(k_rej_)
            h = min(h_new, tf - t)
            k = k_new

        return (ts, ys, fe, h_acc, k_acc, h_rej, k_rej)
    else:
        raise Exception("\'" + str(adaptive) + 
            "\' is not a valid value for the argument \'adaptive\'")

    return (ts, ys, fe)

def euler_fixed_step(f, tn, yn, h, p):
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

def midpoint_fixed_step(f, tn, yn, h, p):
    r = int(round(p/2))
    Y = np.zeros((r+1,2*r+1, len(yn)), dtype=(type(yn[0])))
    T = np.zeros((r+1,r+1, len(yn)), dtype=(type(yn[0])))
    fe = 0
    for k in range(1,r+1):
        Y[k,0] = yn
        Y[k,1] = Y[k,0] + h/(2*k)*f(Y[k,0], tn)
        for j in range(2,2*k+1):
            Y[k,j] = Y[k,j-2] + (h/k)*f(Y[k,j-1], tn + (j-1)*(h/(2*k)))
            fe = fe + 1
        T[k,1] = Y[k,2*k]

    for k in range(2, r+1):
        for j in range(k, r+1):
            T[j,k] = T[j,k-1] + (T[j,k-1] - T[j-1,k-1])/((j/(j-k+1))**2 - 1)

    return (T[r,r], T[r-1,r-1], fe)


def midpoint_adapt_order(f, tn, yn, h, k, Atol, Rtol):
    k_max = 8
    k_min = 3
    k = min(k_max, max(k_min, k))
    A_k = lambda k: k*(k+1)
    H_k = lambda h, k, err_k: h*0.94*(0.65/err_k)**(1/(2*k-1)) 
    W_k = lambda Ak,Hk: Ak/Hk

    Y = np.zeros((k+2,2*(k+1)+1, len(yn)), dtype=(type(yn[0])))
    T = np.zeros((k+2,k+2, len(yn)), dtype=(type(yn[0])))
    fe = 0

    h_rej = []
    k_rej = []

    # compute the first k-1 lines extrapolation tableau
    for i in range(1,k):
        Y[i,0] = yn
        Y[i,1] = Y[i,0] + h/(2*i)*f(Y[i,0], tn)
        for j in range(2,2*i+1):
            Y[i,j] = Y[i,j-2] + (h/i)*f(Y[i,j-1], tn + (j-1)*(h/(2*i)))
            fe = fe + 1
        T[i,1] = Y[i,2*i]

    for i in range(2, k):
        for j in range(i, k):
            T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((j/(j-i+1))**2 - 1)

    err_k_2 = error_norm(T[k-2,k-3], T[k-2,k-2], Atol, Rtol)
    err_k_1 = error_norm(T[k-1,k-2], T[k-1,k-1], Atol, Rtol)
    h_k_2 = H_k(h, k-2, err_k_2)
    h_k_1 = H_k(h, k-1, err_k_1)
    w_k_2 = W_k(A_k(k-2), h_k_2)
    w_k_1 = W_k(A_k(k-1), h_k_1)

    if err_k_1 <= 1:
        # convergence in line k-1
        y = T[k-1,k-1]
        k_new = k if w_k_1 < 0.9*w_k_2 else k-1
        h_new = h_k_1 if k_new <= k-1 else h_k_1*A_k(k)/A_k(k-1)

    elif err_k_1 > ((k+1)*k)**2:
        # convergence monitor
        # reject (h, k) and restart with new values accordingly
        k_new = k-1
        h_new = min(h_k_1, h)
        h_rej.append(h)
        k_rej.append(k)
        y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, tn, 
            yn, h_new, k_new, Atol, Rtol)
        fe = fe + fe_
        h_rej.append(h_rej_)
        k_rej.append(k_rej_)

    else:
        # compute line k of extrapolation tableau
        Y[k,0] = yn
        Y[k,1] = Y[k,0] + h/(2*k)*f(Y[k,0], tn)
        for j in range(2,2*k+1):
            Y[k,j] = Y[k,j-2] + (h/k)*f(Y[k,j-1], tn + (j-1)*(h/(2*k)))
            fe = fe + 1
        T[k,1] = Y[k,2*k]

        for i in range(2, k+1):
            T[k,i] = T[k,i-1] + (T[k,i-1] - T[k-1,i-1])/((k/(k-i+1))**2 - 1)

        err_k = error_norm(T[k,k-1], T[k,k], Atol, Rtol)
        h_k = H_k(h, k, err_k)
        w_k = W_k(A_k(k), h_k)
        
        if err_k <= 1:
            # convergence in line k
            y = T[k,k]
            k_new = k-1 if w_k_1 < 0.9*w_k else (
                    k+1 if w_k < 0.9*w_k_1 else k)
            h_new = h_k_1 if k_new == k-1 else (
                    h_k if k_new == k else h_k*A_k(k+1)/A_k(k))
        elif err_k > (k+1)**2:
            # second convergence monitor 
            # reject (h, k) and restart with new values accordingly
            k_new = k-1 if w_k_1 < 0.9*w_k else k
            h_new = min(h_k_1 if k_new == k-1 else h_k, h)
            h_rej.append(h)
            k_rej.append(k)
            y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, 
                tn, yn, h_new, k_new, Atol, Rtol)
            fe = fe + fe_
            h_rej.append(h_rej_)
            k_rej.append(k_rej_)

        else: 
            # hope for convergence in line k+1
            # compute line k of extrapolation tableau 
            Y[(k+1),0] = yn
            Y[(k+1),1] = Y[(k+1),0] + h/(2*(k+1))*f(Y[(k+1),0], tn)
            for j in range(2,2*(k+1)+1):
                Y[(k+1),j] = Y[(k+1),j-2] + (h/(k+1))*f(Y[(k+1),j-1], tn + (j-1)*(h/(2*(k+1))))
                fe = fe + 1
            T[(k+1),1] = Y[(k+1),2*(k+1)]

            for i in range(2, (k+1)+1):
                T[(k+1),i] = T[(k+1),i-1] + (T[(k+1),i-1] - T[(k+1)-1,i-1])/(((k+1)/((k+1)-i+1))**2 - 1)

            err_k1 = error_norm(T[(k+1),(k+1)-1], T[(k+1),(k+1)], Atol, Rtol)
            h_k1 = H_k(h, (k+1), err_k1)
            w_k1 = W_k(A_k((k+1)), h_k1)

            if err_k1 <= 1:
                # convergence in line k+1
                y = T[k+1,k+1]
                if w_k_1 < 0.9*w_k:
                    k_new = k+1 if w_k1 < 0.9*w_k_1 else k-1
                else:
                    k_new = k+1 if w_k1 < 0.9*w_k else k

                h_new = h_k_1 if k_new == k-1 else (
                        h_k if k_new == k else h_k1)
            else: 
                # no convergence
                # reject (h, k) and restart with new values accordingly
                k_new = k-1 if w_k_1 < 0.9*w_k else k
                h_new = min(h_k_1 if k_new == k-1 else h_k, h)
                h_rej.append(h)
                k_rej.append(k)
                y, h, k, h_new, k_new, h_rej_, k_rej_, fe_ = midpoint_adapt_order(f, 
                    tn, yn, h_new, k_new, Atol, Rtol)
                fe = fe + fe_
                h_rej.append(h_rej_)
                k_rej.append(k_rej_)

    return (y, h, k, h_new, k_new, h_rej, k_rej, fe)

def ex_euler_serial(f, t0, tf, y0, adaptive="order", p=4, step_size=0.5, Atol=0, 
        Rtol=0, exact=(lambda t: t)):
    '''
        An instantiation of extrapolation_serial() function with Euler's method.
        For more details, refer to extrapolation_serial() function.
    
        TODO: implement adaptive order extrapolation based on Euler
    '''
    method = euler_fixed_step

    if adaptive == "order":
        # not implemented yet, so change to adaptive step for now 
        adaptive = "step"

    return extrapolation_serial(method, f, t0, tf, y0, 
        adaptive=adaptive, p=p, step_size=step_size, Atol=Atol, Rtol=Rtol, 
        exact=exact)

def ex_midpoint_serial(f, t0, tf, y0, adaptive="order", p=4, step_size=0.5, Atol=0, 
        Rtol=0, exact=(lambda t: t)):
    '''
        An instantiation of extrapolation_serial() function with the midpoint method.
        For more details, refer to extrapolation_serial() function.
    '''

    method = midpoint_adapt_order if adaptive == "order" else midpoint_fixed_step

    return extrapolation_serial(method, f, t0, tf, y0,
        adaptive=adaptive, p=p, step_size=step_size, Atol=Atol, Rtol=Rtol, 
        exact=exact)
    


if __name__ == "__main__":
    import doctest
    doctest.testmod()

