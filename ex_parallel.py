from __future__ import division
import numpy as np
import multiprocessing as mp
import math

try:
    NUM_WORKERS = mp.cpu_count()
except NotImplementedError:
    NUM_WORKERS = 4


def error_norm(y1, y2, Atol, Rtol):
    tol = Atol + np.maximum(y1,y2)*Rtol
    return np.linalg.norm((y1-y2)/tol)/(len(y1)**0.5)

def adapt_step(method, f, tn_1, yn_1, y, y_hat, h, p, Atol, Rtol, pool, seq=(lambda t: 2*t), dense=False):
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
            - seq       -- the step-number sequence. optional; defaults to the 
                        harmonic sequence given by (lambda t: 2*t)

        **Outputs**:
            - y, y_hat  -- the computed solution of orders p and (p-1) at the
                        accepted step size
            - h         -- the accepted step taken to compute y and y_hat
            - h_new     -- the proposed next step size
            - (fe_seq, fe_tot) -- the number of sequential f evaluations, and
                                    the total number of f evaluations
    ''' 

    facmax = 5
    facmin = 0.2
    fac = 0.8
    err = error_norm(y, y_hat, Atol, Rtol)
    h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    fe_seq = 0
    fe_tot = 0

    while err > 1:
        h = h_new
        if dense:
            y, y_hat, (fe_seq_, fe_tot_), poly = method(f, tn_1, yn_1, h, p, pool, seq=seq, dense=dense)
        else:
            y, y_hat, (fe_seq_, fe_tot_) = method(f, tn_1, yn_1, h, p, pool, seq=seq, dense=dense)
        fe_seq += fe_seq_
        fe_tot += fe_tot_
        err = error_norm(y, y_hat, Atol, Rtol)
        h_new = h*min(facmax, max(facmin, fac*((1/err)**(1/p))))

    if dense:
        return (y, y_hat, h, h_new, (fe_seq, fe_tot), poly)
    else:
        return (y, y_hat, h, h_new, (fe_seq, fe_tot))

def extrapolation_parallel(method, f, t0, tf, y0, adaptive="order", p=4,
        seq=(lambda t: 2*t), dense=False, step_size=0.5, Atol=0, Rtol=0,
        exact=(lambda t: t)):
    '''
        Solves the system of IVPs y'(t) = f(y, t) with extrapolation of order 
        p based on method provided. The implementation is parallel.

        **Inputs**:
            - method    -- the method on which the extrapolation is based
            - f         -- the right hand side function of the IVP
                        Must output a non-scalar numpy.ndarray
            - t0        -- the time of the initial value
            - tf        -- if dense==True, tf is a list of times for which the 
                        solution is computed at. In this case, the values in tf
                        must be in an increasing order and all are strictly 
                        greater than t0. if dense==False, then tf is a single 
                        float number for which the solution is computed at.
            - y0        -- the value of y(t0). Must be a non-scalar numpy.ndarray
            - adaptive  -- can be either of three values:
                            "fixed" = use fixed step size and order.
                            "step"  = use adaptive step size but fixed order.
                            "order" = use adaptive step size and adaptive order.
                        optional; defaults to "order"
            - p         -- the order of extrapolation if adaptive is not "order",
                        and the starting order otherwise. optional; defaults to 4
            - seq       -- the step-number sequence. optional; defaults to the 
                        harmonic sequence given by (lambda t: 2*t)
            - dense     -- when set, the dense output routine runs. 
                        optional; defaults to False
            - step_size -- the fixed step size when adaptive="fixed", and the
                        starting step size otherwise. optional; defaults to 0.5
            - Atol, Rtol -- the absolute and relative tolerance of the local 
                         error. optional; both default to 0
            - exact     -- the exact solution to the IVP. Only used for 
                        debugging. optional; defaults to (lambda t: t).
                        Must output a non-scalar numpy.ndarray

        **Outputs**:
            - y [or ys]        -- the computed solution for y(tf)
            - (fe_seq, fe_tot) -- the number of sequential f evaluations, and
                                    the total number of f evaluations
    '''

    pool = mp.Pool(NUM_WORKERS)

    fe_seq = 0
    fe_tot = 0
    if adaptive == "fixed":
        ts, h = np.linspace(t0, tf, (tf-t0)/step_size + 1, retstep=True)
        y = 1*y0

        for i in range(len(ts) - 1):
            if dense:
                y, _, (fe_seq_, fe_tot_), poly = method(f, ts[i], y, h, p, pool, seq=seq, dense=dense)
            else:
                y, _, (fe_seq_, fe_tot_) = method(f, ts[i], y, h, p, pool, seq=seq, dense=dense)
            fe_seq += fe_seq_
            fe_tot += fe_tot_
    
    elif adaptive == "step":
        assert p > 1, "order of method must be greater than 1 if adaptive=step"
        if dense:
            t_max = tf[-1]
            t_index = 0
            ys = len(tf)*[None]
        else:
            t_max = tf

        y, t = 1*y0, t0
        h = min(step_size, t_max-t0)

        while t < t_max:
            if dense:
                y_, y_hat, (fe_seq_, fe_tot_), poly = method(f, t, y, h, p, pool, seq=seq, dense=dense)
            else:
                y_, y_hat, (fe_seq_, fe_tot_) = method(f, t, y, h, p, pool, seq=seq, dense=dense)
            
            fe_seq += fe_seq_
            fe_tot += fe_tot_

            if dense:
                reject_inter = True
                while reject_inter:
                    y_temp, _, h, h_new, (fe_seq_, fe_tot_), poly = adapt_step(method, f, t, 
                        y, y_, y_hat, h, p, Atol, Rtol, pool, seq=seq, dense=dense)
                    reject_inter = False
                    while t_index < len(tf) and tf[t_index] <= t + h:
                        ys_, errint, h_int = poly((tf[t_index] - t)/h)
                        
                        if errint <= 10:
                            y = 1*y_temp
                            ys[t_index] = 1*ys_
                            t_index += 1
                            reject_inter = False
                        else:
                            h = h_int
                            fe_seq += fe_seq_
                            fe_tot += fe_tot_
                            reject_inter = True
                            break
            else:
                y, _, h, h_new, (fe_seq_, fe_tot_) = adapt_step(method, f, t, 
                    y, y_, y_hat, h, p, Atol, Rtol, pool, seq=seq, dense=dense)
            
            t += h
            fe_seq += fe_seq_
            fe_tot += fe_tot_
            h = min(h_new, t_max - t)
    
    elif adaptive == "order":
        if dense:
            t_max = tf[-1]
            t_index = 0
            ys = len(tf)*[None]
        else:
            t_max = tf

        y, t, k = 1*y0, t0, p
        h = min(step_size, t_max-t0)

        sum_ks, sum_hs = 0, 0
        num_iter = 0

        while t < t_max:
            if dense:
                reject_inter = True
                while reject_inter:
                    y_temp, h, k, h_new, k_new, (fe_seq_, fe_tot_), poly = method(f, t, y, h, 
                        k, Atol, Rtol, pool, seq=seq, dense=dense)

                    reject_inter = False
                    old_index = t_index
                    while t_index < len(tf) and tf[t_index] <= t + h:
                        ys_, errint, h_int = poly((tf[t_index] - t)/h)
                        
                        if errint <= 10:
                            ys[t_index] = 1*ys_
                            t_index += 1
                            reject_inter = False
                        else:
                            h = h_int
                            fe_seq += fe_seq_
                            fe_tot += fe_tot_
                            reject_inter = True
                            t_index = old_index
                            break

                    if not reject_inter:
                        y = 1*y_temp

            else:
                y, h, k, h_new, k_new, (fe_seq_, fe_tot_) = method(f, t, y, h, 
                    k, Atol, Rtol, pool, seq=seq, dense=dense)
            t += h
            fe_seq += fe_seq_
            fe_tot += fe_tot_

            sum_ks += k
            sum_hs += h
            num_iter += 1

            h = min(h_new, t_max - t)
            k = k_new

        pool.close()
        if dense:
            return (ys, (fe_seq, fe_tot), sum_hs/num_iter, sum_ks/num_iter)
        else:
            return (y, (fe_seq, fe_tot), sum_hs/num_iter, sum_ks/num_iter)
    else:
        raise Exception("\'" + str(adaptive) + 
            "\' is not a valid value for the argument \'adaptive\'")

    pool.close()

    if dense:
        return (ys, (fe_seq, fe_tot))
    else:
        return (y, (fe_seq, fe_tot))

def compute_stages_dense((f, tn, yn, h, k_nj_lst)):
    res = []
    for (k, nj) in k_nj_lst:
        nj = int(nj)
        Y = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        f_yj = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        Y[0] = yn
        f_yj[0] = f(Y[0], tn)
        Y[1] = Y[0] + h/nj*f_yj[0]
        for j in range(2,nj+1):
            if j == nj/2 + 1:
                y_half = Y[j-1]
            f_yj[j-1] = f(Y[j-1], tn + (j-1)*(h/nj))
            Y[j] = Y[j-2] + (2*h/nj)*f_yj[j-1]

        f_yj[nj] = f(Y[nj], tn + h)
        res += [(k, nj, Y[nj], y_half, f_yj)]

    return res

def compute_stages((f, tn, yn, h, k_nj_lst)):
    res = []
    for (k, nj) in k_nj_lst:
        nj = int(nj)
        Y = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        Y[0] = yn
        Y[1] = Y[0] + h/nj*f(Y[0], tn)
        for j in range(2,nj+1):
            Y[j] = Y[j-2] + (2*h/nj)*f(Y[j-1], tn + (j-1)*(h/nj))
        res += [(k, nj, Y[nj])]

    return res

def balance_load(k, seq=(lambda t: 2*t)):
    if k <= NUM_WORKERS:
        k_nj_lst = [[(i,seq(i))] for i in range(k, 0, -1)]
    else:
        k_nj_lst = [[] for i in range(NUM_WORKERS)]
        index = range(NUM_WORKERS)
        i = k
        while 1:
            if i >= NUM_WORKERS:
                for j in index:
                    k_nj_lst[j] += [(i, seq(i))]
                    i -= 1
            else:
                for j in index:
                    if i == 0:
                        break
                    k_nj_lst[j] += [(i, seq(i))]
                    i -= 1
                break
            index = index[::-1]

    fe_tot = 0 
    for i in range(len(k_nj_lst)):
        fe_tot += sum([pair[1] for pair in k_nj_lst[i]])
    
    fe_seq = sum([pair[1] for pair in k_nj_lst[0]])

    return (k_nj_lst, fe_seq, fe_tot)

def compute_ex_table(f, tn, yn, h, k, pool, seq=(lambda t: 2*t), dense=False):
    T = np.zeros((k+1,k+1, len(yn)), dtype=(type(yn[0])))
    k_nj_lst, fe_seq, fe_tot = balance_load(k, seq=seq)
    jobs = [(f, tn, yn, h, k_nj) for k_nj in k_nj_lst]

    if dense:
        results = pool.map(compute_stages_dense, jobs, chunksize=1)
    else:
        results = pool.map(compute_stages, jobs, chunksize=1)

    # process the returned results from the pool 
    if dense:
        y_half = (k+1)*[None]
        f_yj = (k+1)*[None]
        hs = (k+1)*[None]
        for res in results:
            for (k_, nj_, Tk_, y_half_, f_yj_) in res:
                T[k_, 1] = Tk_
                y_half[k_] = y_half_
                f_yj[k_] = f_yj_
                hs[k_] = h/nj_
    else:
        for res in results:
            for (k_, nj_, Tk_) in res:
                T[k_, 1] = Tk_

    # compute extrapolation table 
    for i in range(2, k+1):
        for j in range(i, k+1):
            T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j)/(seq(j-i+1)))**2 - 1)

    if dense:
        Tkk = T[k,k]
        f_Tkk = f(Tkk, tn+h)
        fe_seq +=1
        fe_tot +=1
        return (T, fe_seq, fe_tot, yn, Tkk, f_Tkk, y_half, f_yj, hs)
    else:
        return (T, fe_seq, fe_tot)


def finite_diff(j, f_yj, hj):
    max_order = 2*j
    nj = len(f_yj) - 1
    coeff = [1,1]
    dj = (max_order+1)*[None]
    dj[1] = 1*f_yj[nj/2]
    dj[2] = (f_yj[nj/2+1] - f_yj[nj/2-1])/(2*hj)
    for order in range(2,max_order):
        coeff = [1] + [coeff[j] + coeff[j+1] for j in range(len(coeff)-1)] + [1]
        index = [nj/2 + order - 2*i for i in range(order+1)]

        sum_ = 0
        for i in range(order+1):
            sum_ += ((-1)**i)*coeff[i]*f_yj[index[i]]
        dj[order+1] = sum_ / (2*hj)**order 

    return dj

def compute_ds(y_half, f_yj, hs, k, seq=(lambda t: 4*t-2)):
    dj_kappa = np.zeros((2*k+1, k+1), dtype=(type(y_half[1])))
    ds = np.zeros((2*k+1), dtype=(type(y_half[1])))
    
    for j in range(1,k+1):
        dj_kappa[0,j] = 1*y_half[j]
        nj = len(f_yj[j])-1
        dj_ = finite_diff(j,f_yj[j], hs[j])
        for kappa in range(1,2*j+1):    
            dj_kappa[kappa,j] = 1*dj_[kappa]

    skip = 0
    for kappa in range(2*k+1):
        T = np.zeros((k+1-int(skip/2), k+1 - int(skip/2)), dtype=(type(y_half[1])))
        T[:,1] = 1*dj_kappa[kappa, int(skip/2):]

        for i in range(2, k+1-int(skip/2)):
            for j in range(i, k+1-int(skip/2)):
                T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j)/(seq(j-i+1)))**2 - 1)
        ds[kappa] = 1*T[k-int(skip/2),k-int(skip/2)] 
        if not(kappa == 0):
            skip +=1

    return ds 

def interpolate(y0, Tkk, f_Tkk, y_half, f_yj, hs, H, k, Atol, Rtol, seq=(lambda t: 4*t-2)):
    u = 2*k-3
    u_1 = u - 1
    ds = compute_ds(y_half, f_yj, hs, k, seq=seq)
    
    a_u = (u+5)*[None]
    a_u_1 = (u_1+5)*[None]
    
    for i in range(u+1):
        a_u[i] = (H**i)*ds[i]/math.factorial(i)
    for i in range(u_1 + 1):
        a_u_1[i] = (H**i)*ds[i]/math.factorial(i)

    A_inv_u = (2**(u-2))*np.matrix(
                [[(-2*(3 + u))*(-1)**u,     -(-1)**u,        2*(3 + u),     -1], 
                 [(4*(4 + u))*(-1)**u,       2*(-1)**u,      4*(4 + u),     -2], 
                 [(8*(1 + u))*(-1)**u,       4*(-1)**u,     -8*(1 + u),      4], 
                 [(-16*(2 + u))*(-1)**u,    -8*(-1)**u,     -16*(2 + u),     8]]
                ) 

    A_inv_u_1 = (2**(u_1-2))*np.matrix(
                [[(-2*(3 + u_1))*(-1)**u_1,     -(-1)**u_1,        2*(3 + u_1),     -1], 
                 [(4*(4 + u_1))*(-1)**u_1,       2*(-1)**u_1,      4*(4 + u_1),     -2], 
                 [(8*(1 + u_1))*(-1)**u_1,       4*(-1)**u_1,     -8*(1 + u_1),      4], 
                 [(-16*(2 + u_1))*(-1)**u_1,    -8*(-1)**u_1,     -16*(2 + u_1),     8]]
                ) 

    b1_u = 1*y0
    for i in range(u+1):
        b1_u -= a_u[i]/(-2)**i

    b1_u_1 = 1*y0
    for i in range(u_1+1):
        b1_u_1 -= a_u_1[i]/(-2)**i

    b2_u = H*f_yj[1][0]
    for i in range(1, u+1):
        b2_u -= i*a_u[i]/(-2)**(i-1)

    b2_u_1 = H*f_yj[1][0]
    for i in range(1, u_1+1):
        b2_u_1 -= i*a_u_1[i]/(-2)**(i-1)

    b3_u = 1*Tkk
    for i in range(u+1):
        b3_u -= a_u[i]/(2**i)

    b3_u_1 = 1*Tkk
    for i in range(u_1+1):
        b3_u_1 -= a_u_1[i]/(2**i)

    b4_u = H*f_Tkk
    for i in range(1, u+1):
        b4_u -= i*a_u[i]/(2**(i-1))

    b4_u_1 = H*f_Tkk
    for i in range(1, u_1+1):
        b4_u_1 -= i*a_u_1[i]/(2**(i-1))

    b_u = np.array([b1_u,b2_u,b3_u,b4_u])
    b_u_1 = np.array([b1_u_1,b2_u_1,b3_u_1,b4_u_1])

    x = A_inv_u*b_u
    x = np.array(x)

    x_1 = A_inv_u_1*b_u_1
    x_1 = np.array(x)

    a_u[u+1] = x[0]
    a_u[u+2] = x[1]
    a_u[u+3] = x[2]
    a_u[u+4] = x[3]

    a_u_1[u_1+1] = x[0]
    a_u_1[u_1+2] = x[1]
    a_u_1[u_1+3] = x[2]
    a_u_1[u_1+4] = x[3]

    # polynomial of degree u+4 defined on [0,1] and centered about 1/2
    # also returns the interpolation error (errint). If errint > 10, then reject
    # step 
    def poly (t):
        res = 1*a_u[0] 
        for i in range(1, len(a_u)):
            res += a_u[i]*((t-0.5)**i)
        
        res_u_1 = 1*a_u_1[0] 
        for i in range(1, len(a_u_1)):
            res_u_1 += a_u_1[i]*((t-0.5)**i)
        
        errint = error_norm(res, res_u_1, Atol, Rtol)
        h_int = H*((1/errint)**(1/(u+4)))
        
        return (res, errint, h_int)

    return poly

def midpoint_fixed_step(f, tn, yn, h, p, pool, seq=(lambda t: 2*t), dense=False):
    r = int(round(p/2))
    if dense:
        T, fe_seq, fe_tot, y0, Tkk, f_Tkk, y_half, f_yj, hs = compute_ex_table(f, tn, yn, h, r, pool, seq=seq, dense=dense)
        poly = interpolate(y0, Tkk, f_Tkk, y_half, f_yj, hs, h, r, Atol, Rtol, seq=seq)
        return (T[r,r], T[r-1,r-1], (fe_seq, fe_tot), poly)
    else:
        T, fe_seq, fe_tot = compute_ex_table(f, tn, yn, h, r, pool, seq=seq, dense=dense)
        return (T[r,r], T[r-1,r-1], (fe_seq, fe_tot))

def midpoint_adapt_order(f, tn, yn, h, k, Atol, Rtol, pool, seq=(lambda t: 2*t), dense=False):
    k_max = 10
    k_min = 3
    k = min(k_max, max(k_min, k))
    def A_k(k):
        sum_ = 0
        for i in range(k):
            sum_ += seq(i+1)
        return max(seq(k), sum_/NUM_WORKERS)

    H_k = lambda h, k, err_k: h*0.94*(0.65/err_k)**(1/(2*k-1)) 
    W_k = lambda Ak, Hk: Ak/Hk

    if dense:
        T, fe_seq, fe_tot, y0, Tkk, f_Tkk, y_half, f_yj, hs = compute_ex_table(f, tn, yn, h, k, pool, seq=seq, dense=dense)
    else:
        T, fe_seq, fe_tot = compute_ex_table(f, tn, yn, h, k, pool, seq=seq, dense=dense)

    # compute the error and work function for the stages k-2 and k
    err_k_2 = error_norm(T[k-2,k-3], T[k-2,k-2], Atol, Rtol)
    err_k_1 = error_norm(T[k-1,k-2], T[k-1,k-1], Atol, Rtol)
    err_k   = error_norm(T[k,k-1],   T[k,k],     Atol, Rtol)
    h_k_2   = H_k(h, k-2, err_k_2)
    h_k_1   = H_k(h, k-1, err_k_1)
    h_k     = H_k(h, k,   err_k)
    w_k_2   = W_k(A_k(k-2), h_k_2)
    w_k_1   = W_k(A_k(k-1), h_k_1)
    w_k     = W_k(A_k(k),   h_k)


    if err_k_1 <= 1:
        # convergence in line k-1
        if err_k <= 1:
            y = T[k,k]
        else:
            y = T[k-1,k-1]

        k_new = k if w_k_1 < 0.9*w_k_2 else k-1
        h_new = h_k_1 if k_new <= k-1 else h_k_1*A_k(k)/A_k(k-1)

        if dense:
            poly = interpolate(y0, Tkk, f_Tkk, y_half, f_yj, hs, h, k, Atol, Rtol, seq=seq)

    elif err_k <= 1:
        # convergence in line k
        y = T[k,k]

        k_new = k-1 if w_k_1 < 0.9*w_k else (
                k+1 if w_k < 0.9*w_k_1 else k)
        h_new = h_k_1 if k_new == k-1 else (
                h_k if k_new == k else h_k*A_k(k+1)/A_k(k))

        if dense:
            poly = interpolate(y0, Tkk, f_Tkk, y_half, f_yj, hs, h, k, Atol, Rtol, seq=seq)

    else: 
        # no convergence
        # reject (h, k) and restart with new values accordingly
        k_new = k-1 if w_k_1 < 0.9*w_k else k
        h_new = min(h_k_1 if k_new == k-1 else h_k, h)
        
        if dense:
            y, h, k, h_new, k_new, (fe_seq_, fe_tot_), poly = midpoint_adapt_order(f, tn, yn, h_new, 
                k_new, Atol, Rtol, pool, seq=seq, dense=dense)
        else:
            y, h, k, h_new, k_new, (fe_seq_, fe_tot_) = midpoint_adapt_order(f, tn, yn, h_new, 
                k_new, Atol, Rtol, pool, seq=seq, dense=dense)
        
        fe_seq += fe_seq_
        fe_tot += fe_tot_

    if dense:
        return (y, h, k, h_new, k_new, (fe_seq, fe_tot), poly)
    else:
        return (y, h, k, h_new, k_new, (fe_seq, fe_tot))

def ex_midpoint_parallel(f, t0, tf, y0, adaptive="order", p=4, dense=False,
        step_size=0.5, Atol=0, Rtol=0, exact=(lambda t: t)):
    '''
        An instantiation of extrapolation_serial() function with the midpoint method.
        For more details, refer to extrapolation_serial() function.
    '''
    if dense:
        seq = lambda t: 4*t - 2     # {2,6,10,14,...} sequence for dense output
    else:
        seq = lambda t: 2*t         # harmonic sequence for midpoint method

    method = midpoint_adapt_order if adaptive == "order" else midpoint_fixed_step

    return extrapolation_parallel(method, f, t0, tf, y0,
        adaptive=adaptive, p=p, seq=seq, dense=dense, step_size=step_size,
        Atol=Atol, Rtol=Rtol, exact=exact)
