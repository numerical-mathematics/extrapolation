"""
This is the main file for ParEx, a suite of parallel extrapolation solvers for
initial value problems.  The code is based largely on material from the
following two volumes:

    - *Solving Ordinary Differential Equations I: Nonstiff Problems*,
      by Hairer, Norsett and Wanner
    - *Solving Ordinary Differntial Equations II: Stiff and
      Differential-Algebraic Problems*, by Hairer and Wanner
"""

from __future__ import division
import numpy as np
import multiprocessing as mp
import math
from scipy import optimize
import scipy
import forward_diff
gmres = scipy.sparse.linalg.gmres

#This global variable does not affect multiprocessing performance
#as it is used only in the sequential parts of the code.
NUM_WORKERS = None
jacobian_old = 0

def _set_NUM_WORKERS(nworkers):
    """Set number of parallel workers.

    Used to determine how many processes (maximum) will be used by
    multiprocessing when building the extrapolation table.

    Parameters
    ----------
    nworkers : int
        Number of processes to allow.  If not specified, use value returned by
        mp.cpu_count().
    """
    global NUM_WORKERS
    if nworkers is None:
        try:
            NUM_WORKERS = mp.cpu_count()
        except NotImplementedError:
            NUM_WORKERS = 4
    else:
        NUM_WORKERS = max(nworkers, 1)

def _error_norm(y1, y2, atol, rtol):
    """
    Compute Euclidean grid-norm of difference between vectors y1 and y2,
    scaled based on relative and absolute tolerances.  Tolerances are
    satisfied if the return value is smaller than unity.
    Based on II.4.11 (ref I).

    Parameters
    ----------
    y1, y2 : array_like
        Vectors whose difference will be used.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.

    Returns
    -------
    float
        Scaled norm of y1-y2.
    """
    tol = atol + np.maximum(np.abs(y1), np.abs(y2))*rtol
    return np.linalg.norm((y1-y2)/tol)/(len(y1)**0.5)

"""
BEGINNING BLOCK 1: ODE numerical methods formulas and one step
solvers(explicit, implicit and semi-implicit).

Note that the semi-implicit methods are also referred to by the term
"linearly implicit".

Methods' common structure:
def solve_implicit/explicit/semi-implicit(ode_fun, jac_fun, y_olds, t_old,
                                          f_old, dt, args)
    @param ode_fun (callable ode_fun(y,t,args)): derivative of u(t) (ODE RHS, ode_fun(u,t))
    @param jac_fun (callable jac_fun(y,t,args)): Jacobian of ode_fun.
    @param y_olds (2-tuple) : previous solution values (two previous values, at t_n-2 and t_n-1)
            used to obtain the next point (using two previous values because
            midpoint explicit method needs the previous two points, otherwise
            the t_n-2 value can remain unused)
    @param t_old (float): time at which previous solution is found
    @param f_old (array): function evaluation at the y_old (at t_n-1), so that
           it can be reused if it was already calculated for other purposes
           (Jacobian estimation or dense output interpolation)
    @param dt (float): step length
    @param args (tuple): extra arguments
    @param extra arguments to be passed to some of the methods:
        @param J00 (2D array): (only for semi-implicit) Jacobian estimation at
                   the previous stage value (at each extrapolation stage
                   different number of steps are taken, depending on the step
                   sequence chosen).

    @return (yj, f_yj, fe_tot, je_tot):
        @return yj (array): solution calculated at t_old+dt
        @return f_yj (array): function evaluation value at y_old, t_old
        @return fe_tot (int): number of function evaluations done
        @return je_tot (int): number of Jacobian evaluations done
"""

def linear_solve(A, b, iterative=False, tol=1.e-8, x0=None):
    """Solve Ax=b for x.  Used in semi-implicit method.

    Parameters
    ----------
    A : n x n array
    b : length n array
    iterative : boolean
        If true, use an iterative solver.
    tol : float
        Error tolerance if using an iterative solver.
    x0 : length n array
        Initial guess if using an iterative solver.

    Returns
    -------
    x : length n array
        Approximate solution of Ax = b.
    """
    if iterative:
        # TODO: choose an appropriate value of maxiter to distribute work
        # between taking more steps and having a more accurate solution
        dy, info = gmres(A, b, tol=solver_parameters['min_tol'], x0=x0,
                         maxiter=100)
        if info >0:
            print("Info: maximum iterations reached for sparse system solver (GMRES).")
        return dy
    else:
        if(scipy.sparse.issparse(A)):
            return scipy.sparse.linalg.spsolve(A, b)
        else:
            return np.linalg.solve(A, b)


def _solve_implicit_step(f, jacobian, initial_guess, tol):
    """
    Find the root of the function f() using as initial approximation
    initial_guess.  Used in fully implicit methods.

    Parameters
    ----------
    f : callable f(y,t,args)
        Residual function.
    jacobian : callable jacobian(y,t,args)
        Jacobian of f.
    initial_guess : array
        Estimated value of f to use as initial guess
    tol : dict
        Error tolerance passed to fsolve

    Returns
    -------
    yj : array
        Solution calculated at t_old+dt (root of f)
    fe_tot : int
        Number of function evaluations performed
    je_tot : int
        Number of Jacobian evaluations performed
    """
    # TODO: Problem -> fsolve doesn't seem to work well with xtol parameter
    # TODO: pass to the zero finder solver the tolerance required to the
    # overall problem.  Then the error by the solver won't limit the global
    # error of the ODE solution.
    # TODO: change solver so it doesn't do the 2 extra unnecessary function
    # evaluations
    # https://github.com/scipy/scipy/issues/5369
    # TODO: add extra 2 function evaluations

    x, infodict, _, _ = optimize.fsolve(f, initial_guess,
                                        fprime = jacobian, full_output = True,
                                        xtol=tol)

    if("njev" in infodict):
        return (x, infodict["nfev"], infodict["njev"])
    else:
        return (x, infodict["nfev"], 0)


def _semi_implicit_midpoint(ode_fun, jac_fun, y_olds, t_old, f_old, dt, args,
                            solver_parameters, J00, I):
    """
    Calculate solution at t_old+dt using the semi-implicit midpoint
    formula.  Based on equations IV.9.16a-b of Ref II.
    """
    y_older, y_old = y_olds
    je_tot=0

    if(y_older is None):  # Use Euler to get starting value
        return _semi_implicit_euler(ode_fun, jac_fun, y_olds, t_old,
                                   f_old, dt, args, solver_parameters,
                                   J00, I)

    if(f_old is None):
        f_yj = ode_fun(*(y_old,t_old)+args)
        fe_tot = 1
    else:  # We already computed it and can just reuse it
        f_yj = f_old
        fe_tot=0

    b = np.dot(-(I+dt*J00),(y_old-y_older)) + 2*dt*f_yj
    A = I-dt*J00

    if(solver_parameters['initialGuess']): # Use Euler for initial guess
        x0, f_yj, fe_tot_,je_tot=_explicit_euler(ode_fun, jac_fun, y_olds,
                                                 t_old, f_yj, dt,
                                                 args, solver_parameters)
        fe_tot += fe_tot_
    else:
        x0=None

    dy = linear_solve(A, b, iterative = solver_parameters['iterative'],
                 tol = solver_parameters['min_tol'], x0=x0)

    y_new = y_old + dy

    return (y_new, f_yj, fe_tot, je_tot)


def _semi_implicit_euler(ode_fun, jac_fun, y_olds, t_old,
                        f_old,dt, args, solver_parameters, J00, I):
    """
    Calculate solution at t_old+dt using the semi-implicit Euler method.
    Based on Section IV.9.25 of Ref II.
    """
    y_older, y_old = y_olds
    je_tot = 0
    if(f_old is None):
        f_yj = ode_fun(*(y_old, t_old)+args)
        fe_tot = 1
    else:
        f_yj = f_old
        fe_tot = 0

    b = dt*f_yj
    A = I-dt*J00

    if(solver_parameters['initialGuess']):
        # TODO: Using explicit Euler as a predictor doesn't seem to be
        # effective (maybe because with extrapolation we are taking too big
        # steps for the predictor be close to the solution).
        # x0, f_yj, fe_tot_,je_tot=_explicit_euler(ode_fun, jac_fun,
        #                 y_olds, t_old, f_yj, dt, args, solver_parameters)
        # fe_tot += fe_tot_
        x0 = y_old
    else:
        x0 = None

    dy = linear_solve(A, b, iterative = solver_parameters['iterative'],
                 tol = solver_parameters['min_tol'], x0=x0)

    y_new = y_old + dy

    return (y_new, f_yj, fe_tot, je_tot)


def _implicit_midpoint(ode_fun, jac_fun, y_olds, t_old, f_old,
                       dt, args, solver_parameters):
    """
    Calculate solution at t_old+dt using the implicit midpoint method.
    Based on IV.9.2 (ref II).
    """
    y_older, y_old = y_olds

    def zero_func(x):
        fval = ode_fun(*((y_old+x)/2, t_old+dt/2) + args)
        return x - y_old - dt * fval

    def jacobian(x):
        II = np.identity(len(x), dtype=float)
        return  np.matrix(II - dt*jac_fun(x,t_old+dt/2))


    if(jac_fun is None):
        jacobian = None

    if(not solver_parameters['initialGuess']):
        initial_guess = y_old
        fe_tot = 0
        je_tot = 0
    else:
        #Estimation of the value as the starting point for the zero solver
        initial_guess, f_yj, fe_tot, je_tot = _explicit_euler(ode_fun, jac_fun,
                y_olds, t_old, f_old, dt, args,
                solver_parameters)

    y_new, fe_tot_, je_tot_ = _solve_implicit_step(zero_func, jacobian,
                                                   initial_guess,
                                                   solver_parameters['min_tol'])
    fe_tot += fe_tot_
    je_tot += je_tot_

    f_yj = ode_fun(*(y_old,t_old)+args)
    fe_tot += 1
    return (y_new, f_yj, fe_tot, je_tot)


def _explicit_midpoint(ode_fun, jac_fun, y_olds, t_old, f_old,
                       dt, args, solver_parameters):
    """
    Calculate solution at t_old+dt using the explicit midpoint method.
    Based on II.9.13b of Ref I.
    """
    y_older, y_old = y_olds
    if(y_older is None): # Use Euler to get additional starting value
        return _explicit_euler(ode_fun, jac_fun, y_olds, t_old,
                               f_old, dt, args, solver_parameters)

    f_yj = ode_fun(*(y_old, t_old)+args)
    fe_tot = 1
    return (y_older + (2*dt)*f_yj, f_yj, fe_tot,0)


def _explicit_euler(ode_fun, jac_fun, y_olds, t_old, f_old,
                    dt, args, solver_parameters):
    """
    Calculate solution at t_old+dt doing one step with explicit Euler.
    Based on II.9.13a of Ref I.
    """
    y_older, y_old = y_olds

    if(f_old is None):
        f_yj = ode_fun(*(y_old, t_old)+args)
        fe_tot = 1
    else:
        f_yj = f_old
        fe_tot = 0

    return (y_old + dt*f_yj, f_yj, fe_tot,0)

#END BLOCK 1: ODE numerical methods formulas (explicit, implicit and semi-implicit)


def _compute_stages((solver, solver_args, ode_fun, jac_fun, tn, yn, f_yn, args,
                     h, j_nj_list, smoothing, solver_parameters)):
    """
    Compute extrapolation tableau values with the order specified and number of
    steps specified in j_nj_list.
    Calculate the T_{k,1} values for the k's in j_nj_list.
    Based on II.9.2 (Definition of the method, Ref. I).

    Parameters
    ----------
    solver: Low-order ODE solver to solve each substep
                   ({midpoint,euler}/{implicit,semi-implicit,explicit})
    solver_args: extra arguments to be passed to the solver and jacobian
    ode_fun (callable ode_fun(y,t,args)): derivative of solution
    jac_fun (callable jac_fun(y,t,args)): Jacobian of ode_fun
    tn (float): initial time
    yn (array): solution value at tn
    f_yn (array): function evaluation (ode_fun) at yn,tn
    args (tuple): extra arguments for ode_fun
    h (float): big step to take to obtain T_{k,1}
    j_nj_list (array of 2-tuples): array with (j,nj) pairs indicating
           which y_{h_j}(tn+h) are calculated
    smoothing (string): specifies if smoothing should be performed:
        -'no': no smoothing
        -'gbs': three point smoothing (based on GBS method); see II.9.13c ref I
                and IV.9.9, Ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint); see
                    IV.9.16c, Ref II.
    solver_parameters (dict): extra arguments used by the solver.  For
           more information see _set_default_solver_parameters(..) function.

    Returns
    -------
    list of tuples. Each item represents all information regarding one
        value of the extrapolation tableau first column of values T_{k,1}.
        The list contains:
            k: order of T
            nj: number of steps to calculate T
            Tj1: T_{k,1}
            y_half (array): intermediate solution at half the interval
                    to T (for symmetric interpolation)
            f_yj (array of arrays(f_yj)): all function evaluations done
                    at intermediate solution points (for symmetric
                    interpolation)
        Y (array of arrays(yj)): all intermediate solution points (yj)
                  calculated to obtain T (for non-symmetric interpolation)
        fe_tot (int): number of total function evaluations
        je_tot (int): number of total jacobian evaluations
    """
    results = []
    for (j,nj) in j_nj_list:
        fe_tot = 0
        je_tot = 0
        nj = int(nj)
        Y = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        f_yj = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        Y[0] = yn
        dt = h/nj

        Y[1], f_yj[0], fe_tot_, je_tot_ = solver(ode_fun, jac_fun, (None,
                                                 Y[0]), tn, f_yn, dt, args,
                                                 solver_parameters,
                                                 **solver_args)
        fe_tot += fe_tot_
        je_tot += je_tot_
        for i in range(2,nj+1):
            Y[i], f_yj[i-1], fe_tot_, je_tot_ = solver(ode_fun, jac_fun,
                                                       (Y[i-2], Y[i-1]),
                                                       tn + (i-1)*(h/nj),
                                                       None,
                                                       dt, args,
                                                       solver_parameters,
                                                       **solver_args)
            fe_tot += fe_tot_
            je_tot += je_tot_

        # TODO: this y_half value is already returned inside Y. Remove and when
        # needed (for interpolation) extract from Y.
        y_half = Y[nj/2]

        #Perform smoothing step
        Tj1 = Y[nj]
        if(not smoothing == 'no'):
            # TODO: this f_yj_unused can be used in the case of interpolation,
            # it should be saved in f_yj so function evaluations are
            # not repeated.
            Tj1, f_yj_unused, fe_tot_, je_tot_ = solver(ode_fun, jac_fun,
                                                        (Y[nj-1], Y[nj]), tn +
                                                        h, None, dt, args,
                                                        solver_parameters,
                                                        **solver_args)
            fe_tot += fe_tot_
            je_tot += je_tot_
            if(smoothing == 'gbs'):
                Tj1 = 1/4*(Y[nj-1]+2*Y[nj]+Tj1)
            elif(smoothing == 'semiimp'):
                Tj1 = 1/2*(Y[nj-1]+Tj1)
        results += [(j, nj, Tj1, y_half, f_yj,Y, fe_tot, je_tot)]

    return results

def _extrapolation_parallel(ode_fun, tspan, y0, solver_args, solver=None,
                            jac_fun=None, args=(), diagnostics=False,
                            rtol=1.0e-8, atol=1.0e-8, h0=0.5, max_steps=10e4,
                            step_ratio_limit=2, k=4, nworkers=None,
                            smoothing='no', seq=None,
                            adaptive=True, solver_parameters={}):
    """
    Solve the system of ODEs dy/dt = ode_fun(y, t0, ...) by extrapolation.

    @param solver (callable(...)): the solver on which the extrapolation is based (euler,mipoint/explicit,implicit,semiimplicit)
    @param solver_args (dict): dictionary with extra parameters to be passed to the solver.
    @param ode_fun (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param jac_fun (callable(y,t,args)): computes the Jacobian of the ode_fun function parameter.
    @param y0 (array): initial condition on y (can be a vector).
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first
            element of this sequence. And the last one the final time.
    @param args (tuple): extra arguments to pass to function.
    @param diagnostics (bool): true if user wants a dictionary of optional outputs as the second output.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param h0 (float):the step size to be attempted on the first step.
    @param max_steps (int): maximum number of (internally defined) steps allowed for each
            integration point in t. Defaults to 10e4
    @param step_ratio_limit (int): multiplicative factor that limits the increase and decrease of the adaptive step
            (next step vs last step length ratio is limited by this factor).
    @param k (int): the number of extrapolation steps if order is fixed, or the starting value otherwise.
    @param nworkers (int): the number of workers working in parallel. If nworkers==None, then
        the the number of workers is set to the number of CPUs on the running machine.
    @param smoothing (string): specifies if a smoothing step should be performed:
        -'no': no smoothing step performed
        -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II.
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param adaptive (string): If True, automatically adapt the step size and order.
    @param solver_parameters (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _set_default_solver_parameters(..) function.

    @return:
        @return ys (2D-array, shape (len(t), len(y0))): array containing the value of y for each desired time in t, with
            the initial value y0 in the first row.
        @return infodict (dict): only returned if diagnostics == True. Dictionary containing additional output information
             KEY        MEANING
            'fe_seq'    cumulative number of sequential derivative evaluations
            'nfe'       cumulative number of total derivative evaluations
            'nst'       cumulative number of successful time steps
            'nje'       cumulative number of either Jacobian evaluations (when
                        analytic Jacobian is provided) or Jacobian estimations
                        (when no analytic Jacobian is provided)
            'h_avg'     average step size
            'k_avg'     average number of extrapolation steps
    """
    solver = method_fcns[solver]

    #Initialize pool of workers to parallelize extrapolation table calculations
    _set_NUM_WORKERS(nworkers)
    pool = mp.Pool(NUM_WORKERS)

    assert len(tspan) > 1, ("""The array tspan must be of length at least 2.
    The initial value time should be the first element of tspan and the last
    element of tspan should be the final time.""")

    # ys contains the solutions at the times specified by tspan
    ys = np.zeros((len(tspan), len(y0)), dtype=(type(y0[0])))
    ys[0] = y0
    t0 = tspan[0]
    steps_taken = 0

    #Initialize diagnostic values
    fe_seq = 0
    fe_tot = 0
    je_tot = 0

    t_max = tspan[-1]
    t_index = 1

    #yn is the previous calculated step (at t_curr)
    yn, t_curr = 1*y0, t0
    h = min(h0, t_max-t0)

    sum_ks, sum_hs = 0, 0

    # Initialize reject_step so that Jacobian is updated
    reject_step = True
    y_old, f_old = None, None

    # Iterate until final time
    while t_curr < t_max:
        reject_step, y_temp, ysolution, f_yn, h, k, h_new, k_new, (fe_seq_, fe_tot_, je_tot_) = _solve_one_step(
                solver, solver_args, ode_fun, jac_fun, t_curr, tspan, t_index, yn, args, h, k,
                atol, rtol, pool, smoothing, seq, adaptive, reject_step, y_old, f_old, solver_parameters)
        # y_old, f_old is used for Jacobian updating
        y_old, f_old = (yn, f_yn)

        # Store values if step is not rejected
        if(not reject_step):
            yn = 1*y_temp
            # ysolution includes all output values in step interval
            if(len(ysolution)!=0):
                ys[t_index:(t_index+len(ysolution))] = ysolution
                t_index += len(ysolution)

            t_curr += h
            # add last solution if it matches an output time
            if(tspan[t_index] == t_curr):
                ys[t_index] = yn
                t_index += 1

        # Update diagnostics
        fe_seq += fe_seq_
        fe_tot += fe_tot_
        je_tot += je_tot_
        sum_ks += k
        sum_hs += h

        steps_taken += 1

        if steps_taken > max_steps:
            raise Exception('Reached Max Number of Steps. t = ' + str(t_curr))

        # Step size can be NaN due to overflows of the RHS
        if(math.isnan(h_new)):
            h_new = h/step_ratio_limit
        # Update to new step (limit the change of h_new by a step_ratio_limit)
        h_new = max(min(h_new, h*step_ratio_limit), h/step_ratio_limit)
        # Don't step past t_max
        h = min(h_new, t_max - t_curr)
        # Avoid taking a step with size close to machine epsilon
        if (not adaptive) and (t_max-(t_curr+h))/t_max<1e-12:
            h = t_max - t_curr

        k = k_new

    #Close pool of workers and return results
    pool.close()

    if diagnostics:
        infodict = {'fe_seq': fe_seq, 'nfe': fe_tot, 'nst': steps_taken,
                    'nje': je_tot, 'h_avg': sum_hs/steps_taken,
                    'k_avg': sum_ks/steps_taken}
        return (ys, infodict)
    else:
        return ys

def _balance_load(k, nworkers, seq=(lambda t: 2*t)):
    """
    Distribute the workload as evenly as possible among the available
    processes. The tasks to be distributed are the calculation of each T_{j,1}
    for j=1...k.  The number of steps required to compute T_{j,1} is given by
    seq(j).  This approach to load-balancing can be more performant than
    relying on dynamic load-balancing.


    Each process is given a list of 2-tuples (j,seq(j)) so that the maximal sum
    of seq(j)'s (workloads) for any one process is minimized.
    The algorithm used here is not optimal for all possible sequences, but is
    typically optimal.  Basically, it gives one task to each process in some
    order, then reverses the list of processes and repeats.

    Parameters
    ----------
    k  : int
        Number of extrapolation steps to be performed (i.e., number of T_{j,1} values)
    nworkers : int
        Number of processes over which to distribute jobs
    seq  : callable(i), int i>=1
        Sequence of step numbers

    Returns
    -------
    j_nj_list : list of lists of 2-tuples
        The list contains nworkers lists, each one containing which T_{j,1}
        values each processor has to calculate (each T_{j,1} is specified by
        the tuple (j,nj)).
    """
    if k <= nworkers: # Just let each process compute one of the T_{j,1}
        j_nj_list = [[(j,seq(j))] for j in range(k, 0, -1)]
    else:
        j_nj_list = [[] for i in range(nworkers)]
        processes = range(nworkers)
        i = k
        while 1:
            if i >= nworkers:
                for j in processes:
                    j_nj_list[j] += [(i, seq(i))]
                    i -= 1
            else:
                for j in processes:
                    if i == 0:
                        break
                    j_nj_list[j] += [(i, seq(i))]
                    i -= 1
                break
            processes = processes[::-1]

    return j_nj_list

def _ideal_speedup(k, seq, nworkers):
    """
    Determine the maximum speedup that could possibly be achieved by using
    nworkers instead of just 1.

    This is only used for evaluating performance.
    """
    steps = [seq(j) for j in range(1,k+1)]
    serial_cost = sum(steps)
    j_nj_list = _balance_load(k, nworkers, seq)
    max_work = 0
    for proc_worklist in j_nj_list:
        work = sum([nj for j, nj in proc_worklist])
        max_work = max(max_work, work)
    return serial_cost / float(max_work)


def _update_jacobian(jacobian_old, ode_fun, yn, tn, yn_1, f_yn_1, args):
    """
    Update the Jacobian estimation.  Also returns (for efficiency)
    the solution derivative at the current time.
    Uses the update formula suggested for Broyden's method.

    Parameters
    ----------
    jacobian_old : 2D array
        Jacobian approximation at the previous time
    ode_fun : callable(y,t,...)
        RHS of the ODE
    yn  : array
        solution at tn (current time)
    tn  : array
        current time
    yn_1 : array
        solution at tn-1 (previous stage time)
    f_yn_1 : array
        function evaluation at yn_1,tn_1
    args : tuple
        extra arguments for ode_fun

    Returns
    -------
    jacobian : 2D array
        Jacobian approximation updated
    f_yn : array
        function evaluation at yn,tn
    """
    f_yn = ode_fun(*(yn,tn)+args)
    incf_yn = f_yn-f_yn_1
    incyn = yn-yn_1
    jacobian = jacobian_old + np.outer((incf_yn-np.dot(jacobian_old,incyn)) \
                                        /np.linalg.norm(incyn, 2),incyn)
    return (jacobian, f_yn)


def _get_jacobian(ode_fun, args, yn, tn, jac_fun, solver_args,
                  last_step_rejected, y_old, f_old, freeze):
    """
    Return the Jacobian approximation J(yn,tn) where J is the deriviative of
    ode_fun.  If jac_fun (analytical Jacobian) is available, jac_fun is used.
    Otherwise, if freeze==False, the Jacobian is estimated with a forward
    difference formula (see forward_diff.Jacobian).  If freeze==True, the last
    Jacobian approximation is used.  If the last step was rejected, we always
    update the Jacobian.

    Parameters
    ----------
    ode_fun : callable(y,t,args)
        RHS of the ODE
    args : tuple
        extra arguments to ode_fun
    yn : array
        solution at tn (current time)
    tn : array
        current time
    jac_fun : callable(y,t,args)
        computes analytically the Jacobian of the ode_fun function parameter.
    solver_args : dict
        contains solver methods' additional parameters.
    last_step_rejected : bool
        True if last step was rejected
    y_old, f_old : 2-tuple
        tuple containing the solution at the previous step (tn-1) and its
        derivative, (yn_1, f_yn_1)

    Returns
    -------
    J00 : 2D array
        The Jacobian
    f_yn : array
        function evaluation at yn,tn
    fe_tot : int
        number of function evaluations (0 if analytical Jacobian, N if
        estimated Jacobian)
    jac_evals : int
        number of Jacobian evaluations (1 if analytical Jacobian)
        or Jacobian estimations (1 if estimated Jacobian)
    """
    update_jacobian = False # TODO: try turning this on and see if it helps
    f_yn = None
    global jacobian_old
    if last_step_rejected: # Unfreeze Jacobian if we run into trouble
        freeze = False

    def func_at_tn(y, args):
        return ode_fun(*(y,tn)+args)

    if jac_fun:  # Jacobian function provided
        J00 = jac_fun(yn, tn)
        fun_evals, jac_evals = 0, 1
    elif not freeze:  # Estimate Jacobian with finite differences
        f_yn = func_at_tn(yn,args)
        J00, fe_tot_ = forward_diff.Jacobian(func_at_tn, yn, f_yn, args)
        fun_evals = fe_tot_ + 1
        jac_evals = 1
        jacobian_old = J00
    elif update_jacobian: # Frozen, but do approximate update
        J00, f_yn = _update_jacobian(jacobian_old, ode_fun, yn, tn,
                                     y_old, f_old, args)
        fun_evals, jac_evals = 1, 0
    else: # Reuse old Jacobian
        J00, f_yn = jacobian_old, None
        fun_evals, jac_evals = 0, 0

    return (J00, f_yn, fun_evals, jac_evals)


def _compute_extrapolation_table(solver, solver_args, ode_fun, jac_fun, tn, yn, args,
                                 h, k, pool, last_step_rejected,
                                 y_old, f_old, seq, smoothing,
                                 solver_parameters):
    """
    Compute the extrapolation tableau for a given big step, order and step
    sequence. The values T_{1,i} are computed concurrently, taking all the
    inner steps necessary and then extrapolating the final value at tn+h.

    @param solver (callable(...)): the solver on which the extrapolation is
           based (euler,mipoint/explicit, implicit,semiimplicit)
    @param solver_args (dict): dictionary with extra parameters to be passed to
           the solver solver.
    @param ode_fun (callable(y, t,args)): computes the derivative of y at t (i.e.
           the right hand side of the IVP).
    @param jac_fun (callable(y,t,args)): computes the Jacobian of the ode_fun
           function parameter.
    @param tn (float): starting integration time
    @param yn (array): solution at tn (current time)
    @param args (tuple): extra arguments to pass to function.
    @param h (float): integration step to take (the output, without
           interpolation, will be calculated at t_curr+h) This value matches
           with the value H in ref I and ref II.
    @param k (int): number of extrapolation steps to take in this step
           (determines the number of extrapolations performed to achieve a
           better integration output, equivalent to the size of the
           extrapolation tableau).
    @param pool: multiprocessing pool of workers (with as many workers as
           processors) that will parallelize the calculation of each of the initial
           values of the extrapolation tableau (T_{i,1} i=1...k).
    @param last_step_rejected (bool): whether previously taken step was
           rejected or not
    @param y_old, f_old (2-tuple): tuple containing the solution at the
           previous step (tn-1) and its function evaluation, (yn_1, f_yn_1)
    @param seq (callable(i), int i>=1): the step-number sequence (examples
           II.9.1 , 9.6, 9.35 ref I).
    @param smoothing (string): specifies if a smoothing step should be
           performed:
            -'no': no smoothing step performed
            -'gbs': three point smoothing step (based on GBS method), II.9.13c
             ref I and IV.9.9 ref II.
            -'semiimp': two point smoothing step (for semiimplicit midpoint),
             IV.9.16c ref II.
    @param solver_parameters (dict): extra arguments needed to define completely
           the solver's behavior.  Should not be empty, for more information see
           _set_default_solver_parameters(..) function.

    @return (T, y_half, f_yj, yj, f_yn, hs,(fe_seq, fe_tot, je_tot)):
        @return T (2D array): filled extrapolation tableau (size k) with all the T_{i,j} values in the lower
                triangular side
        @return y_half (2D array): array containing for each extrapolation value (1...k) an array with the intermediate (at half
                the integration interval) solution value.
        @return f_yj (3D array): array containing for each extrapolation value (1...k) an array with all the function evaluations
                done at the intermediate solution values.
        @return yj (3D array): array containing for each extrapolation value (1...k) an array with all the intermediate solution
                values obtained to calculate each T_{i,1}.
        @return f_yn (array): function (RHS) evaluation at yn,tn
        @return hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (ref I)
        @return (fe_seq, fe_tot,je_tot):
            @return fe_seq (int): cumulative number of sequential derivative evaluations performed for this step
            @return fe_tot (int): cumulative number of total derivative evaluations performed for this step
            @return je_tot (int): cumulative number of either Jacobian evaluations (when analytic Jacobian is
                    provided) or Jacobian estimations (when no analytic Jacobian is provided) performed
                    for this step
    """
    T = np.zeros((k+1,k+1, len(yn)), dtype=(type(yn[0])))
    j_nj_list = _balance_load(k, NUM_WORKERS, seq=seq)

    if solver in ( method_fcns[Solvers.SEMI_IMPLICIT_EULER], 
                   method_fcns[Solvers.SEMI_IMPLICIT_MIDPOINT]):
        jac, f_yn, fe_tot, je_tot = _get_jacobian(ode_fun, args, yn, tn, jac_fun,
                                             solver_args, last_step_rejected,
                                             y_old, f_old,
                                             solver_parameters['freezeJac'])
        solver_args['J00'] = jac
    else:
        je_tot = 0
        fe_tot = 0
        f_yn = None

    jobs = [(solver, solver_args, ode_fun, jac_fun, tn, yn, f_yn, args, h, k_nj,
             smoothing, solver_parameters) for k_nj in j_nj_list]
    results = pool.map(_compute_stages, jobs, chunksize=1)

    # Here's how this would be done in serial:
    # res = _compute_stages((solver, solver_args, ode_fun, jac_fun, tn, yn, f_yn, args, h, j_nj_list, smoothing))

    # At this stage fe_tot has only counted the function evaluations for the
    # jacobian estimation (which are not parallelized)
    fe_seq = 1*fe_tot
    fe_tot_stage_max=0
    # process the results returned from the pool
    y_half = (k+1)*[None]
    f_yj = (k+1)*[None]
    yj = (k+1)*[None]
    hs = (k+1)*[None]

    for res in results:
        fe_tot_stage = 0
        for (k_, nj_, Tk_, y_half_, f_yj_, yj_, fe_tot_, je_tot_) in res:
            T[k_, 1] = Tk_
            y_half[k_] = y_half_
            f_yj[k_] = f_yj_
            yj[k_] = yj_
            hs[k_] = h/nj_
            fe_tot += fe_tot_
            je_tot += je_tot_
            fe_tot_stage += fe_tot_
        # Count the maximum number of sequential function evaluations taken
        fe_tot_stage_max = max(fe_tot_stage_max, fe_tot_stage)

    fe_seq += fe_tot_stage_max
    symmetric = (solver in symmetric_methods)
    _fill_extrapolation_table(T, 0, seq, symmetric)

    return (T, y_half, f_yj, yj, f_yn, hs,(fe_seq, fe_tot, je_tot))

def _fill_extrapolation_table(T, j_initshift, seq, symmetric):
    """
    Fill extrapolation table using the first column values T_{i,1}. This
    function computes the rest of values T_{i,j} (lower triangular values).

    The formula to extrapolate the T_{i,j} values takes into account if the
    method to compute T_{i,1} was symmetric or not (See Section II.9 of Ref.
    I).

    @param T (2D array): extrapolation table (lower triangular) containing
            the first column calculated and to be filled
    @param j_initshift (int): step sequence index matching the number of steps taken to compute the first
            value T_{1,1} of this extrapolation tableau. I.e. if step sequence is {2,6,10,14...} and
            T_{1,1} was calculated with 6 steps then j_initshift=1 and T_{i,1} are assumed to be calculated
            with seq(i+j_initshift), thus {6,10,14...}
    @param seq (callable(i) int i>=1): step sequence of the first column values
    @param symmetric (bool): whether the solver used to compute the first column of T is symmetric or not.
    """
    for i in range(2, T.shape[0]):
        for j in range(i, T.shape[0]):
            if(symmetric):
                T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j+j_initshift)/seq(j+j_initshift-i+1))**2 - 1)
            else:
                T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j+j_initshift)/seq(j+j_initshift-i+1)) - 1)

def _centered_diff(j, f_yj, hj):
    """
    Computes through the centered differentiation formula different order
    derivatives of y for a given extrapolation step (for every T{1,j}). In
    other words, it calculates II.9.39 of Ref. I (step 1) for all kappa for a
    fixed j (i.e. for a fixed extrapolation step).

    Used by _interpolate_sym.

    @param j (int): which extrapolation inner-stage is used
    @param f_yj (2D array): array with all the function evaluations of all the intermediate solution
            values obtained to calculate each T_{j,1}.
    @param hj (array): inner step taken in the j-th extrapolation step, H/nj (II.9.1 ref I).

    @return dj (2D array): array containing for each kappa=1...2j the kappa-th derivative of y estimated
        using the j-th extrapolation step values
    """
    max_order = 2*j
    nj = len(f_yj) - 1
    coeff = [1,1]
    dj = (max_order+1)*[None]
    dj[1] = 1*f_yj[nj/2]  # First derivative
    dj[2] = (f_yj[nj/2+1] - f_yj[nj/2-1])/(2*hj) # Second derivative

    for order in range(2,max_order):  # Higher-order derivatives
        coeff = [1] + [coeff[j] + coeff[j+1] for j in range(len(coeff)-1)] + [1]
        index = [nj/2 + order - 2*i for i in range(order+1)]
        sum_ = 0
        for i in range(order+1):
            sum_ += ((-1)**i)*coeff[i]*f_yj[index[i]]
        dj[order+1] = sum_ / (2*hj)**order

    return dj

def _backward_diff(j, yj, hj, lam):
    """
    Computes through the backward differentiation formula different order
    derivatives of y for a given extrapolation step (for every T{1,j}). In
    other words, it calculates VI.5.43 ref II (step 1) for all kappa for a
    fixed j (for a fixed extrapolation step).

    Used for _interpolate_nonsym.

    @param j (int): which extrapolation inner-stage is used
    @param yj (2D array): array with all the intermediate solution values obtained to calculate each T_{j,1}.
    @param hj (array): inner step taken in the j-th extrapolation step, H/nj (II.9.1 ref I).
    @param lam (int): either 0 or 1, check definition and use in ref II pg 439

    @return rj (2D array): array containing for each kappa=1...j-lam the kappa-th derivative of y estimated
        using the j-th extrapolation step values
    """
    max_order = j-lam
    nj = len(yj) - 1
    coeff = [1,1]
    rj = (max_order+1)*[None]
    rj[1] = (yj[nj] - yj[nj-1])/hj

    for order in range(2,max_order+1):
        coeff = [1] + [coeff[j] + coeff[j+1] for j in range(len(coeff)-1)] + [1]
        index = [nj - i for i in range(order+1)]

        sum_ = 0
        for i in range(order+1):
            sum_ += ((-1)**i)*coeff[i]*yj[index[i]]
        rj[order] = sum_ /hj**order

    return rj

def _compute_rs(yj, hs, k, seq=(lambda t: 4*t-2)):
    """
    Approximate derivatives of y using backward differentiation and
    extrapolation.  Delegates VI.5.43 ref II (step 1) to _backward_diff() and
    here performs the extrapolation of the derivatives. Follows VI.5.44
    ref II (step 2).

    Used for _interpolate_nonsym.

    Note: compared to the reference, k and kappa are interchanged in this code,
    to maintain the same notation as ref I (see _compute_ds function)

    Contains a parameter to be chosen: lam={0,1}.

    lam=1 requires less work and the order of interpolation is enough
    given the global error committed. Theorem VI.5.7 ref II (interpolation
    error).

    @param yj (3D array): array containing for each extrapolation value (1...k) an array with all the intermediate solution
            values obtained to calculate each T_{i,1}.
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I)
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).

    @return rs (2D array): array containing for each kappa=1...k the kappa-th derivative of y
            (extrapolated k-kappa-lam times)
    """
    lam=1
    dj_kappa = np.zeros((k+1-lam, k+1), dtype=(type(yj[1][0])))
    rs = np.zeros((k+1-lam), dtype=(type(yj[1][0])))

    for j in range(1+lam,k+1):
        dj_ = _backward_diff(j,yj[j], hs[j],lam)
        for kappa in range(1,j+1-lam):
            dj_kappa[kappa,j] = 1*dj_[kappa]

    for kappa in range(1,k+1-lam):
        numextrap = k+1-kappa-lam
        T = np.zeros((numextrap+1, numextrap+1), dtype=(type(yj[1][0])))
        T[:,1] = 1*dj_kappa[kappa, (kappa+lam-1):]
        _fill_extrapolation_table(T, lam+kappa-1, seq, symmetric=False)

        rs[kappa] = 1*T[numextrap,numextrap]

    return rs

def _compute_ds(y_half, f_yj, hs, k, seq=(lambda t: 4*t-2)):
    """
    Compute derivatives of y using centered differentiation and
    extrapolation. In other words, it
    delegates II.9.39 ref I (step 1) to _backward_diff(.) function and
    here it performs the extrapolation of the derivatives. It follows II.9 step
    2 ref I.

    Used for _interpolate_sym.

    @param y_half (2D array): array containing for each extrapolation value (1...k) an array with the intermediate (at half
            the integration interval) solution value.
    @param f_yj (3D array): array containing for each extrapolation value (1...k) an array with all the function evaluations
            done at the intermediate solution values.
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I)
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).

    @return ds (2D array): array containing for each kappa=0...2k the kappa-th derivative of y
            (extrapolated k-l times where l see definition at II.9 step 2 ref I)
    """
    dj_kappa = np.zeros((2*k+1, k+1), dtype=(type(y_half[1])))
    ds = np.zeros((2*k+1), dtype=(type(y_half[1])))

    for j in range(1,k+1):
        dj_kappa[0,j] = 1*y_half[j]
        dj_ = _centered_diff(j, f_yj[j], hs[j])
        for kappa in range(1,2*j+1):
            dj_kappa[kappa,j] = 1*dj_[kappa]

    #skip value is used because extrapolation is required
    #(k-l) times for dj_(2l-1) and dj_(2l)
    skip = 0
    for kappa in range(2*k+1):
        numextrap = k-int(skip/2)
        T = np.zeros((numextrap+1, numextrap+1), dtype=(type(y_half[1])))
        T[:,1] = 1*dj_kappa[kappa, int(skip/2):]

        _fill_extrapolation_table(T, int(skip/2), seq, symmetric=True)

        ds[kappa] = 1*T[numextrap,numextrap]
        if (kappa != 0):
            skip += 1

    return ds

def _interpolation_poly(a_u, a_u_1, H, shift, atol, rtol):
    """
    Get interpolation polynomial with a_u coefficients and adjust variable to
    problem with shift.  It returns a polynomial that returns for every
    theta in (0,1) the interpolation value at x0+theta*H.  See theorem II.9.5
    ref I and theorem VI.5.7 ref II.

    @param a_u (array): coefficients of the interpolation polynomial
    @param a_u_1 (array): coefficients of the interpolation polynomial of one degree less (for error estimation)
    @param H (float): step taken by solver (coefficients a_u and a_u_1 were calculated on (0,1) interval)
    @param degree (int): degree of polynomial a_u (should match len(a_u)-1)
    @param shift (float): variable change that was made to calculate a_u and a_u_1, i.e. x-shift,
            because coefficients a_u and a_u_1 were calculated on (0-shift,1-shift) interval.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).

    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function)
    """

    def poly(t):
        """
        Interpolation polynomial

        @param t(float): value in interval (0,1)

        @return (res, errint, h_int):
            @return res (float): interpolation result at x0+theta*H, P(theta).
                    See theorem II.9.5 ref I and theorem VI.5.7 ref II.
            @return errint (float): estimated interpolation error (uses the one degree less polynomial to
                    estimate the interpolation error). See II.9.45 ref I.
            @return h_int (float): suggested next step (in case of rejected interpolation).
                    See formula after II.9.45 ref I.
        """
        degree = len(a_u) - 1

        res     = np.polyval(a_u[::-1],   t-shift)
        res_u_1 = np.polyval(a_u_1[::-1], t-shift)

        errint = _error_norm(res, res_u_1, atol, rtol)

        h_int = H*((1/errint)**(1/degree))

        return (res, errint, h_int)

    return poly

def _interpolate_nonsym(y0, Tkk, yj, hs, H, k, atol, rtol,
                        seq=(lambda t: 4*t-2)):
    """
    Non symmetrical formula (for example used for Euler's method) to
    interpolate dense output values. It calculates a polynomial to interpolate
    any value from t0 (time at y0) to t0+H (time at Tkk). Based on Dense
    Output, VI.5 pg 438-439.

    Returns a polynomial that fulfills the conditions at VI.5.44 (step 3). To
    take into account: this coefficients were calculated for the shifted
    polynomial with x -> x-1.

    @param y0 (float): solution of ODE at the previous step, at t0
    @param Tkk (float): solution of ODE once the step was taken, at t0+H
    @param yj (3D array): array containing for each extrapolation value (1...k) an array with all the intermediate solution
            values obtained to calculate each T_{i,1}.
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I)
    @param H (float): integration step to take (the output, without interpolation, will be calculated at t_curr+h)
            This value matches with the value H in ref I and ref II.
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).

    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function in _interpolation_poly(.))
    """
    u = k
    u_1 = u - 1

    rs = _compute_rs(yj, hs, k, seq=seq)

    #a_u are the coefficients for the interpolation polynomial
    a_u = (u+1)*[None]
    a_u_1 = (u_1+1)*[None]

    a_u[0] = 1*Tkk
    sumcoeff=0
    for i in range(1,u):
        a_u[i] = (H**i)*rs[i]/math.factorial(i)
        sumcoeff += (-1)**i*a_u[i]

    a_u[u] = 1/(-1)**u*(y0-a_u[0]-sumcoeff)

    a_u_1[0:u_1] = 1*a_u[0:u_1]
    a_u_1[u_1] = 1/(-1)**u_1*(y0-a_u_1[0]-sumcoeff+(-1)**u_1*a_u[u_1])

    return _interpolation_poly(a_u, a_u_1, H, 1, atol, rtol)


def _interpolate_sym(y0, Tkk, f_Tkk, y_half, f_yj, hs, H, k, atol, rtol,
                     seq=(lambda t: 4*t-2)):
    """
    Symmetrical formula (for example used for the midpoint method) to interpolate
    dense output values. It calculates a polynomial to interpolate any value
    from t0 (time at y0) to t0+H (time at Tkk). Based on Dense Output for the
    GBS Method, II.9 pg 237-239.

    Returns a polynomial that fulfills the conditions at II.9.40 (step 3). To
    take into account: this coefficients were calculated for the shifted
    polynomial with x -> x-1/2.

    @param y0 (float): solution of ODE at the previous step, at t0
    @param Tkk (float): solution of ODE once the step was taken, at t0+H
    @param f_Tkk (float): function evaluation at Tkk, t0+H
    @param y_half (2D array): array containing for each extrapolation value (1...k) an array with the intermediate (at half
            the integration interval) solution value.
    @param f_yj (3D array): array containing for each extrapolation value (1...k) an array with all the function evaluations
            done at the intermediate solution values.
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I)
    @param H (float): integration step to take (the output, without interpolation, will be calculated at t_curr+h)
            This value matches with the value H in ref I and ref II.
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).

    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function in _interpolation_poly(.))

    """
    u = 2*k-3
    u_1 = u - 1

    ds = _compute_ds(y_half, f_yj, hs, k, seq=seq)

    a_u = (u+5)*[None]
    a_u_1 = (u_1+5)*[None]

    for i in range(u+1):
        a_u[i] = (H**i)*ds[i]/math.factorial(i)

    a_u_1[0:u_1+1] = 1*a_u[0:u_1+1]

    def A_inv(u):
        return (2**(u-2))*np.matrix(
                [[(-2*(3 + u))*(-1)**u,   -(-1)**u,     2*(3 + u),   -1],
                 [(4*(4 + u))*(-1)**u,     2*(-1)**u,   4*(4 + u),   -2],
                 [(8*(1 + u))*(-1)**u,     4*(-1)**u,  -8*(1 + u),    4],
                 [(-16*(2 + u))*(-1)**u,  -8*(-1)**u,  -16*(2 + u),   8]]
                )


    A_inv_u = A_inv(u)

    A_inv_u_1 = A_inv(u_1)

    b1_u = 1*y0
    b1_u_1 = 1*y0
    for i in range(u_1+1):
        b1_u -= a_u[i]/(-2)**i
        b1_u_1 -= a_u_1[i]/(-2)**i

    b1_u -= a_u[u]/(-2)**u


    b2_u = H*f_yj[1][0]
    b2_u_1 = H*f_yj[1][0]
    for i in range(1, u_1+1):
        b2_u -= i*a_u[i]/(-2)**(i-1)
        b2_u_1 -= i*a_u_1[i]/(-2)**(i-1)

    b2_u -= u*a_u[u]/(-2)**(u-1)


    b3_u = 1*Tkk
    b3_u_1 = 1*Tkk
    for i in range(u_1+1):
        b3_u -= a_u[i]/(2**i)
        b3_u_1 -= a_u_1[i]/(2**i)

    b3_u -= a_u[u]/(2**u)

    b4_u = H*f_Tkk
    b4_u_1 = H*f_Tkk
    for i in range(1, u_1+1):
        b4_u -= i*a_u[i]/(2**(i-1))
        b4_u_1 -= i*a_u_1[i]/(2**(i-1))

    b4_u -= u*a_u[u]/(2**(u-1))

    b_u = np.array([b1_u,b2_u,b3_u,b4_u])
    b_u_1 = np.array([b1_u_1,b2_u_1,b3_u_1,b4_u_1])

    x = A_inv_u*b_u
    x = np.array(x)

    x_1 = A_inv_u_1*b_u_1
    x_1 = np.array(x_1)

    a_u[u+1] = x[0]
    a_u[u+2] = x[1]
    a_u[u+3] = x[2]
    a_u[u+4] = x[3]

    a_u_1[u_1+1] = x_1[0]
    a_u_1[u_1+2] = x_1[1]
    a_u_1[u_1+3] = x_1[2]
    a_u_1[u_1+4] = x_1[3]

    # polynomial of degree u+4 defined on [0,1] and centered about 1/2
    # also returns the interpolation error (errint). If errint > 10, then
    # reject step
    return _interpolation_poly(a_u, a_u_1, H, 0.5, atol, rtol)


def _estimate_next_step_and_order(T, k, h, atol, rtol, seq, adaptive, addWork):
    """
    Estimates next step and order, and whether to reject step, from the results
    obtained by the solver and the requested tolerance.
    Based on the section Order and Step Size Control (II.9 Extrapolation Methods),
    Solving Ordinary Differential Equations I (Hairer, Norsett & Wanner)

    @param T: numpy.ndarray (bidimensional matrix)
        Extrapolation tableau with all the extrapolation values
    @param k : int
        Initial condition on y (can be a vector). Must be a non-scalar
        numpy.ndarray
    @param h : float
        An ordered sequence of time points for which to solve for y. The initial
        value point should be the first element of this sequence.
    @param atol, rtol : float
        The input parameters atol and rtol (absolute tolerance and relative tolerance)
        determine the error control performed by the solver.
    @param seq: callable(k) (k: positive int)
        The step-number sequence used to compute T
    @param adaptive: If True, automatically adapt the step size and order.

    @return
        @return reject_step : whether to reject the step and the solution obtained
            (because it doesn't satisfy the asked tolerances)
        @return y : best solution at the end of the step
        @return h_new : new step to take
        @return k_new : new number of extrapolation steps to use
    """
    if not adaptive:
        return (False, T[k,k], h, k)

    sizeODE = 0
    if(addWork):
        sizeODE = len(T[k,k])

    #Define work function (to minimize)
    def A_k(k):
        """
        Number of sequential RHS evaluations needed to compute k lines of the
        extrapolation table.
        """
        sum_ = sum([seq(i+1) for i in range(k)])
        # sizeODE is the length of the ODE
        return max(seq(k), sum_/NUM_WORKERS)+sizeODE # The second value is only an estimate

    H_k = lambda h, k, err_k: h*0.94*(0.65/err_k)**(1/(2*k-1))
    W_k = lambda Ak, Hk: Ak/Hk

    # compute the error and work function for the stages k-2, k-1 and k
    err_k = []
    h_k = []
    w_k = []

    for i in range(3):
        err_k.append(_error_norm(T[k-i,k-i-1], T[k-i,k-i], atol, rtol))
        h_k.append(H_k(h, k-i, err_k[i]))
        w_k.append(W_k(A_k(k-i), h_k[i]))

    if err_k[1] <= 1: # convergence in line k-1
        if err_k[0] <= 1:
            y = T[k,k]
        else:
            y = T[k-1,k-1]
        k_new = k if w_k[1] < 0.9*w_k[2] else k-1
        h_new = h_k[1] if k_new <= k-1 else h_k[1]*A_k(k)/A_k(k-1)
        reject_step=False
    elif err_k[0] <= 1: # convergence in line k
        y = T[k,k]
        k_new = k-1 if w_k[1] < 0.9*w_k[0] else (
                k+1 if w_k[0] < 0.9*w_k[1] else k)
        h_new = h_k[1] if k_new == k-1 else (
                h_k[0] if k_new == k else h_k[0]*A_k(k+1)/A_k(k))
        reject_step = False
    else:
        # no convergence
        # reject (h, k) and restart with new values
        k_new = k-1 if w_k[1] < 0.9*w_k[0] else k
        h_new = min(h_k[1] if k_new == k-1 else h_k[0], h)
        y = None
        reject_step = True

    # Extra protection for the cases where there has been an
    # overflow. Make sure the step is rejected (there are cases
    # where err_k[x] is not nan)
    if(math.isnan(h_new)):
        reject_step = True

    return (reject_step, y, h_new, k_new)


def _solve_one_step(solver, solver_args, ode_fun, jac_fun, t_curr, t, t_index, yn,
                    args, h, k, atol, rtol, pool, smoothing, seq,
                    adaptive, last_step_rejected, y_old, f_old,
                    solver_parameters):
    """
    Solves one 'big' H step of the ODE (with all its inner H/nj steps and the
    extrapolation). In other words, solve one full stage of the problem (one
    step of parallel extrapolation) and interpolates all the dense output
    values required.

    @param solver (callable(...)): the solver on which the extrapolation is based (euler,mipoint/explicit,
            implicit,semiimplicit)
    @param solver_args (dict): dictionary with extra parameters to be passed to the solver.
    @param ode_fun (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param jac_fun (callable(y,t,args)): computes the Jacobian of the ode_fun function parameter.
    @param t_curr (float): current integration time (end time of the previous successful step taken)
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first
            element of this sequence. And the last one the final time.
    @param t_index (int): index of t (array) at which next output value is requested
            (all values at index<t_index have already been computed).
    @param yn (array): solution at t_curr (current time)
    @param args (tuple): extra arguments to pass to function.
    @param h (float): integration step to take (the output, without interpolation, will be calculated at t_curr+h)
            This value matches with the value H in ref I and ref II.
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param pool: multiprocessing pool of workers (with as many workers as processors) that will parallelize the
            calculation of each of the initial values of the extrapolation tableau (T_{i,1} i=1...k).
    @param smoothing (string): specifies if a smoothing step should be performed:
        -'no': no smoothing step performed
        -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II.
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param adaptive (string): If True, automatically adapt the step size and order.
    @param last_step_rejected (bool): whether previously taken step was rejected or not
    @param y_old, f_old (2-tuple): tuple containing the solution at the previous step (tn-1) and its
            function evaluation, (yn_1, f_yn_1)
    @param solver_parameters (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _set_default_solver_parameters(..) function.

    @return (reject_step, y, y_solution, f_yn, h, k, h_new, k_new, (fe_seq, fe_tot, je_tot)):
        @return reject_step (bool): whether this step should be rejected or not. True when this step was not successful
                (estimated error of the final solution or the interpolation solutions too large for the tolerances
                required) and has to be recalculated (with a new step size h_new and order k_new).
        @return y (array): final solution obtained for this integration step, at t_curr+h.
        @return y_solution (2D array): array containing the solution for every output value required (in t parameter)
                 that fell in this integration interval (t_curr,t_curr+h). These values were interpolated.
        @return f_yn (array): function evaluation at yn (initial point solution)
        @return h (float): step taken in this step
        @return k (int): order taken in this step
        @return h_new (float): new suggested step to take in next integration step
        @return k_new (int): new suggested number of extrapolation steps in next integration step
        @return (fe_seq, fe_tot,je_tot):
            @return fe_seq (int): cumulative number of sequential derivative evaluations performed for this step
            @return fe_tot (int): cumulative number of total derivative evaluations performed for this step
            @return je_tot (int): cumulative number of either Jacobian evaluations (when analytic Jacobian is
                    provided) or Jacobian estimations (when no analytic Jacobian is provided) performed
                    for this step
    """

    dense_output_needed = t[t_index] < t_curr + h
    symmetric = (solver in symmetric_methods)

    if dense_output_needed and symmetric:
        seq = lambda t: 2*(2*t-1)     # {2,6,10,14,...} sequence for dense output

    #Limit k, order of extrapolation
    #If order and step are fixed, do not limit order
    if adaptive:
        k_max = 10
        k_min = 3
        k = min(k_max, max(k_min, k))

    T, y_half, f_yj,yj, f_yn, hs, (fe_seq, fe_tot, je_tot) = _compute_extrapolation_table(solver, solver_args, ode_fun, jac_fun,
                t_curr, yn, args, h, k, pool, last_step_rejected, y_old, f_old, seq, smoothing, solver_parameters)

    reject_step, y, h_new, k_new = _estimate_next_step_and_order(T, k, h, atol, rtol, seq, adaptive, solver_parameters['addWork'])

    y_solution=[]
    if(dense_output_needed and (not reject_step)):
        reject_step, y_solution, h_int, f_evals = _interpolate_values_at_t(ode_fun,
                args, T, k, t_curr, t, t_index, h, hs, y_half, f_yj,yj, yn, atol, rtol, seq, adaptive, symmetric)
        fe_tot += f_evals
        fe_seq += f_evals

        if adaptive:
            if reject_step: # Step rejected because of interpolation error
                h_new = h_int
                k_new = k   # Don't update the order
            elif((h_int is not None) and h_int<h_new):
                h_new = h_int


    return (reject_step, y, y_solution, f_yn, h, k, h_new, k_new, (fe_seq, fe_tot, je_tot))


def _interpolate_values_at_t(ode_fun, args, T, k, t_curr, t, t_index, h, hs,
                             y_half, f_yj,yj, yn, atol, rtol, seq, adaptive,
                             symmetric):
    """
    This function calculates all the intermediate solutions requested as dense
    output that fall within a given integration step. It generates an
    interpolation polynomial and evaluates all the required solutions. If
    the interpolation error is not small enough, the step can be forced to be
    rejected.

    This operation has to be done after the "big" extrapolation step has been
    taken.

    @param ode_fun (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param args (tuple): extra arguments to pass to function.
    @param T (2D array): filled extrapolation tableau (size k) with all the T_{i,j} values in the lower
            triangular side
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param t_curr (float): current integration time (end time of the previous successful step taken)
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first
            element of this sequence. And the last one the final time.
    @param t_index (int): index of t (array) at which next output value is requested
            (all values at index<t_index have already been computed).
    @param h (float): integration step to take (the output, without interpolation, will be calculated at t_curr+h)
            This value matches with the value H in ref I and ref II.
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I)
    @param y_half (2D array): array containing for each extrapolation value (1...k) an array with the intermediate (at half
            the integration interval) solution value.
    @param f_yj (3D array): array containing for each extrapolation value (1...k) an array with all the function evaluations
            done at the intermediate solution values.
    @param yj (3D array): array containing for each extrapolation value (1...k) an array with all the intermediate solution
            values obtained to calculate each T_{i,1}.
    @param yn (array): solution at the beggining of the step
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param adaptive (string): If True, automatically adjust the step size and order.
    @param symmetric (bool): whether the solver to solve one step is symmetric (midpoint/trapezoidal)
            or non-symmetric (euler).

    @return (reject_step, y_solution, h_int, f_evals):
        @return reject_step (bool): whether this step should be rejected or not. True when the interpolation was not successful
                for some of the interpolated values (estimated error of the interpolation too large for the tolerances
                required) and the step has to be recalculated (with a new step size h_int).
        @return y_solution (2D array): array containing the solution for every output value required (in t parameter)
                 that fell in this integration interval (t_curr,t_curr+h). These values were interpolated.
        @return h_int (float): new suggested step to take in next integration step (based on interpolation error
                estimation) if step is rejected due to interpolation large error. See ref I, II.9...
        @return f_evals (int): number of derivative evaluations performed for the interpolation
    """
    Tkk = T[k,k]
    f_Tkk = ode_fun(*(Tkk, t_curr+h) + args)
    f_evals = 1

    # Calculate interpolating polynomial
    if symmetric:
        # Do last evaluations to construct polynomials
        # they are done here to avoid extra function evaluations if
        # interpolation is not needed. These extra function evaluations are only
        # needed for the symmetric interpolation
        for j in range(1,len(T[:,1])):
            Tj1 = T[j,1]
            f_yjj = f_yj[j]
            # TODO: reuse last function evaluation to calculate next step
            f_yjj[-1] = ode_fun(*(Tj1, t_curr + h) + args)
            f_evals += 1
        poly = _interpolate_sym(yn, Tkk, f_Tkk, y_half, f_yj, hs, h, k, atol,rtol, seq)
    else:
        poly = _interpolate_nonsym(yn, Tkk, yj, hs, h, k, atol,rtol, seq)

    y_solution=[]
    old_index = t_index
    h_int = None

    # Interpolate to requested output times within the current step
    while t_index < len(t) and t[t_index] < t_curr + h:

        y_poly, errint, h_int = poly((t[t_index] - t_curr)/h)

        if (errint <= 10) or (not adaptive): # Accept, so far
            y_solution.append(y_poly)
            t_index += 1
        else: # Reject
            h = h_int
            t_index = old_index
            reject_step = True
            return (reject_step, y_solution, h_int, f_evals)

    reject_step=False
    return (reject_step, y_solution, h_int, f_evals)

def _set_default_solver_parameters(N, atol, rtol, addWork):
    """
    Set additional parameters that change the behavior of the solver.

    @param N (int): size of the ODE system
    @param atol (float): absolute tolerance required
    @param rtol (float): relative tolerance required
    @param addWork (bool): whether to add extra work to the work estimation used to compute
            next step to take. Should only be True when some form of Jacobian estimation
            or evaluation is performed.

    @return solver_parameters (dict):
             KEY            MEANING
            'min_tol'      minimum tolerance, see BLOCK 1 functions, used in implicit and semi implicit methods
            'addWork'      add extra work to work estimation, see _estimate_next_step_and_order(..)
            'freezeJac'    whether to freeze Jacobian estimation, see _get_jacobian(..)
            'iterative'    whether system solver should be iterative or exact,
                                see BLOCK 1 functions, used in semi implicit methods
            'initialGuess'
    """

    solver_parameters = {}

    # Some algorithms that this solver uses terminate when some objective
    # function is under some tolerance.
    # Then, we use the minimum tolerance (more restrictive value) as the
    # required tolerance for such algorithm
    solver_parameters['min_tol'] = min(atol,rtol)
    # TODO: this value should be set to True, but it seems that
    # for the semi-implicit methods (as pg 140, SODEX paragraph, IV.9
    # ref II suggests) that makes the solver perform much worse (try with
    # BRUSS-2D problem)
    solver_parameters['addWork'] = False

    solver_parameters['initialGuess'] = False

    # TODO: check this 15 threshold to see its viability
    # as it was chosen empirically with not enough examples
    if(N>15):
        solver_parameters['freezeJac'] = True
        solver_parameters['iterative'] = True
    else:
        solver_parameters['freezeJac'] = False
        solver_parameters['iterative'] = False

    return solver_parameters

class Solvers(object):
    EXPLICIT_MIDPOINT = 'explicit midpoint'
    IMPLICIT_MIDPOINT = 'implicit midpoint'
    SEMI_IMPLICIT_MIDPOINT = 'semi-implicit midpoint'
    EXPLICIT_EULER = 'explicit euler'
    SEMI_IMPLICIT_EULER = 'semi-implicit euler'

method_fcns = {
        Solvers.EXPLICIT_MIDPOINT : _explicit_midpoint,
        Solvers.IMPLICIT_MIDPOINT : _implicit_midpoint,
        Solvers.SEMI_IMPLICIT_MIDPOINT : _semi_implicit_midpoint,
        Solvers.EXPLICIT_EULER : _explicit_euler,
        Solvers.SEMI_IMPLICIT_EULER : _semi_implicit_euler,
        }

defaults = {
            'solver' : Solvers.EXPLICIT_MIDPOINT,
            'atol'   : 1.e-8,
            'rtol'   : 1.e-8,
            'h0'     : 0.5,
            'max_steps' : 10e4,
            'step_ratio_limit' : 2,
            'adaptive' : 'order',
            'diagnostics' : False,
            'nworkers' : None,
            'jac_fun' : None,
            'p' : 4,
            'smoothing' : 'no',
        }

symmetric_methods = (_explicit_midpoint,
                     _implicit_midpoint,
                     _semi_implicit_midpoint,
                    )

def solve(ode_fun, tspan, y0, **kwargs):
    """
    General extrapolation solver function.

    Solves the system of IVPs dy/dt = ode_fun(y, t0, ...) with parallel
    extrapolation.

    Important: the default values of the functions are set to achieve the optimal
    performance.  It is highly recommended to use such default values (this
    applies to robustness, adaptive, seq and smoothing parameters).

    Structure:

        @param ode_fun (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
        @param jac_fun (callable(y,t,args)): computes analytically the Jacobian of the ode_fun function parameter.
        @param y0 (array): initial condition on y (can be a vector).
        @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first
                element of this sequence. And the last one the final time.
        @param args (tuple): extra arguments to pass to function.
        @param diagnostics (bool): If true, count and return # of function
                evaluations.
        @param rtol, atol (float): relative tolerance and absolute tolerance;
                                   the solver attempts to ensure that the local
                                   error satisfies this at each step.
        @param h0 (float):the step size to be attempted on the first step.
        @param max_steps (int): maximum number of (internally defined) steps allowed for each
                integration point in t. Defaults to 10e4
        @param robustness (int): multiplicative factor that limits the increase and decrease of the adaptive step
                (next step vs last step length ratio is limited by this factor).
        @param smoothing (string): specifies if a smoothing step should be performed:
            -'no': no smoothing step performed
            -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
            -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II.
        @param p (int): the order of extrapolation if order is fixed, or the starting order otherwise.
        @param nworkers (int): the number of workers working in parallel. If nworkers==None, then
            the the number of workers is set to the number of CPUs on the running machine.
        @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
            The step number sequence can be changed forcefully in some cases for interpolation purposes.
            See _getDenseAndSequence(.) function.
        @param adaptive (string): specifies the strategy of integration. Can take three values:
            - "fixed" = use fixed step size and order strategy.
            - "order" or any other string = use adaptive step size and adaptive order strategy (recommended).

        @return:
            @return ys (2D-array, shape (len(t), len(y0))): array containing the value of y for each desired time in t, with
                the initial value y0 in the first row.
            @return infodict (dict): only returned if diagnostics == True. Dictionary containing additional output information
                 KEY        MEANING
                'fe_seq'    cumulative number of sequential derivative evaluations
                'nfe'       cumulative number of derivative evaluations
                'nst'       cumulative number of successful time steps
                'nje'       cumulative number of either Jacobian evaluations (when
                            analytic Jacobian is provided) or Jacobian estimations
                            (when no analytic Jacobian is provided)
                'h_avg'     average step size
                'p_avg'     average extrapolation order

        CHECK THIS:
            IMPORTANT: all input parameters should be float types; this solver
            doesn't work properly if for example y0 is not a float but an int
            variable (i.e. 1 instead of 1.) TODO: add protection so that int inputs
            are formated to floats (and algorithm works)

    """
    # Set per-method defaults
    if kwargs['solver'] == Solvers.EXPLICIT_MIDPOINT:
        kwargs.setdefault('seq', lambda t: 2*t)
    elif kwargs['solver'] == Solvers.IMPLICIT_MIDPOINT:
        kwargs.setdefault('smoothing', 'gbs')
        kwargs.setdefault('seq', lambda t : 2*(2*t-1))
    elif kwargs['solver'] == Solvers.SEMI_IMPLICIT_MIDPOINT:
        kwargs.setdefault('smoothing', 'semiimp')
        kwargs.setdefault('seq', lambda t : 2*(2*t-1))
    elif kwargs['solver'] == Solvers.EXPLICIT_EULER:
        kwargs.setdefault('seq', lambda t : t)
    elif kwargs['solver'] == Solvers.SEMI_IMPLICIT_EULER:
        kwargs.setdefault('seq', lambda t : 2*(2*t-1))

    # Set universal defaults
    for key, value in defaults.iteritems():
        if not key in kwargs.iterkeys():
            kwargs[key] = value

    if kwargs['solver'] in (Solvers.SEMI_IMPLICIT_EULER, Solvers.SEMI_IMPLICIT_MIDPOINT):
        solver_args = {}
        solver_args["I"] = np.identity(len(y0), dtype=float)
        if kwargs['jac_fun']:
            J = kwargs['jac_fun'](y0,tspan[0])
            if scipy.sparse.issparse(J): # If J is sparse, use sparse identity
                solver_args["I"] = scipy.sparse.identity(len(y0), dtype=float,
                                                         format='csr')
        else:
            solver_args["J00"] = None
    else:
        solver_args = {}


    # Fix this:
    solver_parameters = _set_default_solver_parameters(len(y0), kwargs['atol'], kwargs['rtol'], addWork=False)
    kwargs['solver_parameters'] = solver_parameters

    if kwargs['solver'] in (Solvers.EXPLICIT_EULER, Solvers.SEMI_IMPLICIT_EULER):
        kwargs['k'] = kwargs.pop('p')
    else:
        kwargs['k'] = kwargs.pop('p')//2

    #return _extrapolation_parallel(ode_fun, tspan, y0, kwargs)
    return _extrapolation_parallel(ode_fun, tspan, y0, solver_args, **kwargs)
