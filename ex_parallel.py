from __future__ import division
import numpy as np
import multiprocessing as mp
import math
from scipy import optimize
import scipy
import forward_diff

'''
IMPORTANT: this code is based basically based on two references.
    ref I: Solving Ordinary Differential Equations I: Nonstiff Problems by Hairer, Norsett and Wanner
    ref II: Solving Ordinary Differntial Equations II: Stiff and Differential-Algebraic Problems by Hairer and Wanner
'''

#This global variable does not limit multiprocessing performance
#as it is used always in the sequential part of the code.
NUM_WORKERS = None

def set_NUM_WORKERS(nworkers):
    '''
    Set number of parallel workers to be used to parallelize solver's operations
    (if None it is set to the number of processors of computer). 
    
    @param nworkers (int): number of parallel workers
    '''
    global NUM_WORKERS
    if nworkers == None:
        try:
            NUM_WORKERS = mp.cpu_count()
        except NotImplementedError:
            NUM_WORKERS = 4
    else: 
        NUM_WORKERS = max(nworkers, 1)

def _error_norm(y1, y2, atol, rtol):
    '''
    Error norm/measure between y1, y2.
    Based on II.4.11 (ref I).
    
    @param y1 (array): first value
    @param y2 (array): second value
    @param atol (float): absolute tolerance required
    @param rtol (float): relative tolerance required
    '''
    tol = atol + np.maximum(np.abs(y1),np.abs(y2))*rtol
    return np.linalg.norm((y1-y2)/tol)/(len(y1)**0.5)

'''''
BEGINNING BLOCK 1: ODE numerical methods formulas and one step solvers(explicit, implicit and semi-implicit).

Obs: semi-implicit is also called in the reference books as linearly implicit.

Methods' common structure:
def method_implicit/explicit/semi-implicit(f, grad, previousValues, previousTime,f_previousValue, step, args)
    @param f (callable f(y,t,args)): derivative of u(t) (ODE RHS, f(u,t))
    @param grad (callable grad(y,t,args)): Jacobian of f.
    @param previousValues (2-tuple) : previous solution values (two previous values, at t_n-2 and t_n-1) 
            used to obtain the next point (using two previous values because midpoint explicit
            method needs the previous two points, otherwise the t_n-2 value can remain unused)
    @param previousTime (float): time at which previous solution is found
    @param f_previousValue (array): function evaluation at the previousValue (at t_n-1), so that it can be
            reused if it was already calculated for other purposes (Jacobian estimation or dense output interpolation)
    @param step (float): step length
    @param args (tuple): extra arguments
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    @param extra arguments to be passed to some of the methods:
        @param J00 (2D array): (only for semi-implicit) Jacobian estimation at the previous stage value
                (at each extrapolation stage different number of steps are taken, depending on the
                step sequence chosen).
        
    @return (yj, f_yj, fe_tot, je_tot):
        @return yj (array): solution calculated at previousTime+step
        @return f_yj (array): function evaluation value at previousValue, previousTime
        @return fe_tot (int): number of function evaluations done in this method
        @return je_tot (int): number of Jacobian evaluations done in this method
'''''

def _solve_implicit_step(zero_f, zero_grad, estimatedValue, addSolverParam):
    '''
    Find the root of the zero_f function using as initial approximation estimatedValueExplicit.
    This zero_f root is the next step solution. The _solve_implicit_step solves one step of an implicit method.
    
    @param zero_f (callable zero_f(y,t,args)): function to find the root of (estimation of next step)
    @param zero_grad (callable zero_grad(y,t,args)): Jacobian of zero_f.
    @param estimatedValue (array): estimated value of the zero_f function to use as initial value
            for the root solver
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    
    @return (yj, fe_tot, je_tot):
        @return yj (array): solution calculated at previousTime+step (root of the zero_f)
        @return fe_tot (int): number of function evaluations done in this method
        @return je_tot (int): number of Jacobian evaluations done in this method 
    '''
    #TODO: Problem -> fsolve doesn't seem to work well with xtol parameter
    #TODO: pass to the zero finder solver the tolerance required to the overall problem 
    # by doing this the error by the solver doesn't limit the global error of the ODE's solution
    #TODO change solver so it doesn't do the 2 extra unnecessary function evaluations
    #https://github.com/scipy/scipy/issues/5369
    #TODO: add extra 2 function evaluations
    
    x, infodict, ier, mesg = optimize.fsolve(zero_f,estimatedValue, 
            fprime = zero_grad, full_output = True, xtol=addSolverParam['min_tol'])

    if("njev" in infodict):
        return (x, infodict["nfev"], infodict["njev"])
    else:
        return (x, infodict["nfev"], 0)

#     optObject = optimize.root(zero_f, estimatedValueExplicit, jac = zero_grad)
# #     if(useGrad and not zero_grad is None):
# #         print ("jac ev: " + str(optObject.njev))
# #     print ("func ev: " + str(optObject.nfev))
#     return (optObject.x, optObject.nfev)


def _midpoint_semiimplicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam, J00, I, Isparse):
    '''
    Calculates solution at previousTime+step doing one step with a midpoint semiimplicit formula (linearly implicit midpoint)
    Based on IV.9.16a-b (ref II).
    
    '''
    
    previousPreviousValue, previousValue = previousValues
    je_tot=0
       
    if(previousPreviousValue is None):
        return _euler_semiimplicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam, J00, I, Isparse)
    
    if(f_previousValue is None):
        f_yj = f(*(previousValue,previousTime)+args)
        fe_tot = 1
    else:
        f_yj = f_previousValue
        fe_tot=0
    
    if(not addSolverParam['initialGuess']):
        xval=None
    else:
        xval, f_yj, fe_tot_,je_tot=_euler_explicit(f, grad, previousValues, previousTime, f_yj, step, args, addSolverParam)
        fe_tot += fe_tot_

    b = np.dot(-(I+step*J00),(previousValue-previousPreviousValue)) + 2*step*f_yj
    
    if(scipy.sparse.issparse(J00)):
        A=_calculateMatrix(Isparse,J00,step)
        if(addSolverParam['iterative']):
            sol, info= scipy.sparse.linalg.gmres(A, b, tol=addSolverParam['min_tol'],x0=xval, maxiter=100)                             
            if info >0:
                print("Info: maximum iterations reached for sparse system solver (GMRES).")
        else:
            sol = scipy.sparse.linalg.spsolve(A, b)
            
    else:
        A=_calculateMatrix(I,J00,step)
        if(not addSolverParam['iterative']):
            sol = np.linalg.solve(A, b)
        else:
            sol, info= scipy.sparse.linalg.gmres(A, b, tol=addSolverParam['min_tol'],x0=xval, maxiter=100)    
            
    x = previousValue + sol
    
    return (x, f_yj, fe_tot, je_tot)


def _calculateMatrix(I,J00,step):
    '''
    Calculates matrix needed for semi implicit methods.
    
    @param I (2D array): identity matrix
    @param J00 (2D array): Jacobian matrix
    @param step (float): step length
    
    @return I-step*J00 (matrix).
    '''
    
    return I-step*J00

def _euler_semiimplicit(f, grad, previousValues, previousTime, f_previousValue,step, args, addSolverParam, J00, I, Isparse):
    '''
    Calculates solution at previousTime+step doing one step with a euler semiimplicit formula (linearly implicit euler)
    Based on IV.9.25 (ref II).
    
    Takes into account when solving the linearly implicit :
        -Whether the Jacobian (J00) is sparse or not
        -Whether the linear solver should be iterative (gmres) or exact
    
    IMPORTANT: changes in this function will be probably also wanted in midpoint_semiimplicit code (beware!).
    '''
    
    previousPreviousValue, previousValue = previousValues
    je_tot=0
    if(f_previousValue is None):
        f_yj = f(*(previousValue,previousTime)+args)
        fe_tot = 1
    else:
        f_yj = f_previousValue
        fe_tot=0
        
    if(not addSolverParam['initialGuess']):
        xval=None
    else:
        #TODO: The option of doing an explicit step doesn't seem to be effective (maybe because with
        # extrapolation we are taking too big steps for the explicit solver to be close to the solution).
#         xval, f_yj, fe_tot_,je_tot=_euler_explicit(f, grad, previousValues, previousTime, f_yj, step, args, addSolverParam)
#         fe_tot += fe_tot_
        xval = previousValue
    
    b=step*f_yj
    
    #TODO: change this checking of the sparsity type of the matrix only once
    # at the beginning of the ODE solving 
    if(scipy.sparse.issparse(J00)):
        A=_calculateMatrix(Isparse,J00,step)
        if(addSolverParam['iterative']):
            #TODO: choose an appropriate maxiter parameter to distribute work between taking more steps and having a
            #more accurate solution
            sol, info= scipy.sparse.linalg.gmres(A, b, tol=addSolverParam['min_tol'],x0=xval, maxiter=100)                             
            if info >0:
                print("Info: maximum iterations reached for sparse system solver (GMRES).")
        else:
            sol = scipy.sparse.linalg.spsolve(A, b)
            
    else:
        A=_calculateMatrix(I,J00,step)
        if(not addSolverParam['iterative']):
            sol = np.linalg.solve(A, b)
        else:
            sol, info= scipy.sparse.linalg.gmres(A, b, tol=addSolverParam['min_tol'],x0=xval, maxiter=100)                             
    
    x = previousValue + sol

    return (x, f_yj, fe_tot, je_tot)


def _midpoint_implicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam):
    '''
    Calculates solution at previousTime+step doing one step with a midpoint implicit
    Based on IV.9.2 (ref II).
    
    '''
    previousPreviousValue, previousValue = previousValues
    
    def zero_func(x):
        return x-previousValue-step*f(*((previousValue+x)/2, previousTime+step/2) + args)
    
    def zero_grad(x):
        return np.matrix(np.identity(len(x), dtype=float) - step*grad(x,previousTime+step/2))

    
    if(grad is None):
        zero_grad=None
        
    if(not addSolverParam['initialGuess']):
        estimatedValue = previousValue
        fe_tot = 0
        je_tot = 0
    else:
        #Estimation of the value as the starting point for the zero solver 
        estimatedValue, f_yj, fe_tot, je_tot=_euler_explicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam)
        
    x, fe_tot_, je_tot_ = _solve_implicit_step(zero_func, zero_grad, estimatedValue, addSolverParam)
    fe_tot +=fe_tot_
    je_tot += je_tot_
    
    f_yj= f(*(previousValue,previousTime)+args)
    fe_tot += 1
    return (x, f_yj, fe_tot, je_tot)


def _midpoint_explicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam):
    '''
    Calculates solution at previousTime+step doing one step with a midpoint explicit
    Based on II.9.13b (ref I).
    
    '''
    
    previousPreviousValue, previousValue = previousValues
    if(previousPreviousValue is None):
        return _euler_explicit(f, grad, previousValues, previousTime, f_previousValue,step, args, addSolverParam)
    
    f_yj = f(*(previousValue, previousTime)+args)
    fe_tot=1
    return (previousPreviousValue + (2*step)*f_yj, f_yj, fe_tot,0)

def _euler_explicit(f, grad, previousValues, previousTime,f_previousValue, step, args, addSolverParam):
    '''
    Calculates solution at previousTime+step doing one step with a euler explicit
    Based on II.9.13a (ref I).
    
    '''
        
    previousPreviousValue, previousValue = previousValues
    
    if(f_previousValue is None):
        f_yj = f(*(previousValue,previousTime)+args)
        fe_tot = 1
    else:
        f_yj = f_previousValue
        fe_tot=0
    
    return (previousValue + step*f_yj, f_yj, fe_tot,0)


'''''
END BLOCK 1: ODE numerical methods formulas (explicit, implicit and semi-implicit)
'''''

def _compute_stages((method, methodargs, func, grad, tn, yn, f_yn, args, h, j_nj_list, smoothing, addSolverParam)):
    '''
    Compute extrapolation tableau values with the order specified and number of steps specified in j_nj_list.
    It calculates the T_{k,1} values for the k's in j_nj_list.  
    
    Based on II.9.2 (Definition of the Method ref I)
    
    @param method: ODE solver method to solve one step (midpoint,euler/implicit,semiimplicit,explicit)
    @param methodargs: extra arguments to be passed to the solver method
    @param func (callable func(y,t,args)): derivative of u(t) (ODE RHS, f(u,t))
    @param grad (callable grad(y,t,args)): Jacobian of f.
    @param tn (float): initial time
    @param yn (array): solution value at tn
    @param f_yn (array): function evaluation (func) at yn,tn
    @param args (tuple): extra arguments for func
    @param h (float): big step to take to obtain T_{k,1}
    @param j_nj_list (array of 2-tuples): array with (j,nj) pairs indicating which y_{h_j}(tn+h) are calculated  
    @param smoothing (string): specifies if a smoothing step should be performed:
        -'no': no smoothing step performed
        -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II.
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
        Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    
    @return list of tuples. Each value is represents all information regarding one value of the extrapolation tableau
    first column of values T_{k,1}. the list contains:
        @return k: order of T
        @return nj: number of steps to calculate T
        @return Tj1: T_{k,1}
        @return y_half (array): intermediate solution at half the interval to T (for symmetric interpolation)
        @return f_yj (array of arrays(f_yj)): all function evaluations done at intermediate solution points (for symmetric interpolation)
        @return Y (array of arrays(yj)): all intermediate solution points (yj) calculated to obtain T (for non-symmetric interpolation)
        @return fe_tot (int): number of total function evaluations done to calculate this T
        @return je_tot (int): number of total jacobian evaluations done to calculate this T
    '''
    res = []
    for (j,nj) in j_nj_list:
        fe_tot=0
        je_tot=0
        nj = int(nj)
        Y = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        f_yj = np.zeros((nj+1, len(yn)), dtype=(type(yn[0])))
        Y[0] = yn
        step = h/nj

        Y[1], f_yj[0], fe_tot_, je_tot_ = method(func, grad, (None, Y[0]), tn, f_yn, step, args, addSolverParam, **methodargs)
        fe_tot += fe_tot_
        je_tot += je_tot_
        for i in range(2,nj+1):
            Y[i], f_yj[i-1], fe_tot_ , je_tot_= method(func, grad, (Y[i-2], Y[i-1]), tn + (i-1)*(h/nj), None, step, args, addSolverParam, **methodargs)
            fe_tot += fe_tot_
            je_tot += je_tot_
        
        #TODO: this y_half value is already returned inside Y, remove and when needed (interpolation) extract from Y
        y_half = Y[nj/2]
        
        #Perform smoothing step
        Tj1 = Y[nj]
        if(not smoothing == 'no'):
            #TODO: this f_yj_unused can be used in the case of interpolation, it should be adequately saved in
            #f_yj so function evaluations are not repeated.
            nextStepSolution, f_yj_unused, fe_tot_, je_tot_ = method(func, grad, (Y[nj-1], Y[nj]), tn + h, None, step, args, addSolverParam, **methodargs)
            fe_tot += fe_tot_
            je_tot += je_tot_
            if(smoothing == 'gbs'):
                Tj1 = 1/4*(Y[nj-1]+2*Y[nj]+nextStepSolution)
            elif(smoothing == 'semiimp'):
                Tj1 = 1/2*(Y[nj-1]+nextStepSolution)
        res += [(j, nj, Tj1, y_half, f_yj,Y, fe_tot, je_tot)]

    return res

def __extrapolation_parallel(method, methodargs, func, grad, y0, t, args=(), full_output=False,
        rtol=1.0e-8, atol=1.0e-8, h0=0.5, mxstep=10e4, robustness_factor=2, k=4,
        nworkers=None, smoothing='no', symmetric=True, seq=None, adaptive="order", addSolverParam={}):   
    '''
    Solves the system of IVPs dy/dt = func(y, t0, ...) with parallel extrapolation. 
    
    
    @param method (callable(...)): the method on which the extrapolation is based (euler,mipoint/explicit,implicit,semiimplicit)
    @param methodargs (dict): dictionary with extra parameters to be passed to the solver method.
    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param grad (callable(y,t,args)): computes the Jacobian of the func function parameter.
    @param y0 (array): initial condition on y (can be a vector).
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first 
            element of this sequence. And the last one the final time.
    @param args (tuple): extra arguments to pass to function.
    @param full_output (bool): true if user wants a dictionary of optional outputs as the second output.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param h0 (float):the step size to be attempted on the first step.
    @param mxstep (int): maximum number of (internally defined) steps allowed for each
            integration point in t. Defaults to 10e4
    @param robustness_factor (int): multiplicative factor that limits the increase and decrease of the adaptive step
            (next step vs last step length ratio is limited by this factor).
    @param k (int): the number of extrapolation steps if order is fixed, or the starting value otherwise.
    @param nworkers (int): the number of workers working in parallel. If nworkers==None, then 
        the the number of workers is set to the number of CPUs on the running machine.
    @param smoothing (string): specifies if a smoothing step should be performed:
        -'no': no smoothing step performed
        -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II. 
    @param symmetric (bool): whether the method to solve one step is symmetric (midpoint/trapezoidal)
            or non-symmetric (euler).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param adaptive (string): specifies the strategy of integration. Can take three values:
        - "fixed" = use fixed step size and order strategy.
        - "order" or any other string = use adaptive step size and adaptive order strategy (recommended).
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _getAdditionalSolverParameters(..) function.

    @return: 
        @return ys (2D-array, shape (len(t), len(y0))): array containing the value of y for each desired time in t, with 
            the initial value y0 in the first row.
        @return infodict (dict): only returned if full_output == True. Dictionary containing additional output information
             KEY        MEANING
            'fe_seq'    cumulative number of sequential derivative evaluations
            'nfe'       cumulative number of total derivative evaluations
            'nst'       cumulative number of successful time steps
            'nje'       cumulative number of either Jacobian evaluations (when 
                        analytic Jacobian is provided) or Jacobian estimations
                        (when no analytic Jacobian is provided)
            'h_avg'     average step size
            'k_avg'     average number of extrapolation steps

    '''
    
    #Initialize pool of workers to parallelize extrapolation table calculations
    set_NUM_WORKERS(nworkers)  
    pool = mp.Pool(NUM_WORKERS)

    
    assert len(t) > 1, ("the array t must be of length at least 2, " + 
    "the initial value time should be the first element of t and the last " +
    "element of t the final time")

    # ys contains the solutions at the times specified by t
    ys = np.zeros((len(t), len(y0)), dtype=(type(y0[0])))
    ys[0] = y0
    t0 = t[0]

    #Initialize values
    fe_seq = 0
    fe_tot = 0
    je_tot = 0
    nstp = 0
    cur_stp = 0

    t_max = t[-1]
    t_index = 1

    #yn is the previous calculated step (at t_curr)
    yn, t_curr = 1*y0, t0
    h = min(h0, t_max-t0)

    sum_ks, sum_hs = 0, 0
    
    #Initialize rejectStep so that Jacobian is updated
    rejectStep = True
    previousStepSolution=()

    #Iterate until you reach final time
    while t_curr < t_max:
        rejectStep, y_temp, ysolution,f_yn, h, k, h_new, k_new, (fe_seq_, fe_tot_, je_tot_) = _solve_one_step(
                method, methodargs, func, grad, t_curr, t, t_index, yn, args, h, k, 
                atol, rtol, pool, smoothing, symmetric, seq, adaptive, rejectStep, previousStepSolution,addSolverParam)
        #previousStepSolution is used for Jacobian updating
        previousStepSolution=(yn,f_yn)

        #Store values if step is not rejected
        if(not rejectStep):
            yn = 1*y_temp
            #ysolution includes all intermediate solutions in interval
            if(len(ysolution)!=0):
                ys[t_index:(t_index+len(ysolution))] = ysolution
            
            #Update time
            t_curr += h
            t_index += len(ysolution)
            #add last solution if matches an asked time (in t)
            if(t[t_index]==t_curr):
                ys[t_index] = yn
                t_index+=1

        #Update function evaluations
        fe_seq += fe_seq_
        fe_tot += fe_tot_
        je_tot += je_tot_

        sum_ks += k
        sum_hs += h
        nstp += 1
        cur_stp += 1

        if cur_stp > mxstep:
            raise Exception('Reached Max Number of Steps. Current t = ' 
                + str(t_curr))
        
        #Sometimes step can be NaN due to overflows of the RHS 
        if(math.isnan(h_new)):
            h_new = h/robustness_factor
        #Update to new step (limit the change of h_new by a robustness_factor)
        elif(h_new>h and h_new/h>robustness_factor):
            h_new = h*robustness_factor
        elif(h_new<h and h_new/h<1/robustness_factor):
            h_new = h/robustness_factor
        #Check that h_new doesn't step after t_max
        h = min(h_new, t_max - t_curr)
        #Keeps the code from taking a close to machine precision step
        if (adaptive=="fixed" and (t_max-(t_curr+h))/t_max<1e-12):
            h=t_max-t_curr
            
        k = k_new

    #Close pool of workers and return results
    pool.close()

    if full_output:
        infodict = {'fe_seq': fe_seq, 'nfe': fe_tot, 'nst': nstp, 'nje': je_tot,
                    'h_avg': sum_hs/nstp, 'k_avg': sum_ks/nstp}
        return (ys, infodict)
    else:
        return ys

def _balance_load(k, nworkers, seq=(lambda t: 2*t)):
    '''
    Distributes the workload for the different processors. The tasks to be parallelized are the calculation
    of each T_{j,1} for j=1...k. As the number of steps required to compute T_{j,1} is given by seq(j).
    This manual load-balancing can be more performant than relying on dynamic load-balancing.
    
    Each process is given a list of 2-tuples (j,seq(j)) so that the maximal sum of seq(j)'s (workloads)
    for any one process is minimized.
    
    @param k (int): Number of extrapolation steps to be performed (i.e., number of T_{j,1} values)
    @param seq  (callable(i), int i>=1): sequence of steps to take
    
    @return j_nj_list (list of lists of 2-tuples): the list contains nworkers lists each one containing which
            T_{j,1} values each processor has to calculate (each T_{j,1} is specified by the tuple (j,nj)).

    The algorithm used here is not optimal for all possible sequences, but is typically optimal.
    Basically, it gives one task to each process in some order, then reverses the list of processes
    and repeats.
    '''
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
    Determine the maximum speedup that could possibly be achieved by using nworkers instead of just 1.
    
    This is only used for evaluating performance.

    TODO: What is k here? is it the number of extrapolation steps or the order of extrapolation?
    If it is the order of extrapolation, then this function if not correct as _balance_load takes in 
    number of extrapolation steps
    """
    steps = [seq(j) for j in range(1,k+1)]
    serial_cost = sum(steps)
    j_nj_list = _balance_load(k, nworkers, seq)
    max_work = 0
    for proc_worklist in j_nj_list:
        work = sum([nj for j, nj in proc_worklist])
        max_work = max(max_work, work)
    return serial_cost / float(max_work)


def _updateJ00(previousJ00,func, yn, tn, yn_1, f_yn_1, args):
    '''
    Updates the previous Jacobian estimation without doing just one extra function evaluation
    (which will be reused for next step calculations). It uses the update formula suggested for
    Broyden's method.

    @param previousJ00 (2D-array): Jacobian approximation at the previous time 
    @param func (callable(y,t,...)):RHS of the ODE
    @param yn (array): solution at tn (current time)
    @param tn (array): current time
    @param yn_1 (array): solution at tn-1 (previous stage time)
    @param f_yn_1 (array): function evaluation at yn_1,tn_1
    @param args (tuple): extra arguments for func
    
    @return (updatedJ00,f_yn):
        @return updatedJ00(2D-array): Jacobian approximation updated 
        @return f_yn(array): function evaluation at yn,tn
    
    '''
    #TODO: check if this updating is effective and if so, remove the next
    #two lines of code
    updatedJ00 = previousJ00
    return (updatedJ00,None)

    f_yn = func(*(yn,tn)+args)
    incf_yn = f_yn-f_yn_1
    incyn = yn-yn_1
    updatedJ00 = previousJ00 +np.outer((incf_yn-np.dot(previousJ00,incyn))/np.linalg.norm(incyn, 2),incyn)
    return (updatedJ00,f_yn)

previousJ00 = 0
def _getJacobian(func, args, yn, tn, grad, methodargs, rejectPreviousStep, previousStepSolution,addSolverParam): 
    '''
    Obtains the Jacobian approximation at yn,tn of func. Different possibilities:
        - If grad (analytical Jacobian) is available grad is used to obtain the Jacobian at yn,tn
        - Otherwise (if not freeze) the Jacobian is estimated with a forward difference formula (see forward_diff.Jacobian)
        - Otherwise (if freeze) the last Jacobian approximation is used (unless the previous step was rejected).
            This freezing idea was taken from LSODE solver.
    
    freeze is a parameter in dictionary addSolverParam with the name freezeJac
    
    @param func (callable(y,t,args)):RHS of the ODE
    @param args (tuple): extra arguments for func
    @param yn (array): solution at tn (current time)
    @param tn (array): current time
    @param grad (callable(y,t,args)): computes analytically the Jacobian of the func function parameter.
    @param methodargs (dict): contains solver methods' additional parameters.
            In this function methodargs['J00'] is updated with the Jacobian estimation at yn,tn
    @param rejectPreviousStep (bool): whether previously taken step was rejected or not 
    @param previousStepSolution (2-tuple): tuple containing the solution at the previous step (tn-1) and its
            function evaluation, (yn_1, f_yn_1)    
    @return (f_yn, fe_tot,je_tot):
        @return f_yn (array): function evaluation at yn,tn
        @return fe_tot (int): number of function evaluations (0 if analytical Jacobian, N if estimated Jacobian)
        @return je_tot (int): number of Jacobian evaluations (1 if analytical Jacobian) 
            or Jacobian estimations (1 if estimated Jacobian)
    
    Note that the Jacobian itself is not explicitly returned; it is stored as an entry in methodargs.
    '''
    je_tot=0
    fe_tot=0
    f_yn=None
    #methodargs is an empty dictionary if a Jacobian estimation/evaluation is needed (semi implicit methods)
    if(not methodargs=={}):
        def func_at_tn(y, args):
            return func(*(y,tn)+args)
        if(grad is None):
            
            global previousJ00
            if(addSolverParam['freezeJac'] and not rejectPreviousStep):
                yn_1, f_yn_1 = previousStepSolution
                updatedJ00, f_yn = _updateJ00(previousJ00,func, yn, tn, yn_1, f_yn_1, args)
                methodargs['J00']=updatedJ00
                return (f_yn, fe_tot,je_tot)
    
            f_yn = func_at_tn(yn,args)
            fe_tot += 1
            J00,fe_tot_ = forward_diff.Jacobian(func_at_tn,yn, f_yn, args)
            fe_tot += fe_tot_
            je_tot=1
            methodargs['J00'] =J00 
            previousJ00 = J00
        else:            
            J00 = grad(yn, tn)
            je_tot = 1
            fe_tot = 0
            methodargs['J00'] =J00 
    return (f_yn, fe_tot,je_tot) 

def _compute_extrapolation_table(method, methodargs, func, grad, tn, yn, args, h, k, pool, 
                            rejectPreviousStep,previousStepSolution, seq, smoothing, symmetric, addSolverParam):
    '''
    Computes the extrapolation tableau for a given big step, order and step sequence. It parallelizes the computation
    of each T_{1,i} taking all the inner steps necessary and then extrapolates the final value at tn+h.
    
    @param method (callable(...)): the method on which the extrapolation is based (euler,mipoint/explicit,
            implicit,semiimplicit)
    @param methodargs (dict): dictionary with extra parameters to be passed to the solver method.
    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param grad (callable(y,t,args)): computes the Jacobian of the func function parameter.
    @param tn (float): starting integration time
    @param yn (array): solution at tn (current time)
    @param args (tuple): extra arguments to pass to function.
    @param h (float): integration step to take (the output, without interpolation, will be calculated at t_curr+h)
            This value matches with the value H in ref I and ref II.
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).
    @param pool: multiprocessing pool of workers (with as many workers as processors) that will parallelize the
            calculation of each of the initial values of the extrapolation tableau (T_{i,1} i=1...k).
    @param rejectPreviousStep (bool): whether previously taken step was rejected or not 
    @param previousStepSolution (2-tuple): tuple containing the solution at the previous step (tn-1) and its
            function evaluation, (yn_1, f_yn_1) 
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param smoothing (string): specifies if a smoothing step should be performed:
        -'no': no smoothing step performed
        -'gbs': three point smoothing step (based on GBS method), II.9.13c ref I and IV.9.9 ref II.
        -'semiimp': two point smoothing step (for semiimplicit midpoint), IV.9.16c ref II. 
    @param symmetric (bool): whether the method to solve one step is symmetric (midpoint/trapezoidal)
            or non-symmetric (euler).
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    
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
        @return (fe_seq,fe_tot,je_tot):
            @return fe_seq (int): cumulative number of sequential derivative evaluations performed for this step
            @return fe_tot (int): cumulative number of total derivative evaluations performed for this step
            @return je_tot (int): cumulative number of either Jacobian evaluations (when analytic Jacobian is 
                    provided) or Jacobian estimations (when no analytic Jacobian is provided) performed 
                    for this step

    '''
    T = np.zeros((k+1,k+1, len(yn)), dtype=(type(yn[0])))
    j_nj_list = _balance_load(k, NUM_WORKERS, seq=seq)
    
    f_yn, fe_tot, je_tot = _getJacobian(func, args, yn, tn, grad, methodargs,
                                        rejectPreviousStep, previousStepSolution,
                                        addSolverParam)
    
    jobs = [(method, methodargs, func, grad, tn, yn, f_yn, args, h, k_nj, smoothing, addSolverParam) for k_nj in j_nj_list]
    results = pool.map(_compute_stages, jobs, chunksize=1)

    # Here's how this would be done in serial:
    # res = _compute_stages((method, methodargs, func, grad, tn, yn, f_yn, args, h, j_nj_list, smoothing))
    
    # At this stage fe_tot has only counted the function evaluations for the jacobian estimation
    # (which are not parallelized)
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
        #Count the maximum number of sequential 
        #function evaluations taken
        if(fe_tot_stage_max<fe_tot_stage):
            fe_tot_stage_max = fe_tot_stage 
    
    fe_seq += fe_tot_stage_max
    _fill_extrapolation_table(T, k, 0, seq, symmetric)
    
    return (T, y_half, f_yj, yj, f_yn, hs,(fe_seq, fe_tot, je_tot))

def _fill_extrapolation_table(T, k, j_initshift, seq, symmetric):
    '''
    Fill extrapolation table using the first column values T_{i,1}. This function obtains the rest
    of values T_{i,j} (lower triangular values). 
    
    The formula to extrapolate the T_{i,j} values takes into account if the method to compute T_{i,1}
    was symmetric or not (II.9. and II.9. ref I)
    
    @param T (2D array): extrapolation table (lower triangular) containing
            the first column calculated and to be filled
    @param k (int): table size
    @param j_initshift (int): step sequence index matching the number of steps taken to compute the first
            value T_{1,1} of this extrapolation tableau. I.e. if step sequence is {2,6,10,14...} and
            T_{1,1} was calculated with 6 steps then j_initshift=1 and T_{i,1} are assumed to be calculated
            with seq(i+j_initshift), thus {6,10,14...}
    @param seq (callabel(i) int i>=1): step sequence of the first column values
    @param symmetric (bool): whether the method used to compute the first column of T is symmetric or not.
    
    '''
    # compute extrapolation table 
    for i in range(2, k+1):
        for j in range(i, k+1):
            if(symmetric):
                T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j+j_initshift)/seq(j+j_initshift-i+1))**2 - 1)
            else:
                T[j,i] = T[j,i-1] + (T[j,i-1] - T[j-1,i-1])/((seq(j+j_initshift)/seq(j+j_initshift-i+1)) - 1)        
            
def _centered_finite_diff(j, f_yj, hj):
    '''
    Computes through the centered differentiation formula different order derivatives of y for a given 
    extrapolation step (for every T{1,j}). In other words, it calculates II.9.39 ref I (step 1) for all kappa
    for a fixed j (for a fixed extrapolation step).
    
    Used for _interpolate_sym.
    
    @param j (int): which extrapolation inner-stage is used  
    @param f_yj (2D array): array with all the function evaluations of all the intermediate solution 
            values obtained to calculate each T_{j,1}. 
    @param hj (array): inner step taken in the j-th extrapolation step, H/nj (II.9.1 ref I).
    
    @return dj (2D array): array containing for each kappa=1...2j the kappa-th derivative of y estimated
        using the j-th extrapolation step values

    '''

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

def _backward_finite_diff(j, yj, hj, lam):
    '''
    Computes through the backward differentiation formula different order derivatives of y for a given 
    extrapolation step (for every T{1,j}). In other words, it calculates VI.5.43 ref II (step 1) for all kappa
    for a fixed j (for a fixed extrapolation step).
    
    Used for _interpolate_nonsym.
    
    @param j (int): which extrapolation inner-stage is used  
    @param yj (2D array): array with all the intermediate solution values obtained to calculate each T_{j,1}. 
    @param hj (array): inner step taken in the j-th extrapolation step, H/nj (II.9.1 ref I).
    @param lam (int): either 0 or 1, check definition and use in ref II pg 439
    
    @return rj (2D array): array containing for each kappa=1...j-lam the kappa-th derivative of y estimated
        using the j-th extrapolation step values

    '''
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
    '''
    Computes through the backward differentiation formula and using extrapolation, different order derivatives
    of y. In other words, it delegates VI.5.43 ref II (step 1) to _backward_finite_diff(.) function and here it 
    performs the extrapolation of the derivatives. It follows VI.5.44 ref II (step 2).
    
    Used for _interpolate_nonsym.
    
    Obs: compared to the reference, k and kappa are interchanged in this code, to maintain the same notation as 
    ref I (see _compute_ds function)
    
    Contains a parameter to be chosen: lam={0,1}.
    
    lam=1 requires less work and the order of interpolation is enough
    given the global error committed. Theorem VI.5.7 ref II (interpolation error).
    
    @param yj (3D array): array containing for each extrapolation value (1...k) an array with all the intermediate solution 
            values obtained to calculate each T_{i,1}. 
    @param hs (array): array containing for each extrapolation value (1...k) the inner step taken, H/nj (II.9.1 ref I) 
    @param k (int): number of extrapolation steps to take in this step (determines the number of extrapolations performed
            to achieve a better integration output, equivalent to the size of the extrapolation tableau).    
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).

    @return rs (2D array): array containing for each kappa=1...k the kappa-th derivative of y 
            (extrapolated k-kappa-lam times)

    '''
    lam=1
    dj_kappa = np.zeros((k+1-lam, k+1), dtype=(type(yj[1][0])))
    rs = np.zeros((k+1-lam), dtype=(type(yj[1][0])))
    
    for j in range(1+lam,k+1):
        dj_ = _backward_finite_diff(j,yj[j], hs[j],lam)
        for kappa in range(1,j+1-lam):    
            dj_kappa[kappa,j] = 1*dj_[kappa]
    
    for kappa in range(1,k+1-lam):
        numextrap = k+1-kappa-lam
        T = np.zeros((numextrap+1, numextrap+1), dtype=(type(yj[1][0])))
        T[:,1] = 1*dj_kappa[kappa, (kappa+lam-1):]
        _fill_extrapolation_table(T, numextrap, lam+kappa-1, seq, symmetric=False)

        rs[kappa] = 1*T[numextrap,numextrap] 

    return rs 

def _compute_ds(y_half, f_yj, hs, k, seq=(lambda t: 4*t-2)):
    '''
    Computes through the centered differentiation formula and using extrapolation, different order derivatives
    of y. In other words, it delegates II.9.39 ref I (step 1) to _backward_finite_diff(.) function and here it 
    performs the extrapolation of the derivatives. It follows II.9 step 2 ref I.
    
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

    '''
    dj_kappa = np.zeros((2*k+1, k+1), dtype=(type(y_half[1])))
    ds = np.zeros((2*k+1), dtype=(type(y_half[1])))
    
    for j in range(1,k+1):
        dj_kappa[0,j] = 1*y_half[j]
        dj_ = _centered_finite_diff(j,f_yj[j], hs[j])
        for kappa in range(1,2*j+1):    
            dj_kappa[kappa,j] = 1*dj_[kappa]
    
    #skip value is used because extrapolation is required
    #(k-l) times for dj_(2l-1) and dj_(2l)
    skip = 0
    for kappa in range(2*k+1):
        numextrap = k-int(skip/2)
        T = np.zeros((numextrap+1, numextrap+1), dtype=(type(y_half[1])))
        T[:,1] = 1*dj_kappa[kappa, int(skip/2):]
        
        _fill_extrapolation_table(T, numextrap, int(skip/2), seq, symmetric=True)
        
        ds[kappa] = 1*T[numextrap,numextrap]
        if (kappa != 0):
            skip +=1

    return ds 

def _getPolynomial(a_u,a_u_1,H,degree,pol_shift,atol,rtol):
    '''
    Get interpolation polynomial with a_u coefficients and adjust variable to problem with pol_shift.
    It returns a polynomial that returns for every theta in (0,1) the interpolation value at x0+theta*H.
    See theorem II.9.5 ref I and theorem VI.5.7 ref II.
    
    @param a_u (array): coefficients of the interpolation polynomial
    @param a_u_1 (array): coefficients of the interpolation polynomial of one degree less (for error estimation)
    @param H (float): step taken by solver (coefficients a_u and a_u_1 were calculated on (0,1) interval)
    @param degree (int): degree of polynomial a_u (should match len(a_u)-1)
    @param pol_shift (float): variable change that was made to calculate a_u and a_u_1, i.e. x-pol_shift,
            because coefficients a_u and a_u_1 were calculated on (0-pol_shift,1-pol_shift) interval.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    
    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function)
    '''
    
    def poly (t):
        '''
        Interpolation polynomial
        
        @param t(float): value in interval (0,1) 
        
        @return (res, errint, h_int):
            @return res (float): interpolation result at x0+theta*H, P(theta).
                    See theorem II.9.5 ref I and theorem VI.5.7 ref II.
            @return errint (float): estimated interpolation error (uses the one degree less polynomial to 
                    estimate the interpolation error). See II.9.45 ref I.
            @return h_int (float): suggested next step (in case of rejected interpolation).
                    See formula after II.9.45 ref I. 
        
        '''
        res = 1*a_u[0] 
        for i in range(1, len(a_u)):
            res += a_u[i]*((t-pol_shift)**i)
        
        res_u_1 = 1*a_u_1[0] 
        for i in range(1, len(a_u_1)):
            res_u_1 += a_u_1[i]*((t-pol_shift)**i)
        
        errint = _error_norm(res, res_u_1, atol, rtol)

        h_int = H*((1/errint)**(1/degree))
        
        return (res, errint, h_int)

    return poly

def _interpolate_nonsym(y0, Tkk, yj, hs, H, k, atol, rtol,
        seq=(lambda t: 4*t-2)):
    '''
    Non symmetrical formula (for example used for euler's method) to interpolate dense output values. It
    calculates a polynomial to interpolate any value from t0 (time at y0) to t0+H (time at Tkk). Based on
    Dense Output, VI.5 pg 438-439.
    
    Returns a polynomial that fulfills the conditions at VI.5.44 (step 3). To take into account: this coefficients
    were calculated for the shifted polynomial with x -> x-1.
    
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

    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function in _getPolynomial(.))

    '''
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
        sumcoeff+=(-1)**i*a_u[i]

    a_u[u] = 1/(-1)**u*(y0-a_u[0]-sumcoeff)
    
    a_u_1[0:u_1]=1*a_u[0:u_1];
    a_u_1[u_1]=1/(-1)**u_1*(y0-a_u_1[0]-sumcoeff+(-1)**u_1*a_u[u_1])
    
    return _getPolynomial(a_u,a_u_1,H,u,1,atol,rtol)
    

def _interpolate_sym(y0, Tkk, f_Tkk, y_half, f_yj, hs, H, k, atol, rtol,
        seq=(lambda t: 4*t-2)):
    '''
    Symmetrical formula (for example used for midpoint's method) to interpolate dense output values. It
    calculates a polynomial to interpolate any value from t0 (time at y0) to t0+H (time at Tkk). Based on
    Dense Output for the GBS Method, II.9 pg 237-239.
    
    Returns a polynomial that fulfills the conditions at II.9.40 (step 3). To take into account: this coefficients
    were calculated for the shifted polynomial with x -> x-1/2.
    
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

    @return poly (callable(t)): interpolation polynomial (see definition of poly(t) function in _getPolynomial(.))

    '''
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
                );
        
        
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
    # also returns the interpolation error (errint). If errint > 10, then reject
    # step 
    return _getPolynomial(a_u,a_u_1,H,u+4,0.5,atol,rtol)
    

def _getDenseAndSequence(t_final, t, t_index, seq, symmetric):
    '''
    Returns whether dense output is needed because an intermediate solution
    is wanted at this step. Also chooses the step sequence to use for this step. The step sequence
    is only forcefully changed if dense output is required and the method is symmetric (symmetric
    interpolation requires a step sequence that fulfills: II.9.sth ref I).
    
    @param t_final (float): final time after step is taken
    @param t (array): all times when output is requested
    @param t_index (int): index of t (array) at which next output value is requested
            (all values at index<t_index have already been computed).
    @param seq (callable(i), int i>=1)): the step-number sequence used to compute T.
    
    @return 
        @return dense : whether at the interval currently calculated there is intermediate solutions
            to be calculated by dense output
        @return seq : step sequence to take for this step
 
    '''
    
    dense=True
    #See if any intermediate points are asked in the interval we are
    #calculating (the value at t_final is not interpolated)

    timeOutRange = t[t_index]>=t_final
    
    if(timeOutRange):
        dense=False
    
    if dense and symmetric:
        seq = lambda t: 2*(2*t-1)     # {2,6,10,14,...} sequence for dense output
        
    return (dense,seq)


def _estimate_next_step_and_order(T, k, h, atol, rtol, seq, adaptive, addSolverParam): 
    '''
    Estimates next step and order, and whether to reject step, from the results
    obtained by the solver and the tolerance asked
    
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
    @param adaptive: string
        Specifies the strategy of integration. Can take two values:
        -- "fixed"  = use fixed step size and fixed order strategy.
        -- "order" = use adaptive step size and adaptive order strategy.
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
        Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    
    @return
        @return rejectStep : whether to reject the step and the solution obtained
            (because it doesn't satisfy the asked tolerances)
        @return y : best solution at the end of the step
        @return h_new : new step to take
        @return k_new : new number of extrapolation steps to use
 
    '''
    
    if(adaptive=="fixed"):
        return (False, T[k,k], h, k)
    
    sizeODE=0
    if(addSolverParam['addWork']):
        sizeODE = len(T[k,k])
        
    #Define work function (to minimize)      
    def A_k(k):
        """
           Expected time to compute k lines of the extrapolation table,
           in units of RHS evaluations.
        """
        sum_ = 0
        for i in range(k):
            sum_ += seq(i+1)
        #sizeODE is the length of the ODE
        return max(seq(k), sum_/NUM_WORKERS)+sizeODE # The second value is only an estimate

    H_k = lambda h, k, err_k: h*0.94*(0.65/err_k)**(1/(2*k-1)) 
    W_k = lambda Ak, Hk: Ak/Hk
    
     # compute the error and work function for the stages k-2, k-1 and k
    err_k_2 = _error_norm(T[k-2,k-3], T[k-2,k-2], atol, rtol)
    err_k_1 = _error_norm(T[k-1,k-2], T[k-1,k-1], atol, rtol)
    err_k   = _error_norm(T[k,k-1],   T[k,k],     atol, rtol)
    h_k_2   = H_k(h, k-2, err_k_2)
    h_k_1   = H_k(h, k-1, err_k_1)
    h_k     = H_k(h, k,   err_k)
    w_k_2   = W_k(A_k(k-2), h_k_2)
    w_k_1   = W_k(A_k(k-1), h_k_1)
    w_k     = W_k(A_k(k),   h_k)


    #Order and Step Size Control (II.9 Extrapolation Methods),
    #Solving Ordinary Differential Equations I (Hairer, Norsett & Wanner)
    if err_k_1 <= 1:
        # convergence in line k-1
        if err_k <= 1:
            y = T[k,k]
        else:
            y = T[k-1,k-1]

        k_new = k if w_k_1 < 0.9*w_k_2 else k-1
        h_new = h_k_1 if k_new <= k-1 else h_k_1*A_k(k)/A_k(k-1)
            
        rejectStep=False

    elif err_k <= 1:
        # convergence in line k
        y = T[k,k]

        k_new = k-1 if w_k_1 < 0.9*w_k else (
                k+1 if w_k < 0.9*w_k_1 else k)
        h_new = h_k_1 if k_new == k-1 else (
                h_k if k_new == k else h_k*A_k(k+1)/A_k(k))
            
        rejectStep=False
    else: 
        # no convergence
        # reject (h, k) and restart with new values accordingly
        k_new = k-1 if w_k_1 < 0.9*w_k else k
        h_new = min(h_k_1 if k_new == k-1 else h_k, h)
        y=None
        rejectStep=True
    
    #Extra protection for the cases where there has been an
    #overflow. Make sure the step is rejected (there is cases
    #where err_k_x is not nan
    if(math.isnan(h_new)):
        rejectStep = True
        
    return (rejectStep, y, h_new, k_new)


def _solve_one_step(method, methodargs, func, grad, t_curr, t, t_index, yn, args, h, k, atol, rtol, 
                   pool, smoothing, symmetric, seq, adaptive, rejectPreviousStep, previousStepSolution, addSolverParam):
    '''
    Solves one 'big' H step of the ODE (with all its inner H/nj steps and the extrapolation). In other words, 
    solve one full stage of the problem (one step of parallel extrapolation) and interpolates all the dense 
    output values required. 
    
    
    @param method (callable(...)): the method on which the extrapolation is based (euler,mipoint/explicit,
            implicit,semiimplicit)
    @param methodargs (dict): dictionary with extra parameters to be passed to the solver method.
    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param grad (callable(y,t,args)): computes the Jacobian of the func function parameter.
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
    @param symmetric (bool): whether the method to solve one step is symmetric (midpoint/trapezoidal)
            or non-symmetric (euler).
    @param seq (callable(i), int i>=1): the step-number sequence (examples II.9.1 , 9.6, 9.35 ref I).
    @param adaptive (string): specifies the strategy of integration. Can take three values:
        - "fixed" = use fixed step size and order strategy.
        - "order" or any other string = use adaptive step size and adaptive order strategy (recommended).
    @param rejectPreviousStep (bool): whether previously taken step was rejected or not 
    @param previousStepSolution (2-tuple): tuple containing the solution at the previous step (tn-1) and its
            function evaluation, (yn_1, f_yn_1)
    @param addSolverParam (dict): extra arguments needed to define completely the solver's behavior.
            Should not be empty, for more information see _getAdditionalSolverParameters(..) function.
    
    @return (rejectStep, y, y_solution,f_yn, h, k, h_new, k_new, (fe_seq, fe_tot, je_tot)):
        @return rejectStep (bool): whether this step should be rejected or not. True when this step was not successful
                (estimated error of the final solution or the interpolation solutions too large for the tolerances 
                required) and has to be recalculated (with a new step size h_new and order k_new).
        @return y (array): final solution obtained for this integration step, at t_curr+h.
        @return y_solution (2D array): array containing the solution for every output value required (in t parameter)
                 that fell in this integration interval (t_curr,t_curr+h). This values were interpolated.
        @return f_yn (array): function evaluation at yn (initial point solution)
        @return h (float): step taken in this step
        @return k (int): order taken in this step
        @return h_new (float): new suggested step to take in next integration step
        @return k_new (int): new suggested number of extrapolation steps in next integration step
        @return (fe_seq,fe_tot,je_tot):
            @return fe_seq (int): cumulative number of sequential derivative evaluations performed for this step
            @return fe_tot (int): cumulative number of total derivative evaluations performed for this step
            @return je_tot (int): cumulative number of either Jacobian evaluations (when analytic Jacobian is 
                    provided) or Jacobian estimations (when no analytic Jacobian is provided) performed 
                    for this step
    
    '''
    
    dense, seq = _getDenseAndSequence(t_curr+h, t, t_index, seq, symmetric)
    
    #Limit k, order of extrapolation
    #If order and step are fixed, do not limit order
    if(not adaptive == 'fixed'):
        k_max = 10
        k_min = 3
        k = min(k_max, max(k_min, k))

    T, y_half, f_yj,yj, f_yn, hs, (fe_seq, fe_tot, je_tot) = _compute_extrapolation_table(method, methodargs, func, grad, 
                t_curr, yn, args, h, k, pool, rejectPreviousStep, previousStepSolution, seq, smoothing, symmetric,addSolverParam)
    
    rejectStep, y, h_new, k_new = _estimate_next_step_and_order(T, k, h, atol, rtol, seq, adaptive, addSolverParam)
    
    y_solution=[]
    if((not rejectStep) & dense):
        rejectStep, y_solution, h_int, (fe_tot_, fe_seq_) = _interpolate_values_at_t(func, args, T, k, t_curr, t, t_index, h, hs, y_half, f_yj,yj, yn, 
                                                                         atol, rtol, seq, adaptive, symmetric)
        fe_tot += fe_tot_
        fe_seq += fe_seq_
        
        if(not adaptive=="fixed"):
            if(rejectStep):
                h_new = 1*h_int
                #Use same order if step is rejected by the interpolation (do not use the k_new of the adapted order)
                k_new = 1*k  
            elif((h_int is not None) and h_int<h_new):
                h_new = 1*h_int
            

    return (rejectStep, y, y_solution,f_yn, h, k, h_new, k_new, (fe_seq, fe_tot, je_tot))


def _interpolate_values_at_t(func, args, T, k, t_curr, t, t_index, h, hs, y_half, f_yj,yj, yn,
                            atol, rtol, seq, adaptive, symmetric):
    '''
    This function calculates all the intermediate solutions asked as dense output (in parameter t) that fall in this
    integration step. It generates an interpolation polynomial and it calculates all the required solutions. If 
    the interpolation error is not good enough the step can be forced to be rejected.
    
    This operation has to be done once the extrapolation "big" step has been taken.
    
    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
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
    @param adaptive (string): specifies the strategy of integration. Can take three values:
        - "fixed" = use fixed step size and order strategy.
        - "order" or any other string = use adaptive step size and adaptive order strategy (recommended).
    @param symmetric (bool): whether the method to solve one step is symmetric (midpoint/trapezoidal)
            or non-symmetric (euler).
    
    @return (rejectStep, y_solution, h_int, (fe_tot, fe_seq)):
        @return rejectStep (bool): whether this step should be rejected or not. True when the interpolation was not successful
                for some of the interpolated values (estimated error of the interpolation too large for the tolerances 
                required) and the step has to be recalculated (with a new step size h_int).
        @return y_solution (2D array): array containing the solution for every output value required (in t parameter)
                 that fell in this integration interval (t_curr,t_curr+h). This values were interpolated.
        @return h_int (float): new suggested step to take in next integration step (based on interpolation error
                estimation) if step is rejected due to interpolation large error. See ref I, II.9...
        @return (fe_tot,fe_seq):
            @return fe_tot (int): cumulative number of total derivative evaluations performed for the interpolation
            @return fe_seq (int): cumulative number of sequential derivative evaluations performed for the interpolation
    
    '''
    
    fe_tot = 0
    fe_seq = 0    
        
    Tkk = T[k,k]
    f_Tkk = func(*(Tkk, t_curr+h) + args)
    fe_seq +=1
    fe_tot +=1
    
    #Calculate interpolating polynomial
    if(symmetric):
        #Do last evaluations to construct polynomials
        #they are done here to avoid extra function evaluations if interpolation is not needed
        #This extra function evaluations are only needed for the symmetric interpolation
        for j in range(1,len(T[:,1])):
            Tj1=T[j,1]
            f_yjj = f_yj[j]
            #TODO: reuse last function evaluation to calculate next step
            f_yjj[-1] = func(*(Tj1, t_curr + h) + args)
            fe_tot+=1
            fe_seq +=1
        poly = _interpolate_sym(yn, Tkk, f_Tkk, y_half, f_yj, hs, h, k, atol,rtol, seq)
    else:
        poly = _interpolate_nonsym(yn, Tkk, yj, hs, h, k, atol,rtol, seq)

    y_solution=[]
    old_index = t_index
    h_int=None

    #Interpolate values at the asked t times in the current range calculated
    while t_index < len(t) and t[t_index] < t_curr + h:
            
        y_poly, errint, h_int = poly((t[t_index] - t_curr)/h)
        
        if adaptive=="fixed":
            y_solution.append(1*y_poly)
            cur_stp = 0
            t_index += 1
        elif errint <= 10:
            y_solution.append(1*y_poly)
            cur_stp = 0
            t_index += 1
        else:
            h = h_int
            t_index = old_index
            rejectStep=True
            return (rejectStep, y_solution, h_int, (fe_tot, fe_seq))
             
    rejectStep=False
    return (rejectStep, y_solution, h_int, (fe_tot, fe_seq))

def _getAdditionalSolverParameters(N,atol,rtol,addWork):
    '''
    Set additional parameters that change slightly the behavior of the solver.
    See each parameter use to understand their behavior.
    
    
    @param N (int): size of the ODE system
    @param atol (float): absolute tolerance required
    @param rtol (float): relative tolerance required
    @param addWork (bool): whether to add extra work to the work estimation used to compute
            next step to take. Should only be True when some form of Jacobian estimation
            or evaluation is performed.
                        
    @return addSolverParam (dict):
             KEY            MEANING
            'min_tol'      minimum tolerance, see BLOCK 1 functions, used in implicit and semi implicit methods
            'addWork'      add extra work to work estimation, see _estimate_next_step_and_order(..)
            'freezeJac'    whether to freeze Jacobian estimation, see _getJacobian(..)
            'iterative'    whether system solver should be iterative or exact, 
                                see BLOCK 1 functions, used in semi implicit methods
            'initialGuess' 

    '''
    
    addSolverParam = {}
    
    # Some algorithms that this solver uses terminate when some objective function is under some tolerance.
    # Then, we use the minimum tolerance (more restrictive value) as the required tolerance for such algorithm
    addSolverParam['min_tol'] = min(atol,rtol)
    #TODO: this value should be set to addWork, but it seems that setting it to True
    # for the semi implicit methods (as pg 140, SODEX paragraph, IV.9 ref II suggests)
    # makes the solver perform much worse (try with BRUSS-2D problem)
    addSolverParam['addWork'] = False
    
    addSolverParam['initialGuess'] = False
    
    #TODO: check this 15 threshold to see it's viability
    #as it was chosen empirically with not enough examples
    if(N>15):
        addSolverParam['freezeJac'] = True
        addSolverParam['iterative'] = True
    else:
        addSolverParam['freezeJac'] = False
        addSolverParam['iterative'] = False
    
    return addSolverParam

'''
BEGINNING BLOCK 2: General extrapolation solvers' functions. These functions can be used to solve any ODE.
These are recommended to use for developing and testing purposes
    
Each function solves the system of IVPs dy/dt = func(y, t0, ...) with parallel extrapolation.

Important: the default values of the functions are set to achieve the optimal performance, it is highly
recommended to use such default values (this applies to robustness, adaptive, seq and smoothing parameters).

All this functions have the same structure, and their names explicitly say the type of solver method used
(midpoint,euler/explicit,semiimplicit,implicit). Structure:

    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param grad (callable(y,t,args)): computes analytically the Jacobian of the func function parameter.
    @param y0 (array): initial condition on y (can be a vector).
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first 
            element of this sequence. And the last one the final time.
    @param args (tuple): extra arguments to pass to function.
    @param full_output (bool): true if user wants a dictionary of optional outputs as the second output.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param h0 (float):the step size to be attempted on the first step.
    @param mxstep (int): maximum number of (internally defined) steps allowed for each
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
        @return infodict (dict): only returned if full_output == True. Dictionary containing additional output information
             KEY        MEANING
            'fe_seq'    cumulative number of sequential derivative evaluations
            'nfe'       cumulative number of total derivative evaluations
            'nst'       cumulative number of successful time steps
            'nje'       cumulative number of either Jacobian evaluations (when 
                        analytic Jacobian is provided) or Jacobian estimations
                        (when no analytic Jacobian is provided)
            'h_avg'     average step size
            'p_avg'     average extrapolation order


    CHECK THIS:
        IMPORTANT: all input parameters should be float types, this solver doesn't work
        properly if for example y0 is not a float but an int variable (i.e. 1 instead of 1.)
        TODO: add protection so that int inputs are formated to floats (and algorithm works)

'''

def ex_midpoint_explicit_parallel(func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, robustness=2, smoothing = 'no', seq=(lambda t: 2*t), p=4, 
        nworkers=None, adaptive="order"):
    ''' 
    Parallel extrapolation with midpoint explicit method
    
    '''
    
    method = _midpoint_explicit
    
    addSolverParam = _getAdditionalSolverParameters(len(y0), atol, rtol, addWork=False)
    
    k=p//2

    if full_output:
        (ys, infodict) = __extrapolation_parallel(method,  {}, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
             k=k, nworkers=nworkers, smoothing=smoothing, symmetric=True, seq=seq, adaptive=adaptive,
             addSolverParam=addSolverParam)
        infodict['p_avg'] = 2*infodict.pop('k_avg')
        return (ys, infodict) 
    else:       
        return __extrapolation_parallel(method,  {}, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
            k=k, nworkers=nworkers, smoothing=smoothing, symmetric=True, seq=seq, adaptive=adaptive,
            addSolverParam=addSolverParam)



def ex_midpoint_implicit_parallel(func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, robustness=2, smoothing = 'gbs', seq=(lambda t: 2*(2*t-1)), p=4,
        nworkers=None, adaptive="order"):
    ''' 
    Parallel extrapolation with midpoint implicit method
    
    '''
    
    method = _midpoint_implicit
    
    k=p//2
    
    addSolverParam = _getAdditionalSolverParameters(len(y0), atol, rtol, addWork=True)

    if full_output:
        (ys, infodict) = __extrapolation_parallel(method, {}, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
             k=k, nworkers=nworkers, smoothing=smoothing, symmetric=True, seq=seq, adaptive=adaptive,
             addSolverParam=addSolverParam)
        infodict['p_avg'] = 2*infodict.pop('k_avg')
        return (ys, infodict)
    else:
        return __extrapolation_parallel(method, {}, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
             k=k, nworkers=nworkers, smoothing=smoothing, symmetric=True, seq=seq, adaptive=adaptive,
             addSolverParam=addSolverParam)


def ex_midpoint_semi_implicit_parallel(func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, robustness=2, smoothing = 'semiimp', seq=(lambda t: 2*(2*t-1)), p=4,
        nworkers=None, adaptive="order"):
    ''' 
    Parallel extrapolation with midpoint semi-implicit method
    
    '''
    
    method = _midpoint_semiimplicit
    
    k=p//2
    
    addSolverParam = _getAdditionalSolverParameters(len(y0), atol, rtol, addWork=True)
    
    methodargs = {}
    methodargs["J00"] = None
    
    methodargs["I"] = np.identity(len(y0), dtype=float)
    methodargs["Isparse"] = np.identity(len(y0), dtype=float)  

    if full_output:
        (ys, infodict) = __extrapolation_parallel(method, methodargs, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
             k=k, nworkers=nworkers, smoothing=smoothing, symmetric = True, seq=seq, adaptive=adaptive,
             addSolverParam=addSolverParam)
        infodict['p_avg'] = 2*infodict.pop('k_avg')
        return (ys, infodict)
    else:
        return __extrapolation_parallel(method, methodargs, func, grad, y0, t, args=args,
            full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
             k=k, nworkers=nworkers, smoothing=smoothing, symmetric = True, seq=seq, adaptive=adaptive,
             addSolverParam=addSolverParam)


def ex_euler_explicit_parallel(func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, robustness=2, smoothing = 'no', seq=(lambda t: t), p=4,
        nworkers=None, adaptive="order"):
    ''' 
    Parallel extrapolation with euler explicit method
    
    #TODO: this method hasn't been properly tested and optimized (sequence is not known as optimal)
    '''
    
    method = _euler_explicit
    
    addSolverParam = _getAdditionalSolverParameters(len(y0), atol, rtol, addWork=False)

    return __extrapolation_parallel(method, {}, func, grad, y0, t, args=args,
        full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
         k=p, nworkers=nworkers, smoothing=smoothing, symmetric = False, seq=seq, adaptive=adaptive,
         addSolverParam=addSolverParam)
    
    
def ex_euler_semi_implicit_parallel(func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, robustness=2, smoothing = 'no', seq=(lambda t: 2*(2*t-1)), p=4,
        nworkers=None, adaptive="order"):
    ''' 
    Parallel extrapolation with euler semi-implicit method
    
    '''
    
    method = _euler_semiimplicit
    
    addSolverParam = _getAdditionalSolverParameters(len(y0), atol, rtol, addWork=True)
    
    methodargs = {}
    methodargs["J00"] = None
    
    #The identity matrices are created here so that they can be reused (and are not 
    #recalculated), and also because global variables cause problems with multiprocessing
    #(processes would lock on the I or Isparse global variable)
    methodargs["I"] = np.identity(len(y0), dtype=float)
    methodargs["Isparse"] = np.identity(len(y0), dtype=float)  

    return __extrapolation_parallel(method, methodargs, func, grad, y0, t, args=args,
        full_output=full_output, rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, robustness_factor=robustness,
         k=p, nworkers=nworkers, smoothing=smoothing, symmetric = False, seq=seq, adaptive=adaptive,
         addSolverParam=addSolverParam)

'''
END BLOCK 2: General extrapolation solvers' functions. These functions can be used to solve any ODE.
These are the user interface functions.
'''
    
'''
USER INTERFACE Main Function: ODE solver based on extrapolation methods based on II.9 ref I (non stiff problems) and 
IV.9 ref II (stiff problems)

General function similar to BLOCK 2 functions, with less parameters, thus for a general user. To develop and test use
BLOCK 2 functions as those functions have more parameters to be chosen (if this functions is used such 
parameters are set to the 'optimal').

(method, func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, p=4)
        
    @param method (string): which method does the user want to use. Options:
        -'midpoint explicit': non-stiff solver
        -'midpoint implicit': stiff solver
        -'midpoint semi implicit': stiff solver
        -'euler explicit': non-stiff solver
        -other strings: 'euler semi implicit' stiff solver
    @param func (callable(y, t,args)): computes the derivative of y at t (i.e. the right hand side of the IVP).
    @param grad (callable(y,t,args)): computes analytically the Jacobian of the func function parameter.
    @param y0 (array): initial condition on y (can be a vector).
    @param t (array): a sequence of time points for which to solve for y. The initial value point should be the first 
            element of this sequence. And the last one the final time.
    @param args (tuple): extra arguments to pass to function.
    @param full_output (bool): true if user wants a dictionary of optional outputs as the second output.
    @param rtol, atol (float): the input parameters rtol (relative tolerance) and atol (absolute tolerance)
            determine the error control performed by the solver. See  function _error_norm(y1, y2, atol, rtol).
    @param h0 (float):the step size to be attempted on the first step.
    @param mxstep (int): maximum number of (internally defined) steps allowed for each
            integration point in t. Defaults to 10e4
    @param p (int): the order of extrapolation if order is fixed, or the starting order otherwise.
    @param nworkers (int): the number of workers working in parallel. If nworkers==None, then 
            the the number of workers is set to the number of CPUs on the running machine.
    @param adaptive (string): specifies the strategy of integration. Can take three values:
        - "fixed" = use fixed step size and order strategy.
        - "order" or any other string = use adaptive step size and adaptive order strategy (recommended).

    @return: 
        @return ys (2D-array, shape (len(t), len(y0))): array containing the value of y for each desired time in t, with 
            the initial value y0 in the first row.
        @return infodict (dict): only returned if full_output == True. Dictionary containing additional output information
             KEY        MEANING
            'fe_seq'    cumulative number of sequential derivative evaluations
            'nfe'       cumulative number of total derivative evaluations
            'nst'       cumulative number of successful time steps
            'nje'       cumulative number of either Jacobian evaluations (when 
                        analytic Jacobian is provided) or Jacobian estimations
                        (when no analytic Jacobian is provided)
            'h_avg'     average step size
            'p_avg'     average extrapolation order

'''

methods = {
        'explicit midpoint' : ex_midpoint_explicit_parallel,
        'implicit midpoint' : ex_midpoint_implicit_parallel,
        'semi-implicit midpoint' : ex_midpoint_semi_implicit_parallel,
        'explicit Euler' : ex_euler_explicit_parallel,
        'semi-implicit Euler' : ex_euler_semi_implicit_parallel
        }

def extrapolation_parallel(method_name, func, grad, y0, t, args=(), full_output=0, rtol=1.0e-8,
        atol=1.0e-8, h0=0.5, mxstep=10e4, p=4, nworkers=None, adaptive = 'order'):
        
        method_function = methods[method_name]
        return method_function(func, grad, y0, t, args=args, full_output=full_output,
                               rtol=rtol, atol=atol, h0=h0, mxstep=mxstep, p=p,
                               nworkers=nworkers, adaptive=adaptive)
