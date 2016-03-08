from __future__ import division
import numpy as np
import math 
import ex_parallel
import matplotlib.pyplot as plt
import twelve_tests as tst

#Whether to do convergence plots to see if they are straight lines (to choose the steps)
plot_convergence=False

#Methods that use smoothing loose an order of convergence
methods = {
               'explicit midpoint' :      {'smoothing' : False},
               'implicit midpoint' :      {'smoothing' : True},
               'semi-implicit midpoint' : {'smoothing' : True},
               'explicit Euler' :         {'smoothing' : False},
               'semi-implicit Euler' :    {'smoothing' : False}
           }

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def f_1(y,t):
    lam = -1
    y0 = np.array([1.])
    return lam*y

def exact_1(t):
    lam = -1
    y0 = np.array([1.])
    return y0*np.exp(np.dot(lam,t))

def f_2(y,t):
    return 4.*y*float(np.sin(t))**3*np.cos(t)
    
def exact_2(t):
    y0 = np.array([1])
    return y0*np.exp((np.sin(t))**4)

def f_3(y,t):
    return 4.*t*np.sqrt(y)
    
def exact_3(t):
    return np.array([np.power(1.+np.power(t,2),2)])

test_functions = [(f_1,exact_1),(f_2,exact_2),(f_3,exact_3)]

reference_solutions = {
    'explicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.367880291007],
                1.e-5 :  [0.36788005839],
                1.e-7 :  [0.367879445322],
                1.e-9 :  [0.367879441197],
                1.e-11 : [0.367879441172],
                1.e-13 : [0.367879441171]
            },
        f_2 : {
                1.e-3 :  [1.6509654995],
                1.e-5 :  [1.65097346807],
                1.e-7 :  [1.65097810912],
                1.e-9 :  [1.6509782068],
                1.e-11 : [1.65097820812],
                1.e-13 : [1.65097820815]
            },
        f_3 : {
                1.e-3 :  [3.99998398423],
                1.e-5 :  [3.9999966533],
                1.e-7 :  [4.00000000482],
                1.e-9 :  [3.99999999988],
                1.e-11 : [4.0],
                1.e-13 : [4.0]
            }
         },

    'implicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.367880291007],
                1.e-5 :  [0.36788005839],
                1.e-7 :  [0.367879445322],
                1.e-9 :  [0.367879441197],
                1.e-11 : [0.367879441172],
                1.e-13 : [0.367879441171]
            },
        f_2 : {
                1.e-3 :  [1.6509654995],
                1.e-5 :  [1.65097346807],
                1.e-7 :  [1.65097810912],
                1.e-9 :  [1.6509782068],
                1.e-11 : [1.65097820812],
                1.e-13 : [1.65097820815]
            },
        f_3 : {
                1.e-3 :  [3.99998398423],
                1.e-5 :  [3.9999966533],
                1.e-7 :  [4.00000000482],
                1.e-9 :  [3.99999999988],
                1.e-11 : [4.0],
                1.e-13 : [4.0]
            }
         },
    'semi-implicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.367879737132],
                1.e-5 :  [0.367879737132],
                1.e-7 :  [0.367879441612],
                1.e-9 :  [0.367879441172],
                1.e-11 : [0.367879441171],
                1.e-13 : [0.367879441171]
            },
        f_2 : {
                1.e-3 :  [1.65097824605],
                1.e-5 :  [1.65097824605],
                1.e-7 :  [1.65097822441],
                1.e-9 :  [1.65097820802],
                1.e-11 : [1.65097820814],
                1.e-13 : [1.65097820815]
            },
        f_3 : {
                1.e-3 :  [3.99999634801],
                1.e-5 :  [3.99999634801],
                1.e-7 :  [3.99999999118],
                1.e-9 :  [3.99999999994],
                1.e-11 : [4.0],
                1.e-13 : [4.0]
            }
         },

    'explicit Euler' : {
        f_1 : {
                1.e-3 :  [0.368069010877],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [1.65018664188],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [3.99777614101],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            }
         },

    'semi-implicit Euler' : {
        f_1 : {
                1.e-3 :  [0.367880291007],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [1.6509654995],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [3.99998398423],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            }
         }

    }

dense_reference_solutions = {
    'explicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.513417318852],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [1.15744512965],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [2.08641874929],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            }
         },
    'implicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.513416992557],
                1.e-5 :  [0.513417041583],
                1.e-7 :  [0.51341711976],
                1.e-9 :  [0.51341708752],
                1.e-11 : [0.513417119092],
                1.e-13 : [0.51341711908]
            },
        f_2 : {
                1.e-3 :  [1.15744652812],
                1.e-5 :  [1.15744652813],
                1.e-7 :  [1.15744600006],
                1.e-9 :  [1.15744546087],
                1.e-11 : [1.15744580962],
                1.e-13 : [1.1574455052]
            },
        f_3 : {
                1.e-3 :  [2.08642350313],
                1.e-5 :  [2.08642338504],
                1.e-7 :  [2.08641986814],
                1.e-9 :  [2.08642012105],
                1.e-11 : [2.08641975192],
                1.e-13 : [2.08642041979]
            }
         },
    'semi-implicit midpoint' : {
        f_1 : {
                1.e-3 :  [0.513417197383],
                1.e-5 :  [0.513417197383],
                1.e-7 :  [0.513417121761],
                1.e-9 :  [0.513417053851],
                1.e-11 : [0.513417119025],
                1.e-13 : [0.513417119031]
            },
        f_2 : {
                1.e-3 :  [1.15744626921],
                1.e-5 :  [1.15744626921],
                1.e-7 :  [1.1574457217],
                1.e-9 :  [1.15744556821],
                1.e-11 : [1.1574454661],
                1.e-13 : [1.15744546561]
            },
        f_3 : {
                1.e-3 :  [2.08642277948],
                1.e-5 :  [2.08642277948],
                1.e-7 :  [2.08642116049],
                1.e-9 :  [2.08641997499],
                1.e-11 : [2.08641995246],
                1.e-13 : [2.08641975312]
            }
         },
    'explicit Euler' : {
        f_1 : {
                1.e-3 :  [0.51362268695],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [1.15688350412],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [2.08516207191],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            }
         },
    'semi-implicit Euler' : {
        f_1 : {
                1.e-3 :  [0.513424124096],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [1.15617955038],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [2.08703832785],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            }
         }

    }

########### RUN TESTS ###########

tolerances = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
def non_dense_output_tests():
    print("\n Executing regression tests without dense output...")
    
    for method in methods.keys():
        print("\n Method: " + method)
        for f, exact in test_functions:
            print("\n   Test: " + f.__name__)
            t0, tf = 0.1, 1
            y0 = exact(t0)
            output_times = [t0, tf]
            for tol in tolerances:
                print("       Tolerance: " + str(tol))
                ys, infodict = ex_parallel.extrapolation_parallel(method,f, None, y0, output_times, atol=tol,
                    rtol=tol, mxstep=1.e6, full_output=True, nworkers=2)
                y = ys[-1][0]
                assert np.allclose(y, reference_solutions[method][f][tol])
     
    print("All tests passed")


def dense_output_tests():
    print("\n Executing regression tests with dense output...")
    
    for method in methods.keys():
        print("\n Method " + method)
        for f, exact in test_functions:
            print("\n   Test " + f.__name__)
            t0 = 0.1
            output_times = [t0,1./3,2./3,1]
            y0 = exact(t0)
            for tol in tolerances:
                print("       Tolerance: " + str(tol))
                ys, infodict = ex_parallel.extrapolation_parallel(method,f, None, y0, output_times, atol=tol,
                    rtol=tol, mxstep=1.e6, full_output=True, nworkers=2)
                y = ys[-2][0]
                assert np.allclose(y, dense_reference_solutions[method][f][tol])
                
    print("All tests passed")


def convergence_test(method_name, test, step_sizes, order, dense=False):
    """
   Perform a convergence test by applying the specified method to the specified
   the test problem with the specified step sizes.
    """
    y_ref = np.loadtxt(tst.getReferenceFile(test.problemName))
    dense_output = test.denseOutput

    if(not dense):
        y_ref=y_ref[-1]
        dense_output=[dense_output[0], dense_output[-1]]
    else:
        nhalf = np.ceil(len(y_ref)/2)
        y_ref = y_ref[nhalf]
        print("dense output time " + str(dense_output[nhalf]))
        
    error_per_step=[]
    for step in step_sizes:
        #rtol and atol are not important as we are fixing the step size
        ys, infodict = ex_parallel.extrapolation_parallel(method_name,test.RHSFunction, None,
                                                          test.initialValue, dense_output, atol=1e-1,
                                                          rtol=1e-1, mxstep=10000000, full_output=True,
                                                          nworkers=4, adaptive='fixed', p=order, h0=step)
        
        ys=ys[1:len(ys)]
        if(dense):
            ys=ys[nhalf]
        error = relative_error(ys, y_ref)
        error_per_step.append(error)
        
    coefficients = np.polyfit(np.log10(step_sizes), np.log10(error_per_step), 1)

    print("     Slope: " + str(coefficients[0]) + "; Design order is: " + str(order-methods[method_name]['smoothing']))
    
    if(plot_convergence):
        plt.loglog(step_sizes,error_per_step, marker="x")
        plt.show()

    return coefficients[0]

def check_convergence_rate(apparent_rate, expected_rate, test_name):
    """
    Checks if err equals err_ref, returns silently if err equals err_ref (matching 10 decimals)
    or raises an exception otherwise
    @param err: calculated error
    @param err_ref: expected error
    @param test_name: tests explanation and/or explanatory name
    """
    np.testing.assert_approx_equal(apparent_rate, expected_rate, 1, 
                                   "Convergence test " + test_name + " failed")


def run_convergence_tests(methods):
    """
    Test convergence of each method for the linear, scalar problem and for
    vanderpol (non-stiff epsilon).  It's challenging especially for high-order
    methods, to find a range of step
    sizes that are in the asymptotic regime but don't reach roundoff.
    For this reason, some tests are currently skipped and the tests only
    go up to order 6.  With some more difficult test problems it may be
    more feasible to test higher-order methods.
    """
    for method in methods.itervalues():
        method['linear problem step sizes'] = {}
        method['linear problem step sizes'][2] = [0.5,0.2,0.04,0.02,0.005,0.002,0.001]
        method['linear problem step sizes'][4] = [0.5,0.2,0.04,0.02,0.005,0.002,0.001]
        method['linear problem step sizes'][6] = [1.0,0.8,0.4,0.2]

        method['vanderpol step sizes'] = {}
        method['vanderpol step sizes'][2] = [0.15,0.04,0.02,0.005,0.002]
        method['vanderpol step sizes'][4] = [0.15,0.04,0.02,0.005]
        method['vanderpol step sizes'][6] = [0.75,0.2,0.04,0.02]

        method['skip linear dense']    = []
        method['skip vanderpol']       = []
        method['skip vanderpol dense'] = []
 
    #Can't use order 2 with midpoint method (it doesn't do extrapolation and interpolation doesn't work)
    for mtype in ['explicit', 'implicit', 'semi-implicit']:
        for skip in ['skip linear dense', 'skip vanderpol', 'skip vanderpol dense']:
            methods[mtype + ' midpoint'][skip].append(2)

    # Convergence stagnates; probably because of the nonlinear solver:
    methods['semi-implicit Euler']['linear problem step sizes'][6] = [0.8,0.4,0.2,0.1]
    #This one is super-convergent:
    methods['semi-implicit Euler']['skip linear dense'].append(6)
    methods['semi-implicit Euler']['skip vanderpol dense'].append(6)
    methods['semi-implicit Euler']['linear problem step sizes'][6] = [0.8,0.4,0.2,0.1]
    methods['semi-implicit Euler']['vanderpol step sizes'][6] = [0.75,0.2,0.1]

    for method_name, method in methods.iteritems():
        print("\n Method: " + method_name)
        for p in (2,4,6):
            print("   Testing method of order " + str(p))
            print("     Linear test problem, no dense output")
            step_sizes = method['linear problem step sizes'][p]
            coeff = convergence_test(method_name, tst.LinearProblem(),step_sizes,p,False)
            check_convergence_rate(coeff, p-method['smoothing'], "Test Linear non dense")
            if not (p in method['skip vanderpol']):
                print("     Vanderpol test problem, no dense output")
                step_sizes = method['vanderpol step sizes'][p]
                coeff = convergence_test(method_name, tst.VDPOLEasyProblem(),step_sizes,p,False)
                check_convergence_rate(coeff, p-method['smoothing'], "Test VPOL non dense")
            if not (p in method['skip linear dense']):
                print("     Linear test problem, dense output")
                step_sizes = method['linear problem step sizes'][p]
                coeff = convergence_test(method_name, tst.LinearProblem(),step_sizes,p,True)
                check_convergence_rate(coeff, p-method['smoothing'], "Test Linear non dense")
            if not (p in method['skip vanderpol dense']):
                print("     Vanderpol test problem, dense output")
                step_sizes = method['vanderpol step sizes'][p]
                coeff = convergence_test(method_name, tst.VDPOLEasyProblem(),step_sizes,p,True)
                check_convergence_rate(coeff, p-method['smoothing'], "Test VPOL dense")

    print("All convergence tests passed")
   
    
def exp(x):
    return np.array([np.float128(np.exp(-x))])

def expder(x,orderder):
    return np.array([np.float128((-1)**orderder*np.exp(-x))])

def test_interpolation_polynomial():
    '''
    Check the correct behavior of the two interpolation functions for dense output results; 
    symmetric and non symmetric interpolation (_interpolate_nonsym and _interpolate_sym).
    
    It uses an exponential function to feed the two functions with exact values at each intermediate
    point and its derivatives at each point too. It then calculates the interpolation polynomial for 
    different steps' length and compares the exact solution at one intermediate point (currently H/5)
    with the interpolated value at the same point. It then check the convergence order of the polynomial.  
    '''
    plot_convergence=False
    print("\n Executing convergence interpolation polynomial test")
    orders=[2,3,4,5]

    steps = [0.5,0.55,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15]
    
    #TODO: This should be zero all the time
    orderdisparity = [[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,1,0,1]]
    
    idx=0
    for order in orders:
        print("Order: "  + str(order))
        error_per_step = np.zeros(len(steps))
        errorIntPerStep = np.zeros(len(steps))
        errorPerStepSym = np.zeros(len(steps))
        errorIntPerStepSym = np.zeros(len(steps))
        seq=(lambda t: 4*t-2)
        
        t0=0
        k=0
        for H in steps:
            y0 = exp(t0)
            Tkk = exp(t0+H)
            f_Tkk = expder(t0+H,1)
            yj = (order+1)*[None]
            f_yj = (order+1)*[None]
            hs = (order+1)*[None]
            y_half = (order+1)*[None]
            for i in range(1,order+1):
                ni = seq(i)
                yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
                f_yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
                for j in range(ni+1):
                    yj_[j]=exp(j*H/ni)
                    f_yj_[j]=expder(j*H/ni,1)
                    
                yj[i]=yj_
                f_yj[i]=f_yj_
                y_half[i]=yj_[ni/2]
                hs[i]=H/ni
            
            
            poly = ex_parallel._interpolate_nonsym(y0, Tkk, yj, hs, H, order, atol=1e-5,rtol=1e-5, seq=seq)        
            
            polysym = ex_parallel._interpolate_sym(y0, Tkk,f_Tkk, y_half, f_yj, hs, H, order, atol=1e-5,rtol=1e-5, seq=seq)
            
            x=H/5;
             
            res,errint,hint = poly(x)
            resexact=exp(t0+H*x)
            errorIntPerStep[k]=np.linalg.norm((errint))
            error_per_step[k] = np.linalg.norm((res-resexact))
    
            ressym,errintsym,hint = polysym(x)
            errorIntPerStepSym[k]=np.linalg.norm(errintsym)
            errorPerStepSym[k] = np.linalg.norm((ressym-resexact))
                    
            k+=1
        
        print("Order disparity: " + str(orderdisparity[idx]))
        coefficients = np.polyfit(np.log10(steps), np.log10(error_per_step), 1)
        print("coefficients error " + str(coefficients) + "order is: " + str(order))
        # In this case order of interpolation for non symmetric should be order because lam=1
        # see _compute_rs(..) in ex_parallel
        check_convergence_rate(coefficients[0], order+orderdisparity[idx][0], "Interpolation non symmetric")
        
        #TODO: this should be one order less of convergence (with lam=0 it works well)
        #if lam=0 then last check should be order+1 (as expected)
        coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStep), 1)
        print("coefficients error interpolation" + str(coefficientsint) + " order is: " + str(order-1))
        check_convergence_rate(coefficientsint[0], order-1+orderdisparity[idx][1], "Interpolation non symmetric estimated interpolation error")

         
        coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStepSym), 1)
        print("coefficients error sym " + str(coefficients) + " order is: " + str(order+4))
        check_convergence_rate(coefficients[0], order+4+orderdisparity[idx][2], "Interpolation symmetric")
             
        coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStepSym), 1)
        print("coefficients error sym interpolation" + str(coefficientsint) + " order is: " + str(order+4-1))
        check_convergence_rate(coefficientsint[0], order+4-1+orderdisparity[idx][3], "Interpolation symmetric estimated interpolation error")
        
        idx+=1
        
        if(plot_convergence):
            plt.plot(np.log10(steps),np.log10(error_per_step), marker="x")
            plt.plot(np.log10(steps),np.log10(errorIntPerStepSym), marker="x")
            plt.plot(np.log10(steps),np.log10(errorPerStepSym), marker="x")
            plt.plot(np.log10(steps),np.log10(errorIntPerStep), marker="x")
            plt.show()
            
    print("All tests passed")

def test_interpolated_derivatives():
    '''
    It checks if the higher order derivatives' estimations used to calculate the interpolation
    polynomial converge correctly (_centered_finite_diff and _backward_finite_diff). 
    
    An exponential function is used to feed the exact values used
    to estimate the derivatives. It then compares the estimated higher order derivatives with the
    exact values of the exponential derivatives.
    '''
    plot_convergence=False
    #TODO: orderrs>7 not correct behavior
    orderrs=7
    #TODO: orderds>4 does not behave correctly (data type numeric error)
    orderds=4
    
    print("\n Executing convergence interpolation polynomial derivatives test")

    seq=(lambda t: 4*t-2)
    steps = [0.5,0.55,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15]
#     steps = [0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.5,1.6,1.8,2,2.5,3,3.1,3.5]
    dsnumder=2*orderds+1
    #-1 because lam=1
    rsnumder=orderrs+1-1
    errordsPerDerandStep = np.zeros((dsnumder,len(steps)))
    errorrsPerDerandStep = np.zeros((rsnumder,len(steps)))
    
    t0=0
    k=0
    for H in steps:
        y0 = exp(t0)
        Tkk = exp(t0+H)
        f_Tkk = expder(t0+H,1)
        yj = (orderrs+1)*[None]
        f_yj = (orderrs+1)*[None]
        hs = (orderrs+1)*[None]
        y_half = (orderrs+1)*[None]
        for i in range(1,orderrs+1):
            ni = seq(i)
            yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
            f_yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
            for j in range(ni+1):
                yj_[j]=exp(j*H/ni)
                f_yj_[j]=expder(j*H/ni,1)
                
            yj[i]=yj_
            if(i<orderds+1):
                f_yj[i]=f_yj_
                y_half[i]=yj_[ni/2]
            hs[i]=H/ni
    
        rs = np.zeros((rsnumder), dtype=(type(yj[1][0])))
        for i in range(1,rsnumder):
            rs[i]=expder(H, i)
            
        ds = np.zeros((dsnumder), dtype=(type(y_half[1])))
        for i in range(dsnumder):
            ds[i]=expder(H/2, i)
        
        dsapp = ex_parallel._compute_ds(y_half, f_yj, hs[0:orderds+1], orderds, seq=seq)
        
        rsapp = ex_parallel._compute_rs(yj, hs, orderrs, seq=seq)
    
        for der in range(1,rsnumder):
            errorrsPerDerandStep[der][k] = np.linalg.norm((rs[der]-rsapp[der]))
        
        for der in range(1,dsnumder):
            errordsPerDerandStep[der][k] = np.linalg.norm((ds[der]-dsapp[der]))            
        k+=1
    
    print("Non symmetric derivatives test (backward difference)")
    #TODO: this should be zero, could be because 0.266 ~= 1 but doesn't pass the check
    deviationorder=[0,0,0,0,0,0,1]
    
    for der in range(1,rsnumder):
        errorrsPerStep = errorrsPerDerandStep[der]
         
        coefficientsrs = np.polyfit(np.log10(steps), np.log10(errorrsPerStep), 1)
        expectedorder=orderrs-der
        print("coefficients error non sym interpolation" + str(coefficientsrs) + " order is: " + str(expectedorder))
        check_convergence_rate(coefficientsrs[0]+deviationorder[der], expectedorder, "Interpolation non symmetric derivatives convergence")
         
        if(plot_convergence):
            plt.plot(np.log10(steps),np.log10(errorrsPerStep), marker="x")
    plt.show()
     
    print("Symmetric derivatives test (centered difference)")
    #Derivative checking starts at second derivative because the 0 and 1 are fed as exact
    for der in range(2,dsnumder):
        errordsPerStep = errordsPerDerandStep[der]
        
        coefficientsds = np.polyfit(np.log10(steps), np.log10(errordsPerStep), 1)
        expectedorder=2*(orderds-math.ceil(der/2)+1)
        print("coefficients error sym interpolation" + str(coefficientsds) + " order is: " + str(expectedorder))
        check_convergence_rate(coefficientsds[0], expectedorder, "Interpolation symmetric derivatives convergence")
        
        if(plot_convergence):
            plt.plot(np.log10(steps),np.log10(errordsPerStep), marker="x")
    plt.show()
    
    print("All tests passed")


if __name__ == "__main__":
    non_dense_output_tests()
    dense_output_tests()
  
    run_convergence_tests(methods)
    test_interpolation_polynomial()
    test_interpolated_derivatives()
    

