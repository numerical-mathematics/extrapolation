from __future__ import division
import numpy as np
import math 
import ex_parallel
import matplotlib.pyplot as plt
import twelve_tests as tst

#Whether to do convergence plots to see if they are straight lines (to choose the steps)
plotConv=False

allmethods = ['explicit midpoint','implicit midpoint','semi-implicit midpoint','explicit Euler','semi-implicit Euler']
#Methods that use smoothing loose an order of convergence
methodsmoothing = [0,1,1,0,0]
regression_methods = ['explicit midpoint','implicit midpoint']

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

#TODO: this fourth test function gives singular matrix with semi implicit methods
#mainly because y has to be >0 and when the method gives a y<0 then the function evaluation
#at that estimated value is nan and it blows all the execution afterwards
def f_4(y,t):
    return y/t*np.log(y)
    
def exact_4(t):
    return np.array([np.exp(2.*t)])


alltestfunctions = [(f_1,exact_1),(f_2,exact_2),(f_3,exact_3)]

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
         }
    }

dense_reference_solutions = {
    'explicit midpoint' : {
        f_1 : {
                1.e-3 :  [],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_2 : {
                1.e-3 :  [],
                1.e-5 :  [],
                1.e-7 :  [],
                1.e-9 :  [],
                1.e-11 : [],
                1.e-13 : []
            },
        f_3 : {
                1.e-3 :  [],
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
         }
    }

########### RUN TESTS ###########

tolerances = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
def non_dense_output_tests():
    print("\n Executing regression tests without dense output...")
    
    for method in regression_methods:
        print("\n Method: " + method)
        for f, exact in alltestfunctions:
            print("\n   Test: " + f.__name__)
            t0, tf = 0.1, 1
            y0 = exact(t0)
            output_times = [t0, tf]
            for tol in tolerances:
                print("\n       Tolerance: " + str(tol))
                ys, infodict = ex_parallel.extrapolation_parallel(method,f, None, y0, output_times, atol=tol,
                    rtol=tol, mxstep=1.e6, full_output=True, nworkers=2)
                y = ys[-1][0]
                assert np.allclose(y, reference_solutions[method][f][tol])
     
    print("All tests passed")


def dense_output_tests():
    print("\n Executing regression tests with dense output...")
    
    for method in regression_methods:
        print("\n Method " + method)
        for f, exact in alltestfunctions:
            print("\n   Test " + f.__name__)
            t0 = 0.1
            output_times = [t0,1./3,2./3,1]
            y0 = exact(t0)
            for tol in tolerances:
                print("\n       Tolerance: " + str(tol))
                ys, infodict = ex_parallel.extrapolation_parallel(method,f, None, y0, output_times, atol=tol,
                    rtol=tol, mxstep=1.e6, full_output=True, nworkers=2)
                y = ys[-2][0]
                assert np.allclose(y, dense_reference_solutions[method][f][tol])
                
    print("All tests passed")


def convergenceTest(method, i, test, allSteps, order, dense=False):
    '''''
       Perform a convergence test with the test problem (in test parameter) with
       the given steps in parameter allSteps. 
    '''''
    
    y_ref = np.loadtxt(tst.getReferenceFile(test.problemName))
    denseOutput = test.denseOutput

    if(not dense):
        y_ref=y_ref[-1]
        denseOutput=[denseOutput[0], denseOutput[-1]]
    else:
        nhalf = np.ceil(len(y_ref)/2)
        y_ref = y_ref[nhalf]
        print("dense output time " + str(denseOutput[nhalf]))
        
    k=0
    errorPerStep=[]
    for step in allSteps:
        #rtol and atol are not important as we are fixing the step size
        ys, infodict = ex_parallel.extrapolation_parallel(method,test.RHSFunction, None, test.initialValue, denseOutput, atol=1e-1, 
            rtol=1e-1, mxstep=10000000, full_output=True, nworkers=4, adaptative='fixed', p=order, h0=step)        
#         print("number steps: " + str(infodict['nst']) + " (should be " + str(denseOutput[-1]/step) + ")")
        
        ys=ys[1:len(ys)]
        if(dense):
            ys=ys[nhalf]
        error = relative_error(ys, y_ref)
        errorPerStep.append(error)
        
    coefficients = np.polyfit(np.log10(allSteps), np.log10(errorPerStep), 1)

    print("coefficients: " + str(coefficients) + " order is: " + str(order-methodsmoothing[i]))
    
    if(plotConv):
        plt.plot(np.log10(allSteps),np.log10(errorPerStep), marker="x")
        plt.show()

    return coefficients[0]

def checkConvergenceCoeff(coeff, coeff_ref, test_name):
    """
    Checks if err equals err_ref, returns silently if err equals err_ref (matching 10 decimals)
    or raises an exception otherwise
    @param err: calculated error
    @param err_ref: expected error
    @param test_name: tests explanation and/or explanatory name
    """
    np.testing.assert_approx_equal(coeff, coeff_ref, 1, "Convergence test " + test_name + " failed")


def doAllConvergenceTests():
    global plotConv
    plotConv=False
 
    #linear: 2
    linearSteps2 = np.concatenate((np.linspace(0.5,0.2,4), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.002,10),np.linspace(0.0019,0.001,10)))
    
    #linear: 4
    linearSteps4 = np.concatenate((np.linspace(0.5,0.2,4), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.0035,5)))
    
    #linear: 6 
    linearSteps6 = np.concatenate((np.linspace(0.7,0.2,3), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.015,4)))
    
    #vdpol: 2,4
    vdpolSteps2 = np.concatenate((np.linspace(0.15,0.04,5),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.002,10)))
    
    #vdpol: 6 
    vdpolSteps6 = np.concatenate((np.linspace(0.75,0.2,7), np.linspace(0.19,0.04,7),
                                np.linspace(0.039,0.02,7)))
    #vdpol: 8
    vdpolSteps8 = np.concatenate((np.linspace(1.1,0.75,5),np.linspace(0.73,0.2,8), np.linspace(0.19,0.055,8)))
    
    #linear: 6 exception 1
    linearSteps6ex1 = [1,1/2,1/3,1/4,1/5]
    
    #linear: 6 exception 2
    linearSteps6ex2 = np.concatenate((np.linspace(0.7,0.2,3), np.linspace(0.19,0.04,7),np.linspace(0.039,0.025,4)))
    
    #linear: 6 exception 3
    linearSteps6ex3 = [1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10,1/11,1/12,1/13]
    
    #vdpol: 8 exception 1
    vdpolSteps8ex1 = np.concatenate((np.linspace(0.73,0.2,8), np.linspace(0.19,0.07,8)))
    
    #vdpol: 4 exception 1
    vdpolSteps4ex1 = np.concatenate((np.linspace(0.65,0.2,6), np.linspace(0.19,0.04,7),
                                np.linspace(0.039,0.025,4)))

    #vdpol: 6 exception 1
    vdpolSteps6ex1 = np.concatenate((np.linspace(0.65,0.2,6), np.linspace(0.19,0.055,7)))
    
    #This is needed because some methods converge faster than the others and some steps have to be personalized
    methodslinearstepexception = [[None,None,None],[None,None,linearSteps6ex1],[None,None,None],[None,None,linearSteps6ex2],[None,None,linearSteps6ex3]] 
    
    methodslinearskip = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
    
    methodslineardensestepexception = [[None,None,None],[None,None,linearSteps6ex1],[None,None,None],[None,None,linearSteps6ex2],[None,None,None]] 
    #Can't use order 2 with midpoint method (it doesn't do extrapolation and interpolation doesn't work)
    methodslineardenseskip = [['skip',' ','skip'],['skip',' ','skip'],['skip',' ','skip'],[' ',' ',' '],[' ',' ','skip']] 


    methodsvdpolstepexception = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,vdpolSteps4ex1,vdpolSteps6ex1,None]]
    
    methodsvdpolskip = [[' ',' ',' ',' '],[' ',' ',' ','skip'],[' ',' ',' ','skip'],[' ',' ',' ','skip'],[' ',' ',' ','skip']] 

    methodsvdpoldensestepexception = [[None,vdpolSteps4ex1,None,None],[None,None,None,None],[None,None,None,None],[None,vdpolSteps4ex1,vdpolSteps6ex1,None],[None,vdpolSteps4ex1,vdpolSteps6ex1,None]]

    methodsvdpoldenseskip = [['skip',' ',' ','skip'],['skip',' ',' ','skip'],['skip',' ',' ','skip'],[' ',' ','skip','skip'],[' ',' ',' ','skip']] 

    
    allorderslinear = [2,4,6]
    alllinearsteps = [linearSteps2,linearSteps4,linearSteps6]
    
    allordersvdpol = [2,4,6,8]
    allvdpolsteps = [vdpolSteps2,vdpolSteps2,vdpolSteps6,vdpolSteps8]

    print("\n Executing convergence tests without dense output")
    
    i=0
    for method in allmethods:
        print("\n Method: " + method)
        print("\n Test: Linear Function")
        k=0
        for p in allorderslinear:
            if(methodslinearskip[i][k]!='skip'):
                if(methodslinearstepexception[i][k] is not None):
                    linearsteps=methodslinearstepexception[i][k]
                else:
                    linearsteps=alllinearsteps[k]
                coeff = convergenceTest(method,i, tst.LinearProblem(),linearsteps,p,False)
                checkConvergenceCoeff(coeff, p-methodsmoothing[i], "Test Linear non dense")
            k+=1
            
        print("\n Test: VDPOL Easy (high epsilon) Function")
        k=0
        for p in allordersvdpol:
            if(methodsvdpolskip[i][k]!='skip'):
                if(methodsvdpolstepexception[i][k] is not None):
                    vdpolsteps=methodsvdpolstepexception[i][k]
                else:
                    vdpolsteps=allvdpolsteps[k]
                coeff = convergenceTest(method,i, tst.VDPOLEasyProblem(),vdpolsteps,p,False)
                checkConvergenceCoeff(coeff, p-methodsmoothing[i], "Test VPOL non dense")
            k+=1
        i+=1
          
    print("All tests passed")
    
    print("\n Executing convergence tests with dense output")
    
    i=0
    for method in allmethods:
        print("\n Method: " + method)
        print("\n Test: Linear Function")
        k=0
        for p in allorderslinear:
            if(methodslineardenseskip[i][k]!='skip'):
                if(methodslineardensestepexception[i][k] is not None):
                    linearsteps=methodslineardensestepexception[i][k]
                else:
                    linearsteps=alllinearsteps[k]
                coeff = convergenceTest(method,i, tst.LinearProblem(),linearsteps,p,True)
                checkConvergenceCoeff(coeff, p-methodsmoothing[i], "Test Linear non dense")
            k+=1
          
        print("\n Test: VDPOL Easy (high epsilon) Function")
        k=0
        for p in allordersvdpol:
            if(methodsvdpoldenseskip[i][k]!='skip'):
                if(methodsvdpoldensestepexception[i][k] is not None):
                    vdpolsteps=methodsvdpoldensestepexception[i][k]
                else:
                    vdpolsteps=allvdpolsteps[k]
                coeff = convergenceTest(method,i, tst.VDPOLEasyProblem(),vdpolsteps,p,True)
                checkConvergenceCoeff(coeff, p-methodsmoothing[i], "Test VPOL non dense")
            k+=1
        i+=1
 
    print("All tests passed")
   
    
def exp(x):
    return np.array([np.float128(np.exp(-x))])

def expder(x,orderder):
    return np.array([np.float128((-1)**orderder*np.exp(-x))])

def checkInterpolationPolynomial():
    plotConv=False
    print("\n Executing convergence interpolation polynomial test")
    orders=[2,3,4,5]

    steps = [0.5,0.55,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15]
    
    #TODO: This should be zero all the time
    orderdisparity = [[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,-1,1]]
    
    idx=0
    for order in orders:
        print("Order: "  + str(order))
        errorPerStep = np.zeros(len(steps))
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
            errorPerStep[k] = np.linalg.norm((res-resexact))
    
            ressym,errintsym,hint = polysym(x)
            errorIntPerStepSym[k]=np.linalg.norm(errintsym)
            errorPerStepSym[k] = np.linalg.norm((ressym-resexact))
                    
            k+=1
        
        print("Order disparity: " + str(orderdisparity[idx]))
        coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStep), 1)
        print("coefficients error " + str(coefficients) + "order is: " + str(order))
        # In this case order of interpolation for non symmetric should be order because lam=1
        # see _compute_rs(..) in ex_parallel
        checkConvergenceCoeff(coefficients[0], order+orderdisparity[idx][0], "Interpolation non symmetric")
        
        #TODO: this should be one order less of convergence (with lam=0 it works well)
        #if lam=0 then last check should be order+1 (as expected)
        coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStep), 1)
        print("coefficients error interpolation" + str(coefficientsint) + " order is: " + str(order-1))
        checkConvergenceCoeff(coefficientsint[0], order-1+orderdisparity[idx][1], "Interpolation non symmetric estimated interpolation error")

         
        coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStepSym), 1)
        print("coefficients error sym " + str(coefficients) + " order is: " + str(order+4))
        checkConvergenceCoeff(coefficients[0], order+4+orderdisparity[idx][2], "Interpolation symmetric")
             
        coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStepSym), 1)
        print("coefficients error sym interpolation" + str(coefficientsint) + " order is: " + str(order+4-1))
        checkConvergenceCoeff(coefficientsint[0], order+4-1+orderdisparity[idx][3], "Interpolation symmetric estimated interpolation error")
        
        idx+=1
        
        if(plotConv):
            plt.plot(np.log10(steps),np.log10(errorPerStep), marker="x")
            plt.plot(np.log10(steps),np.log10(errorIntPerStepSym), marker="x")
            plt.plot(np.log10(steps),np.log10(errorPerStepSym), marker="x")
            plt.plot(np.log10(steps),np.log10(errorIntPerStep), marker="x")
            plt.show()
            
    print("All tests passed")

def checkDerivativesForPolynomial():
    plotConv=False
    #TODO: orderrs>7 not correct behaviour
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
        checkConvergenceCoeff(coefficientsrs[0]+deviationorder[der], expectedorder, "Interpolation non symmetric derivatives convergence")
         
        if(plotConv):
            plt.plot(np.log10(steps),np.log10(errorrsPerStep), marker="x")
    plt.show()
     
    print("Symmetric derivatives test (centered difference)")
    #Derivative checking starts at second derivative because the 0 and 1 are fed as exact
    for der in range(2,dsnumder):
        errordsPerStep = errordsPerDerandStep[der]
        
        coefficientsds = np.polyfit(np.log10(steps), np.log10(errordsPerStep), 1)
        expectedorder=2*(orderds-math.ceil(der/2)+1)
        print("coefficients error sym interpolation" + str(coefficientsds) + " order is: " + str(expectedorder))
        checkConvergenceCoeff(coefficientsds[0], expectedorder, "Interpolation symmetric derivatives convergence")
        
        if(plotConv):
            plt.plot(np.log10(steps),np.log10(errordsPerStep), marker="x")
    plt.show()
    
    print("All tests passed")


if __name__ == "__main__":
    non_dense_output_tests()
    dense_output_tests()
  
    doAllConvergenceTests()
    checkInterpolationPolynomial()
    checkDerivativesForPolynomial()
    

