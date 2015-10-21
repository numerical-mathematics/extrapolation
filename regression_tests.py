from __future__ import division
import numpy as np
import math 
import time
import ex_parallel
import fnbod
from compare_test import kdv_init, kdv_func, kdv_solout, burgers_init, burgers_func, burgers_solout
import twelve_tests as tst
import matplotlib.pyplot as plt

plotConv=False

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def regression_tst(func, y0, t, y_ref, tol_boundary=(0,6), h0=0.5, mxstep=10e4,
        adaptive="order", p=4, solout=(lambda t: t), nworkers=2):
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]

    err = np.zeros(len(tol))
    print ''
    for i in range(len(tol)):
        print tol[i]
        ys, infodict = ex_parallel.ex_midpoint_explicit_parallel(func, y0, t, atol=tol[i], 
            rtol=tol[i], mxstep=mxstep, full_output=True, nworkers=nworkers)
        y = solout(ys[1:len(ys)])
        err[i] = relative_error(y, y_ref)
        print infodict
    return err

def f_1(y,t):
    lam = -1j
    y0 = np.array([1 + 0j])
    return lam*y

def exact_1(t):
    lam = -1j
    y0 = np.array([1 + 0j])
    return y0*np.exp(np.dot(lam,t))

def f_2(y,t):
    return 4.*y*float(np.sin(t))**3*np.cos(t)
    
def exact_2(t):
    y0 = np.array([1])
    return y0*np.exp((np.sin(t))**4)

def f_3(y,t):
    return 4.*t*np.sqrt(y)
    
def exact_3(t):
    return np.array([(1.+t**2)**2])

def f_4(y,t):
    return y/t*np.log(y)
    
def exact_4(t):
    return np.array([np.exp(2.*t)])

def f_5(y,t):
    return fnbod.fnbod(y,t)

def check(err, err_ref, test_name):
    """
    Checks if err equals err_ref, returns silently if err equals err_ref (matching 10 decimals)
    or raises an exception otherwise
    @param err: calculated error
    @param err_ref: expected error
    @param test_name: tests explanation and/or explanatory name
    """
    np.testing.assert_array_almost_equal(err, err_ref, 10, "REGRESSION TEST " + test_name + " FAILED")
    
    
########### RUN TESTS ###########

def non_dense_tests():
    print("\n Executing non dense tests")
    # Test 1
    t0, tf = 0, 10
    y0 = exact_1(t0)
    y_ref = exact_1(tf)
    err = regression_tst(f_1, y0, [t0, tf], y_ref)
    err_ref_1 = np.array([  2.6716177733e-03,   2.0211717955e-05,   1.4248356310e-07,   1.1229200827e-09,   9.4184968610e-12,   7.6482844347e-14]) 
    check(err, err_ref_1, "Test 1: linear ODE")
    print err
  
    # Test 2
    t0, tf = 0, 10
    y0 = exact_2(t0)
    y_ref = exact_2(tf)
    err = regression_tst(f_2, y0, [t0, tf], y_ref)
    err_ref_2 = np.array([8.54505911e-02, 2.48291275e-04, 8.17836269e-07, 2.85969698e-09, 2.54201200e-11, 9.10317032e-13])
    check(err, err_ref_2, "Test 2")
    print err
    
    # Test 3
    t0, tf = 0, 10
    y0 = exact_3(t0)
    y_ref = exact_3(tf)
    err = regression_tst(f_3, y0, [t0, tf], y_ref)
    err_ref_3 = np.array([1.93968837e-05, 9.48240508e-08, 1.21300220e-09, 2.83645194e-10, 1.74748516e-13, 5.88438882e-15]) 
    check(err, err_ref_3, "Test 3")
    print err
    
    # Test 4
    t0, tf = 0.5, 10
    y0 = exact_4(t0)
    y_ref = exact_4(tf)
    err = regression_tst(f_4, y0, [t0, tf], y_ref)
    err_ref_4 = np.array([6.77863684e-03, 9.50343703e-05, 4.90535263e-07, 3.36128205e-09, 2.47615358e-11, 2.92147596e-13]) 
    check(err, err_ref_4, "Test 4")
    print err
      
    # Test 5
    t0, tf = 0, 0.08
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference.txt")
    err = regression_tst(f_5, y0, [t0, tf], y_ref)
    err_ref_5 = np.array([  5.8878680341e-01,   7.8285273702e-02,   7.7063734719e-03,     1.6273023075e-05,   6.5385240310e-07,   1.1777903740e-08]) 
    check(err, err_ref_5, "Test 5")
    print err
    
    # Test 6
    t0, tf = 0, 0.003
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv.txt")
    err = regression_tst(kdv_func, y0, [t0, tf], y_ref, solout=kdv_solout)
    err_ref_6 = np.array([1.0836832917e-05,   9.9714488340e-08,   1.0502531783e-10,    1.8228543865e-12,   5.6552581702e-13,   2.3612922649e-12]) 
    check(err, err_ref_6, "Test 6")
    print err
   
    # Test 7
    t0, tf = 0, 3.
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers.txt")
    err = regression_tst(burgers_func, y0, [t0, tf], y_ref, solout=burgers_solout, tol_boundary=(0,4))
    err_ref_7 = np.array([4.4209014256e-08,   4.6450102760e-11,   6.4950357912e-13,   1.6405125244e-14]) 
    check(err, err_ref_7, "Test 7")
    print err
     
    print("All tests passed")


def dense_tests():
    print("\n Executing dense tests")
    # Test 1 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_1(t0)
    y_ref = exact_1([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_1, y0, t, y_ref)
    err_ref_1 = np.array([ 1.9949855355e-03,   1.8730357330e-05,   9.6645858019e-07,   2.9442885980e-07,   2.5058865168e-07,   5.3709499479e-09]) 
    check(err, err_ref_1, "Test 1 dense")
    print err
      
    # Test 2 dense 
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_2(t0)
    y_ref = exact_2([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_2, y0, t, y_ref)
    err_ref_2 = np.array([1.9986826139e-02, 8.0729032796e-05, 6.4759621032e-07, 3.3835824885e-07, 5.6336165545e-08, 2.1270889427e-09])
    check(err, err_ref_2, "Test 2 dense")
    print err
  
    # Test 3 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_3(t0)
    y_ref = exact_3(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_3, y0, t, y_ref)
    err_ref_3 = np.array([9.7730360766e-07, 1.8066876369e-07, 7.3135413266e-08, 5.3546940651e-09, 1.0413606653e-08, 8.3899459586e-11]) 
    check(err, err_ref_3, "Test 3 dense")
    print err
 
    # Test 4 dense
    t0 = 0.5
    t = [t0,2.5,5,7.5,10]
    y0 = exact_4(t0)
    y_ref = exact_4(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_4, y0, t, y_ref)
    err_ref_4 = np.array([6.5637056667e-03, 1.0019008999e-04, 5.7943976364e-07, 5.7887658428e-09, 8.9361296728e-11, 4.9891671699e-11]) 
    check(err, err_ref_4, "Test 4 dense")
    print err
    
    print("All tests passed")
    #TODO: add tests 5, 6 and 7 for dense output

def implicit_dense_tests():
    #TODO: to implement
    print("To implement")

def inputTuple(k,denseOutput,test,firstStep,robustness, order):    
    standardTuple = {'func': test.RHSFunction, 'grad': test.RHSGradient, 'y0': test.initialValue, 't': denseOutput
                     ,'full_output': True, 'h0': firstStep, 'mxstep': 10e8, 'robustness': robustness,
                     'adaptative':'fixed', 'p':order}    
    
    midimplicitTuple = standardTuple.copy()
    midimplicitTuple.update({'smoothing': 'gbs','seq':None})
    midsemiimplicitTuple = standardTuple.copy()
    midsemiimplicitTuple.update({'smoothing': 'semiimp' ,'seq':(lambda t: 2*(2*t-1))})
    eulersemiimplicitTuple = standardTuple.copy()
    eulersemiimplicitTuple.update({'smoothing': 'no','seq':(lambda t: 2*(2*t-1))})
    optimalTuples =[
        midimplicitTuple
        ,
        midsemiimplicitTuple
        ,
        eulersemiimplicitTuple
    ]
    return optimalTuples[k]

def convergenceTest(test, allSteps, order):
    '''''
       Perform a convergence test with the test problem (in test parameter) with
       the given steps in parameter allSteps. 
    '''''
    dense=False
    useOptimal = True
    solverFunctions = [
        ex_parallel.ex_midpoint_implicit_parallel
        ,
        ex_parallel.ex_midpoint_semi_implicit_parallel
        ,
        ex_parallel.ex_euler_semi_implicit_parallel
        ]
    labelsFunction=[
        "Implicit parl"
        ,
        "SemiImp Midpoint parl"
        ,
        "SemiImp Euler parl"
        ]

    y_ref = np.loadtxt(tst.getReferenceFile(test.problemName))
    denseOutput = test.denseOutput
    if(not dense):
        y_ref=y_ref[1]
        denseOutput=[denseOutput[0], denseOutput[1]]
    print(test.problemName + " order: " + str(order))
    k=0
    for solverFunction in solverFunctions:
        errorPerStep=[]
        print(labelsFunction[k])
        for step in allSteps:
            #rtol and atol are not important as we are fixing the step size
            functionTuple=inputTuple(k,denseOutput, test,step,3, order)
            ys, infodict = solverFunction(**functionTuple)
            
            print("number steps: " + str(infodict['nst']) + " (should be " + str(denseOutput[1]/step) + ")")
            ys=ys[1:len(ys)]
            error = np.linalg.norm(y_ref-ys, 2)
            errorPerStep.append(error)
        
        coefficients = np.polyfit(np.log(allSteps), np.log(errorPerStep), 1)
#         np.testing.assert_array_almost_equal(coefficients[0], p, 1, "CONVERGENCE TEST " + test.problemName + " " + labelsFunction[k] + " FAILED")
        print("coefficients: " + str(coefficients) + " order is: " + str(order))
        
        if(plotConv):
            fig = plt.figure()
            fig.suptitle(test.problemName + " " + labelsFunction[k])
            plt.plot(np.log10(allSteps),np.log10(errorPerStep), marker="x")
        print(errorPerStep)
        k+=1
    plt.show()

def doAllConvergenceTests():
    global plotConv
    plotConv=True
    
    linearSteps = [0.5,0.4,0.25,0.1,0.08,0.05,0.025,0.01,0.005,0.003,0.001]
    convergenceTest(tst.LinearProblem(),linearSteps[1:],2)
    convergenceTest(tst.LinearProblem(),linearSteps[1:],4)
    convergenceTest(tst.LinearProblem(),linearSteps[0:7],6)
    convergenceTest(tst.LinearProblem(),linearSteps[0:5],8)
    
#     vdpolSteps=[0.5,0.000005]
#     convergenceTest(tst.VDPOLEasyProblem(),vdpolSteps[:],8)
    

if __name__ == "__main__":
#     non_dense_tests()
#     dense_tests()
#     implicit_dense_tests()
    doAllConvergenceTests()
    




