from __future__ import division
import numpy as np
import math 
import time
import ex_parallel
import ex_parallel_original
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
        ys, infodict = ex_parallel.ex_midpoint_explicit_parallel(func,None, y0, t, atol=tol[i], 
            rtol=tol[i], mxstep=mxstep, full_output=True, nworkers=nworkers)
        y = solout(ys[1:len(ys)])
        err[i] = relative_error(y, y_ref)
        print (y-y_ref)
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
    err_ref_1 = np.array([ 4.3573323310e-04,   2.0155849793e-04,   6.1571235657e-06,
         1.3514865737e-06,   9.9131327408e-08,   1.9365386445e-08]) 
    check(err, err_ref_1, "Test 1 dense")
    print err
      
    # Test 2 dense 
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_2(t0)
    y_ref = exact_2([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_2, y0, t, y_ref)
    err_ref_2 = np.array([  5.1994929145e-03,   8.1213665526e-05,   1.8579393108e-06,
         1.2171334245e-06,   4.1389667376e-08,   9.7372577540e-08])
    check(err, err_ref_2, "Test 2 dense")
    print err
  
    # Test 3 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_3(t0)
    y_ref = exact_3(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_3, y0, t, y_ref)
    err_ref_3 = np.array([ 3.8670623384e-07,   2.4122935757e-07,   1.2818328551e-07,
         2.9523589375e-08,   4.7034490983e-10,   3.4956885841e-11]) 
    check(err, err_ref_3, "Test 3 dense")
    print err
 
    # Test 4 dense
    t0 = 0.5
    t = [t0,2.5,5,7.5,10]
    y0 = exact_4(t0)
    y_ref = exact_4(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_4, y0, t, y_ref)
    err_ref_4 = np.array([  1.0309663893e-02,   6.2166423103e-05,   4.2554994818e-07,
         3.1358489245e-09,   4.3910896661e-11,   7.6981520979e-12]) 
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
        
    standardOldTuple = {'func': test.RHSFunction, 'y0': test.initialValue, 't': denseOutput
                     ,'full_output': True, 'h0': firstStep, 'mxstep': 10e8,
                     'adaptive':'fixed', 'p':order}
    
    midimplicitTuple = standardTuple.copy()
    midimplicitTuple.update({'smoothing': 'no','seq':None})
    midsemiimplicitTuple = standardTuple.copy()
    midsemiimplicitTuple.update({'smoothing': 'no' ,'seq':(lambda t: 2*(2*t-1))})
    eulersemiimplicitTuple = standardTuple.copy()
    eulersemiimplicitTuple.update({'smoothing': 'no','seq':(lambda t: 2*(2*t-1))})
    optimalTuples =[
#         standardTuple
#         ,
#         standardOldTuple
#         ,
#         midimplicitTuple
#         ,
        midsemiimplicitTuple
        ,
        eulersemiimplicitTuple
    ]
    return optimalTuples[k]

labelsFunction=[
#     "New Explicit Midpoint parl"
#     ,
#     "Old Explicit parl"
#     ,
#     "Implicit Midpoint parl"
#     ,
    "SemiImp Midpoint parl"
    ,
    "SemiImp Euler parl"
      ]

def convergenceTest(test, allSteps, order, dense=False):
    '''''
       Perform a convergence test with the test problem (in test parameter) with
       the given steps in parameter allSteps. 
    '''''
    print("\n" + "order: " + str(order) + ", dense: " +str(dense))
    useOptimal = True
    solverFunctions = [
#         ex_parallel.ex_midpoint_explicit_parallel
#         ,
#         ex_parallel_original.ex_midpoint_parallel
#         ,
#         ex_parallel.ex_midpoint_implicit_parallel
#         ,
        ex_parallel.ex_midpoint_semi_implicit_parallel
        ,
        ex_parallel.ex_euler_semi_implicit_parallel
        ]

    y_ref = np.loadtxt(tst.getReferenceFile(test.problemName))
    denseOutput = test.denseOutput
#     denseOutput = np.linspace(0,12,100)
    if(not dense):
        y_ref=y_ref[-1]
        denseOutput=[denseOutput[0], denseOutput[-1]]
    else:
        nhalf = np.ceil(len(y_ref)/2)
        y_ref = y_ref[nhalf]
        print(denseOutput[nhalf])
    print(test.problemName + " order: " + str(order))
    k=0
    errorperFunction = np.zeros((len(labelsFunction)+1,len(allSteps)))
    errorperFunction[0,:]=allSteps
    for solverFunction in solverFunctions:
        errorPerStep=[]
        print(labelsFunction[k])
        for step in allSteps:
            #rtol and atol are not important as we are fixing the step size
            functionTuple=inputTuple(k,denseOutput, test,step,3, order)
            ys, infodict = solverFunction(**functionTuple)
#             print(ys)
#             plt.plot(denseOutput,ys,'ok')
#             plt.show()
            print("number steps: " + str(infodict['nst']) + " (should be " + str(denseOutput[-1]/step) + ")")
            ys=ys[1:len(ys)]
            if(dense):
                ys=ys[nhalf]
            error = np.linalg.norm((y_ref-ys)/y_ref, 2)
            errorPerStep.append(error)
        
        coefficients = np.polyfit(np.log10(allSteps), np.log10(errorPerStep), 1)
#         np.testing.assert_array_almost_equal(coefficients[0], p, 1, "CONVERGENCE TEST " + test.problemName + " " + labelsFunction[k] + " FAILED")
        print("coefficients: " + str(coefficients) + " order is: " + str(order))
        
        if(plotConv):
            fig = plt.figure()
            fig.suptitle(test.problemName + " , " + labelsFunction[k] + " , " + str(step))
            plt.plot(np.log10(allSteps),np.log10(errorPerStep), marker="x")
        print(allSteps)
        print(errorPerStep)
        errorperFunction[k+1,:] = errorPerStep
        k+=1
#     plt.show()
    return errorperFunction

def plotConvergence(allSteps, errorPerFunction, order):
    for k in range(len(labelsFunction)):
        errorPerStep = errorPerFunction[k]
        fig = plt.figure()
        fig.suptitle(labelsFunction[k] + " order " + str(order))
        plt.plot(np.log10(allSteps),np.log10(errorPerStep), marker="x")
        print(labelsFunction[k])
        coefficients = np.polyfit(np.log10(allSteps), np.log10(errorPerStep), 1)
        print("coefficients: " + str(coefficients) + " order is: " + str(order))
        m,b=coefficients
        plt.plot(np.log10(allSteps), m*np.log10(allSteps) + b, '-')

def doAllConvergenceTests():
    global plotConv
    plotConv=True
#     orders = range(2,12,2)
#     linearSteps = [1.5,1.25,1,0.5,0.1,0.07,0.05,0.03,0.01,0.007,0.005,0.002,0.001,0.0007,0.0005,0.0004,0.0003]
#     allerrorPerStep = np.zeros((len(orders),len(linearSteps)))
#     for i in orders:
#         allerrorPerStep[i/2-1] = convergenceTest(tst.LinearProblem(),linearSteps,i)
# #     np.savetxt("AllErrorPerStep.txt", allerrorPerStep)
#     plt.show()
    order = 8
    linearSteps = np.concatenate((np.linspace(0.5,0.2,4), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.002,10),np.linspace(0.0019,0.001,10)))
    errorPerFunction = convergenceTest(tst.LinearProblem(),linearSteps,order,True)
    np.savetxt("allerrorperstep_"+str(order)+"_nosmooth"+"_dense2"+".txt", errorPerFunction)
    plt.show()
    #For vdpol
#     begend = np.array([[0,0],[0,10],[2,23],[1,8],[0,16]])
    #For linear
#     begend = np.array([[0,5],[0,20],[0,15],[0,25],[0,0]])
    #For dense linear
#     begend = np.array([[0,5],[0,20],[0,20],[0,0],[0,0]])
# #     begend = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
#     orders = np.array([2,4,6,8])
#     k=0
#     for order in orders:
#         errorPerFunction = np.loadtxt("allerrorperstep_"+str(order)+"_nosmooth"+"_dense"+".txt")
#         allSteps = errorPerFunction[0]
#         errorPerFunction= errorPerFunction[1:(len(labelsFunction)+1),begend[k,0]:(len(allSteps)-begend[k,1])]
#         allSteps =  allSteps[begend[k,0]:(len(allSteps)-begend[k,1])]
#         plotConvergence(allSteps, errorPerFunction, order)
#         k+=1
#     plt.show()

        
#     convergenceTest(tst.LinearProblem(),[0.05,0.01,0.005,0.001,0.0005,0.0002,0.00015],2,True)
#     convergenceTest(tst.LinearProblem(),[0.5,0.07,0.05,0.01,0.005,0.001,0.0009],4,True)
#     convergenceTest(tst.LinearProblem(),[1,0.7,0.5,0.1,0.07,0.05,0.03,0.01,0.007],6,True)
#     convergenceTest(tst.LinearProblem(),[1.7,1.5,1.25,1,0.75,0.5,0.1,0.07,0.05],8,True)
    #TODO: investigate why with order 10 a singular matrix error is thrown.
#     convergenceTest(tst.LinearProblem(),[2.3,2,1.7,1.5,1.25,1,0.75,0.5,0.1,0.07,0.05],10,True)
#     plt.show()
    
#     vdpolSteps=[0.5,0.000005]
#     convergenceTest(tst.VDPOLEasyProblem(),vdpolSteps[:],8)
    
def polynomial(coeff,x):
    sum=0
    for i in range(len(coeff)):
        sum+=coeff[i]*x**i
    return np.array([sum])

def checkInterpolationPolynomial():
    order=4
    rndpoly = np.random.randn(order+1)
    print(rndpoly)
    steps = [0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    errorPerStep = np.zeros(len(steps))
    errorIntPerStep = np.zeros(len(steps))
    seq=(lambda t: 4*t-2)
    t0=0
    k=0
    for H in steps:
        y0 = polynomial(rndpoly,t0)
        Tkk = polynomial(rndpoly,H)
        yj = (order+1)*[None]
        hs = (order+1)*[None]
        for i in range(1,order+1):
            ni = seq(i)
            yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
            for j in range(ni+1):
                yj_[j]=polynomial(rndpoly,j*H/ni)
            yj[i]=yj_
            hs[i]=H/ni

        poly = ex_parallel.interpolate_nonsym(y0, Tkk, yj, hs, H, order, atol=1e-1,rtol=1e-1, seq=seq)
        
        x=H/2;
        res,errint,hint = poly(x)
        errorIntPerStep[k]=errint
        resexact=polynomial(rndpoly,t0+H*x)
        errorPerStep[k] = np.linalg.norm((res-resexact)/resexact)
        k+=1
    
    print(steps)
    print(errorPerStep)
    fig = plt.figure()
    coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStep), 1)
    print("coefficients error " + str(coefficients))
    plt.plot(np.log10(steps),np.log10(errorPerStep), marker="x")
    
    print(steps)
    print(errorIntPerStep)
    fig = plt.figure()
    coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStep), 1)
    print("coefficients error interpolation" + str(coefficientsint))
    plt.plot(np.log10(steps),np.log10(errorIntPerStep), marker="x")
    plt.show()
#     for x in np.linspace(0,H,6):
#         res,errint,hint = poly(x)
#         print("approx: " +str(res)+" exact: " + str(polynomial(rndpoly,t0+H*x)))

if __name__ == "__main__":
#     non_dense_tests()
#     dense_tests()
#     implicit_dense_tests()
#     doAllConvergenceTests()
    checkInterpolationPolynomial()
    




