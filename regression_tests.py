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
<<<<<<< HEAD
    err_ref_1 = np.array([  2.1630698948e-03,   1.9757059466e-05,   1.3907151930e-07,
         1.1075581382e-09,   9.3295927483e-12,   1.1965242901e-13]) 
    check(err, err_ref_1, "Test 1: linear ODE")
    print err
  
=======
    err_ref_1 = np.array([1.02154130e-03, 1.22710679e-05, 9.25784586e-08, 7.71182340e-10, 4.91119065e-12, 4.44094761e-14]) 
    check(err, err_ref_1, "Test 1: linear ODE")
    print err
 
>>>>>>> master
    # Test 2
    t0, tf = 0, 10
    y0 = exact_2(t0)
    y_ref = exact_2(tf)
    err = regression_tst(f_2, y0, [t0, tf], y_ref)
<<<<<<< HEAD
    err_ref_2 = np.array([4.9732202037e-03,   2.0319706692e-05,   4.3446608751e-07,
         2.6994205605e-09,   4.5295951528e-11,   1.6383672339e-12])
    check(err, err_ref_2, "Test 2")
    print err
    
=======
    err_ref_2 = np.array([8.54505911e-02, 2.48291275e-04, 8.17836269e-07, 2.85969698e-09, 2.54201200e-11, 9.10317032e-13])
    check(err, err_ref_2, "Test 2")
    print err
 
>>>>>>> master
    # Test 3
    t0, tf = 0, 10
    y0 = exact_3(t0)
    y_ref = exact_3(tf)
    err = regression_tst(f_3, y0, [t0, tf], y_ref)
<<<<<<< HEAD
    err_ref_3 = np.array([8.0615031844e-07,   1.3156476857e-07,   2.3068719262e-09,
         2.1587147846e-11,   1.7617503487e-13,   1.6048333136e-15]) 
    check(err, err_ref_3, "Test 3")
    print err
    
=======
    err_ref_3 = np.array([1.93968837e-05, 9.48240508e-08, 1.21300220e-09, 2.83645194e-10, 1.74748516e-13, 5.88438882e-15]) 
    check(err, err_ref_3, "Test 3")
    print err
 
>>>>>>> master
    # Test 4
    t0, tf = 0.5, 10
    y0 = exact_4(t0)
    y_ref = exact_4(tf)
    err = regression_tst(f_4, y0, [t0, tf], y_ref)
<<<<<<< HEAD
    err_ref_4 = np.array([9.9765055429e-03,   5.5735241863e-05,   3.4917465397e-07,
         2.7457698162e-09,   2.1222716857e-11,   4.4841830265e-14]) 
    check(err, err_ref_4, "Test 4")
    print err
      
=======
    err_ref_4 = np.array([6.77863684e-03, 9.50343703e-05, 4.90535263e-07, 3.36128205e-09, 2.47615358e-11, 2.92147596e-13]) 
    check(err, err_ref_4, "Test 4")
    print err
  
>>>>>>> master
    # Test 5
    t0, tf = 0, 0.08
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference.txt")
    err = regression_tst(f_5, y0, [t0, tf], y_ref)
<<<<<<< HEAD
    err_ref_5 = np.array([ 5.7624705654e-01,   2.0529390615e-01,   1.4130955311e-02,
         7.9667511556e-06,   1.7720280179e-07,   5.0404934818e-09]) 
    check(err, err_ref_5, "Test 5")
    print err
    
#     # Test 6
#     t0, tf = 0, 0.003
#     y0 = kdv_init(t0)
#     y_ref = np.loadtxt("reference_kdv.txt")
#     err = regression_tst(kdv_func, y0, [t0, tf], y_ref, solout=kdv_solout)
#     err_ref_6 = np.array([]) 
#     check(err, err_ref_6, "Test 6")
#     print err
#    
#     # Test 7
#     t0, tf = 0, 3.
#     y0 = burgers_init(t0)
#     y_ref = np.loadtxt("reference_burgers.txt")
#     err = regression_tst(burgers_func, y0, [t0, tf], y_ref, solout=burgers_solout, tol_boundary=(0,4))
#     err_ref_7 = np.array([4.4209014256e-08,   4.6450102760e-11,   6.4950357912e-13,   1.6405125244e-14]) 
#     check(err, err_ref_7, "Test 7")
#     print err
     
    print("All tests passed")
=======
    err_ref_5 = np.array([4.6785364959e-01, 8.0415058627e-02, 1.6090309508e-03, 1.0235932826e-05, 1.0988144963e-07, 1.1568408676e-09]) 
    check(err, err_ref_5, "Test 5")
    print err
 
    # Test 6
    t0, tf = 0, 0.003
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv.txt")
    err = regression_tst(kdv_func, y0, [t0, tf], y_ref, solout=kdv_solout)
    err_ref_6 = np.array([1.20373187e-05, 7.98230720e-08, 1.63759715e-10, 1.75095038e-12, 3.66764576e-12, 1.69146958e-12]) 
    check(err, err_ref_6, "Test 6")
    print err
 
    # Test 7
    t0, tf = 0, 3.
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers.txt")
    err = regression_tst(burgers_func, y0, [t0, tf], y_ref, solout=burgers_solout, tol_boundary=(0,4))
    err_ref_7 = np.array([6.92934673e-09, 4.45755379e-11, 6.26721092e-13, 2.49897416e-14]) 
    check(err, err_ref_7, "Test 7")
    print err
>>>>>>> master


def dense_tests():
    print("\n Executing dense tests")
    # Test 1 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_1(t0)
    y_ref = exact_1([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_1, y0, t, y_ref)
<<<<<<< HEAD
#     err_ref_1 = np.array([ 1.2796005275e-03,   1.4160078828e-05,   1.1154717428e-07,
#          6.6923205526e-10,   2.5033682435e-11,   5.4255003010e-12]) 
    err_ref_1 = np.array([ 4.0388541531e-04,   3.6069536594e-06,   1.7350073446e-08,
         2.9149161935e-10,   8.3688477982e-13,   2.0974333704e-13]) 
=======
    err_ref_1 = np.array([1.45770475e-04, 1.14841518e-06, 1.54196490e-06, 3.88285196e-06, 1.26814157e-07, 1.88666778e-08]) 
>>>>>>> master
    check(err, err_ref_1, "Test 1 dense")
    print err
      
    # Test 2 dense 
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_2(t0)
    y_ref = exact_2([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_2, y0, t, y_ref)
<<<<<<< HEAD
#     err_ref_2 = np.array([ 1.4763869532e-03,   5.9302773357e-05,   1.9116074302e-07,
#          1.4820318333e-09,   2.3923358893e-11,   2.1980051033e-12])
    err_ref_2 = np.array([ 2.6556611408e-03,   5.6819363991e-04,   1.0109797259e-06,
         1.3020804372e-08,   7.3727740776e-11,   5.6430123318e-13])
    check(err, err_ref_2, "Test 2 dense")
    print err
  
    # Test 3 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_3(t0)
    y_ref = exact_3(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_3, y0, t, y_ref)
#     err_ref_3 = np.array([ 5.4288121430e-05,   7.3657652413e-07,   4.7892864282e-09,
#          8.3632503120e-11,   1.3985673220e-10,   2.3489431842e-13])
    err_ref_3 = np.array([  3.1995189542e-07,   1.1009718320e-07,   7.5469109379e-09,
         1.0636289556e-11,   8.6409834669e-14,   4.7465438790e-14]) 
    check(err, err_ref_3, "Test 3 dense")
    print err
=======
    err_ref_2 = np.array([3.17429818e-04, 4.86595063e-05, 1.10295116e-05, 1.99391509e-06, 2.11069640e-07, 9.94570884e-08])
    check(err, err_ref_2, "Test 2 dense")
    print err
  
    # Test 3 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_3(t0)
    y_ref = exact_3(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_3, y0, t, y_ref)
    err_ref_3 = np.array([8.99830265e-06, 2.02137818e-06, 4.63896854e-08, 2.05977138e-08, 2.36656866e-09, 1.44506569e-09]) 
    check(err, err_ref_3, "Test 3 dense")
    print err
>>>>>>> master
 
    # Test 4 dense
    t0 = 0.5
    t = [t0,2.5,5,7.5,10]
    y0 = exact_4(t0)
    y_ref = exact_4(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_4, y0, t, y_ref)
<<<<<<< HEAD
#     err_ref_4 = np.array([  1.6218737110e-02,   2.5189774159e-05,   8.2681493532e-07,
#          7.8329039603e-10,   3.2782336939e-11,   7.4040697301e-12])
    err_ref_4 = np.array([  1.1139882478e-02,   5.8477866642e-05,   3.9084471181e-07,
         3.9017464759e-09,   2.8246633161e-11,   2.0101404777e-13]) 
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
    
    #In this optimal set-up remove smoothing step because it makes the algorithm loose one 
    #convergence order (see pag. 135 theorem 9.1)
    midimplicitTuple = standardTuple.copy()
    midimplicitTuple.update({'smoothing': 'no','seq':None})
    midsemiimplicitTuple = standardTuple.copy()
    midsemiimplicitTuple.update({'smoothing': 'no' ,'seq':(lambda t: 2*(2*t-1))})
    eulersemiimplicitTuple = standardTuple.copy()
    eulersemiimplicitTuple.update({'smoothing': 'no','seq':(lambda t: 2*(2*t-1))})
    optimalTuples =[
        standardTuple
        ,
#         standardOldTuple
#         ,
        midimplicitTuple
        ,
        midsemiimplicitTuple
        ,
        eulersemiimplicitTuple
    ]
    return optimalTuples[k]

labelsFunction=[
    "New Explicit Midpoint parl"
    ,
#     "Old Explicit parl"
#     ,
    "Implicit Midpoint parl"
    ,
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
        ex_parallel.ex_midpoint_explicit_parallel
        ,
#         ex_parallel_original.ex_midpoint_parallel
#         ,
        ex_parallel.ex_midpoint_implicit_parallel
        ,
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
 
    order = 4
    #linear: 2,4
    linearSteps = np.concatenate((np.linspace(0.5,0.2,4), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.002,10),np.linspace(0.0019,0.001,10)))
    #vdpol: 2,4
    linearSteps = np.concatenate((np.linspace(0.5,0.2,4), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
                                  np.linspace(0.019,0.005,10),np.linspace(0.0049,0.002,10)))
    
    #linear: 6 
#     linearSteps = np.concatenate((np.linspace(0.7,0.2,3), np.linspace(0.19,0.04,7),np.linspace(0.039,0.02,7),
#                                   np.linspace(0.019,0.005,10)))
    #vdpol: 6,8 
#     linearSteps = np.concatenate((np.linspace(1.5,0.75,5),np.linspace(0.7,0.2,5), np.linspace(0.19,0.04,7),
#                                   np.linspace(0.039,0.02,7)))
    errorPerFunction = convergenceTest(tst.VDPOLEasyProblem(),linearSteps,order,False)
    np.savetxt("allerrorperstep_"+str(order)+"_nosmooth"+"_vdpol"+"_dense"+".txt", errorPerFunction)
    plt.show()

    #For vdpol
#     begend = np.array([[0,0],[0,10],[2,23],[1,8],[0,16]])
    #For linear
#     begend = np.array([[0,5],[0,20],[0,15],[0,25],[0,0]])
    #For dense linear
#     begend = np.array([[0,5],[0,5],[0,25],[0,0],[0,0]])
# #     begend = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
#     orders = np.array([2,4])#,6,8])
#     k=0
#     for order in orders:
#         errorPerFunction = np.loadtxt("allerrorperstep_"+str(order)+"_nosmooth"+"_vdpol"+"_dense"+".txt")
#         allSteps = errorPerFunction[0]
#         errorPerFunction= errorPerFunction[1:(len(labelsFunction)+1),begend[k,0]:(len(allSteps)-begend[k,1])]
#         allSteps =  allSteps[begend[k,0]:(len(allSteps)-begend[k,1])]
#         plotConvergence(allSteps, errorPerFunction, order)
#         k+=1
#     plt.show()

    
    
def polynomial(coeff,x):
    sum=0
    for i in range(len(coeff)):
        sum+=coeff[i]*x**i
#     return np.array([sum])
    return np.array([np.exp(-x)])
    return np.array([np.cos(1.34*x)])

def polynomialder(coeff,x,orderder):
    sum=0
    for i in range(orderder,len(coeff)):
        sum+=math.factorial(i)/math.factorial(i-orderder)*coeff[i]*x**(i-orderder)
#     return np.array([sum])
    return np.array([(-1)**orderder*np.exp(-x)])
    if(np.mod(orderder,2)==1):
        return np.array([(-1.)**((orderder+1)/2)*1.34**orderder*np.sin(1.34*x)])
    else:
        return np.array([(-1.)**(orderder/2)*1.34**orderder*np.cos(1.34*x)])

def checkInterpolationPolynomial():
    order=2
    extraord=0
    #Order for symmetric interpolation is +4
#     extraord=4
    rndpoly = np.random.randn(order+1+extraord)
    rndpoly = np.ones(order+1+extraord)
    print(rndpoly)
    steps = [0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    #for symmetric interpolation
    steps = [0.5,0.55,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15]
#     steps = [0.04,0.06,0.08,0.1,0.2,0.3,0.35,0.4,0.5,0.6]
    errorPerStep = np.zeros(len(steps))
    errorIntPerStep = np.zeros(len(steps))
    errorPerStepSym = np.zeros(len(steps))
    errorIntPerStepSym = np.zeros(len(steps))
    seq=(lambda t: 4*t-2)
    t0=0
    k=0
    for H in steps:
#         H=1
        y0 = polynomial(rndpoly,t0)
        Tkk = polynomial(rndpoly,t0+H)
        f_Tkk = polynomialder(rndpoly, t0+H,1)
        yj = (order+1)*[None]
        f_yj = (order+1)*[None]
        hs = (order+1)*[None]
        y_half = (order+1)*[None]
        for i in range(1,order+1):
            ni = seq(i)
            yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
            f_yj_ = np.zeros((ni+1, len(y0)), dtype=(type(y0[0])))
            for j in range(ni+1):
                yj_[j]=polynomial(rndpoly,j*H/ni)
                f_yj_[j]=polynomialder(rndpoly,j*H/ni,1)
                
            yj[i]=yj_
            f_yj[i]=f_yj_
            y_half[i]=yj_[ni/2]
            hs[i]=H/ni
        
        rs = np.zeros((order+1), dtype=(type(yj[1][0])))
        for i in range(1,order+1):
            rs[i]=polynomialder(rndpoly, H, i)
            
        ds = np.zeros((2*order+1), dtype=(type(y_half[1])))
        for i in range(2*order+1):
            ds[i]=polynomialder(rndpoly, H/2, i)
        
        poly = ex_parallel.interpolate_nonsym(y0, Tkk, yj, hs, H, order, atol=1e-5,rtol=1e-5, seq=seq)        
        
        polysym = ex_parallel.interpolate_sym(y0, Tkk,f_Tkk, y_half, f_yj, hs, H, order, atol=1e-5,rtol=1e-5, seq=seq)

#         for x in np.linspace(0,H,10):
#             res,errint,hint = polysym(x)
#             print("error: " +str(res-polynomial(rndpoly,t0+H*x)))
#             print("errorint: " +str(errint))
        
        x=H/5;
         
        res,errint,hint = poly(x)
        resexact=polynomial(rndpoly,t0+H*x)
#         errorIntPerStep[k]=np.linalg.norm((errint))
#         errorPerStep[k] = np.linalg.norm((res-resexact))

        ressym,errintsym,hint = polysym(x)
        errorIntPerStepSym[k]=np.linalg.norm(errintsym)
        errorPerStepSym[k] = np.linalg.norm((ressym-resexact))

#         dsapp = ex_parallel.compute_ds(y_half, f_yj, hs, order, seq=seq)
# #         rsapp = ex_parallel.compute_rs(yj, hs, order, seq=seq)
#         der=2
#         print("exact->" + str(ds))
#         print("approx->" + str(dsapp))
#         errorPerStep[k] = np.linalg.norm((ds[der]-dsapp[der]))
#         errorPerStep[k] = np.linalg.norm((ds[der]-dsapp[der]))
                
        k+=1
      
#     print(steps)
#     print "error"+str(errorPerStep)
#     fig = plt.figure()
#     coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStep), 1)
#     print("coefficients error " + str(coefficients))
#     plt.plot(np.log10(steps),np.log10(errorPerStep), marker="x")
#           
#     print(steps)
#     print "error int: " + str(errorIntPerStep)
#     fig = plt.figure()
#     coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStep), 1)
#     print("coefficients error interpolation" + str(coefficientsint))
#     plt.plot(np.log10(steps),np.log10(errorIntPerStep), marker="x")
     
    print(steps)
    print(errorPerStepSym)
    fig = plt.figure()
    coefficients = np.polyfit(np.log10(steps), np.log10(errorPerStepSym), 1)
    print("coefficients error sym " + str(coefficients))
    plt.plot(np.log10(steps),np.log10(errorPerStepSym), marker="x")
         
    print(steps)
    print(errorIntPerStepSym)
    fig = plt.figure()
    coefficientsint = np.polyfit(np.log10(steps), np.log10(errorIntPerStepSym), 1)
    print("coefficients error sym interpolation" + str(coefficientsint))
    plt.plot(np.log10(steps),np.log10(errorIntPerStepSym), marker="x")

    plt.show()

=======
    err_ref_4 = np.array([4.43565713e-03,   9.09774798e-05, 6.04212715e-07, 2.39230449e-09, 5.32280637e-11, 6.44359887e-11]) 
    check(err, err_ref_4, "Test 4 dense")
    print err
    
    #TODO: add tests 5, 6 and 7 for dense output

>>>>>>> master

if __name__ == "__main__":
    non_dense_tests()
    dense_tests()
<<<<<<< HEAD
#     implicit_dense_tests()
#     doAllConvergenceTests()
#     checkInterpolationPolynomial()
    




=======
    
>>>>>>> master
