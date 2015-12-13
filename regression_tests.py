from __future__ import division
import numpy as np
import math 
import ex_parallel
import matplotlib.pyplot as plt

allmethods = ['midpoint explicit','midpoint implicit','midpoint semi implicit','euler explicit','euler semi implicit']

regressionvalues = {'midpoint explicit':[
                        [2.4054371001e-06,   4.3055665867e-07,   6.1728696137e-10,
         1.8917700341e-12,   2.7161091660e-15,   2.7161091660e-15],
                        [2.4641284233e-06,   2.4641284233e-06,   1.9030069619e-09,
         1.4625004260e-09,   1.5762550692e-13,   1.0086956501e-14],
                        [2.2475235488e-06,   1.0300735551e-06,   1.1973036784e-08,
         4.4328429816e-11,   2.2526425170e-13,   8.2156503822e-15]                      
                                         ]                    
                    ,'midpoint implicit':[
                        [1.5293204761e-08,   1.1928156003e-08,   1.5104319287e-09,
         8.1136216588e-13,   4.2250587027e-15,   7.5447476834e-16],
                        [ 1.8031544714e-08,   1.8035168218e-08,   3.1238860457e-09,
         6.0817623062e-13,   1.7484057935e-15,   2.4208695602e-15],
                        [3.8327484742e-08,   1.1024071656e-09,   8.2266193857e-11,
         8.5931262106e-14,   5.9952043330e-15,   1.5543122345e-15]                      
                                         ]
                    ,'midpoint semi implicit':[
                        [8.0450452184e-07,   8.0450452184e-07,   1.1962604868e-09,
         7.9778162005e-13,   1.9616343977e-15,   3.0178990734e-15],
                        [ 2.2771937896e-08,   2.2771937896e-08,   9.8490786300e-09,
         7.3030909994e-11,   2.0604289812e-13,   3.0933333269e-15],
                        [9.1299733940e-07,   9.1299733940e-07,   2.2042861980e-09,
         1.5252021868e-11,   1.0214051827e-13,   8.3266726847e-15]                        
                                         ]
                    ,'euler explicit':[
                        [ 5.1530388716e-04,   1.0316403746e-06,   2.1394378449e-08,
         4.2737343969e-10,   3.4526274349e-12,   3.3800469622e-13],
                        [4.7945288696e-04,   4.1676312202e-05,   2.7430285643e-07,
         2.4722151276e-09,   3.9390775571e-11,   1.2338365192e-12],
                        [5.5596474756e-04,   1.5538277741e-06,   4.6832369094e-09,
         3.6981750995e-11,   1.8984813721e-13,   2.2148949341e-13]                       
                                         ]
                    ,'euler semi implicit':[
                        [2.3100925231e-06,   1.6777735428e-06,   1.1281401974e-08,
         6.8942395382e-11,   2.5335262721e-13,   2.8458788262e-13],
                        [7.6971229056e-06,   2.8710945728e-06,   6.0000955762e-08,
         8.1554684121e-10,   1.4732739680e-11,   3.5129507174e-13],
                        [4.0039427062e-06,   8.3667530160e-07,   1.2044836240e-09,
         3.1111668797e-11,   1.4654943925e-14,   1.7763568394e-14]                       
                                         ]}

regressionvaluesdense = {'midpoint explicit':[
                        [3.4642501161e-07,   3.4642501161e-07,   4.3374772739e-09,
         9.4880259659e-13,   1.9190810732e-15,   2.2736380777e-15],
                        [2.9323421672e-07,   2.9323421672e-07,   8.1102149183e-09,
         1.8421823156e-11,   2.8988487086e-13,   2.4080152326e-14],
                        [4.8941277641e-07,   4.8941277641e-07,   8.9348270645e-09,
         6.3902126450e-11,   2.5726071861e-13,   6.4646253816e-15]                        
                                         ]                    
                    ,'midpoint implicit':[
                        [ 4.5573535834e-07,   4.5386635234e-07,   1.3496938571e-07,
         3.7594370070e-10,   3.6022394936e-08,   6.9953683983e-09],
                        [8.2413858176e-07,   8.2414062859e-07,   6.0333530921e-08,
         1.2261597675e-09,   1.1225879293e-09,   3.5890791004e-09],
                        [1.3820565861e-06,   1.3758790352e-06,   7.0553999910e-08,
         1.1037088895e-07,   1.5887463787e-08,   2.6978897733e-09]                       
                                         ]
                    ,'midpoint semi implicit':[
                        [1.6844768967e-06,   1.6844768967e-06,   3.0290113955e-08,
         6.3914198269e-09,   5.8442029708e-08,   8.4710131146e-09],
                        [1.1506381998e-06,   1.1506381998e-06,   1.1279165511e-08,
         5.0249290647e-09,   4.6299269576e-08,   6.1071721315e-10],
                        [1.4565000722e-06,   1.4565000722e-06,   1.6342739136e-08,
         3.2428177132e-08,   6.8445978340e-10,   1.0194164228e-07]                       
                                         ]
                    ,'euler explicit':[
                        [2.3923420321e-04,   2.0785557794e-06,   1.0785176777e-08,
         1.7137734957e-10,   1.3284884920e-12,   2.1175682269e-12],
                        [3.8313415427e-04,   4.1416950951e-06,   2.4942434710e-07,
         2.3087815273e-09,   3.7480260147e-11,   1.8490916834e-12],
                        [4.9191579020e-04,   1.4630518763e-06,   4.9565055200e-09,
         3.5178267705e-11,   8.7695466953e-13,   1.2257225905e-12]                       
                                         ]
                    ,'euler semi implicit':[
                        [1.1878532624e-05,   1.1581152109e-05,   1.2606825472e-08,
         5.8124968603e-11,   2.3268957453e-11,   1.1943356002e-12],
                        [4.6834255496e-04,   1.5255137462e-06,   1.7013210124e-09,
         8.1260873044e-10,   1.5196170114e-11,   4.2986541030e-13],
                        [1.0144633464e-05,   5.8557909975e-08,   3.4065216128e-09,
         3.9356063873e-10,   2.4978308382e-10,   2.6221456459e-12]                        
                                         ]}


def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def regression_tst(method, func, y0, t, y_ref, tol_boundary=(0,6), h0=0.5, mxstep=10e6,
        adaptive="order", p=4, solout=(lambda t: t), nworkers=2):
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]

    err = np.zeros(len(tol))
    print ''
    for i in range(len(tol)):
        print tol[i]
        ys, infodict = ex_parallel.extrapolation_parallel(method,func, None, y0, t, atol=tol[i], 
            rtol=tol[i], mxstep=mxstep, full_output=True, nworkers=nworkers)
        y = solout(ys[1:len(ys)])
        err[i] = relative_error(y, y_ref)
    return err

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
    print("\n Executing regression non dense tests")
    
    for method in allmethods:
        print("\n Method: " + method)
        k=0
        for test in alltestfunctions:
            print("\n Test: " + str(k))
            (f,exact) = test
            t0, tf = 0.1, 1
            y0 = exact(t0)
            y_ref = exact(tf)
            err = regression_tst(method, f, y0, [t0, tf], y_ref)
            err_ref = regressionvalues[method][k] 
            check(err, err_ref, "Test " + str(k))
            k+=1
            print err
     
    print("All tests passed")


def dense_tests():
    print("\n Executing regression dense tests")
    
    for method in allmethods:
        print("\n Method " + method)
        k=0
        for test in alltestfunctions:
            print("\n Test " + str(k))
            (f,exact) = test
            t0 = 0.1
            t=[t0,0.25,0.5,0.75,1]
            y0 = exact(t0)
            y_ref = exact([[t[1]],[t[2]],[t[3]],[t[4]]])
            err = regression_tst(method, f, y0, t, y_ref)
            err_ref = regressionvaluesdense[method][k] 
            check(err, err_ref, "Test " + str(k))
            k+=1
            print err
                
    print("All tests passed")


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

if __name__ == "__main__":
    non_dense_tests()
    dense_tests()

#     doAllConvergenceTests()
#     checkInterpolationPolynomial()
    

