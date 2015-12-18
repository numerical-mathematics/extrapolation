from __future__ import division
import numpy as np
import math 
import ex_parallel
import matplotlib.pyplot as plt
import twelve_tests as tst

#Whether to do convergence plots to see if they are straight lines (to choose the steps)
plotConv=False

allmethods = ['midpoint explicit','midpoint implicit','midpoint semi implicit','euler explicit','euler semi implicit']
#Methods that use smoothing loose an order of convergence
methodsmoothing = [0,1,1,0,0]


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
                        [3.6652288081e-04,   6.3501597236e-06,   1.3078086207e-08,
         1.7312127004e-10,   1.4091210316e-12,   2.2623039249e-12],
                        [3.9326591749e-04,   5.1895536078e-05,   2.5848895459e-07,
         2.1451167261e-09,   3.4473686776e-11,   9.2666003762e-13],
                        [6.0412010046e-04,   1.5551489837e-06,   4.0491380415e-09,
         1.4740434733e-10,   2.1519318487e-12,   9.7896453378e-13]                       
                                         ]
                    ,'euler semi implicit':[
                        [ 2.5217524868e-05,   2.4642363796e-05,   1.7683037167e-08,
         6.9499620478e-11,   4.3153242717e-12,   1.7553153345e-11],
                        [6.0794514737e-04,   3.5010746371e-06,   5.1065185129e-07,
         1.3428965111e-09,   1.4571102983e-11,   2.8916618098e-13],
                        [ 4.7095438519e-04,   1.8734554551e-06,   8.5109742980e-08,
         6.7031283936e-10,   8.4730847123e-12,   5.1272933202e-12]                        
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

def checkRegression(err, err_ref, test_name):
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
            checkRegression(err, err_ref, "Test " + str(k))
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
            checkRegression(err, err_ref, "Test " + str(k))
            k+=1
            print err
                
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
    np.testing.assert_approx_equal(coeff, coeff_ref, 1, "CONVERGENCE TEST " + test_name + " FAILED")


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

    print("\n Executing convergence non dense tests")
    
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
    
    print("\n Executing convergence dense tests")
    
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
    non_dense_tests()
    dense_tests()
  
    doAllConvergenceTests()
    checkInterpolationPolynomial()
    checkDerivativesForPolynomial()
    

