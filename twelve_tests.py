from __future__ import division
from scipy import integrate
import scipy
import numpy as np
from collections import namedtuple, Counter
import time
import ex_parallel
import ex_parallel_original
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from compare_test import kdv_func, kdv_init
from numpy import meshgrid
import fnbruss
import yappi


'''
From book: Solving Ordinary Differential Equations II,
IV.10 Numerical Experiment, Twelve Test Problems
'''

'''
Each ODE problem is defined with: problemName, right hand side function (derivative function), jacobian matrix of RHS function, initial time (float), initial value (np.array),
times at which output is wanted, atolfact absolute tolerance factor-> set to 1. as default (multiplies relative tolerance factor to make absolute tolerance more stringent), 
atol absolute tolerance -> set to None as default (required absolute tolerance for all relative tolerances wanted).
'''
#TODO: for each problem add plot function to plot results
TestProblemDefinition = namedtuple("TestProblemDefinition", ["problemName","RHSFunction", "RHSGradient","initialTime","initialValue", "denseOutput", "atolfact", "atol"])


#Linear problem

def Linearf(y,t):
    lam = -1.
    return lam*y

def LinearProblem():
    denseOutput = np.array([0,0.1,0.2,0.3,0.4,0.5,0.536498185, 0.6,0.7,0.8,0.9,1.])    
    return TestProblemDefinition("Linear", Linearf, None, 0, np.array([1.]),denseOutput ,1.,None)

#VDPOL problem

#OBS: RHS function can't be nested in VDPOLProblem():
#http://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

def VDPOLf(y,t):
    epsilon=1e-6
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLgrad(y,t):
    epsilon=1e-6
    matrix12=1/epsilon*(-2*y[0]*y[1]-1)
    matrix22=1/epsilon*(1-y[0]**2)
    return np.array([[0,1],[matrix12,matrix22]])

def VDPOLProblem():    
    return TestProblemDefinition("VDPOL", VDPOLf, VDPOLgrad, 0, np.array([2.,0.]),np.arange(0,13,1.),1.,None)


def VDPOLMildf(y,t):
    epsilon=1e-2
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLMildProblem():    
    return TestProblemDefinition("VDPOLMild", VDPOLMildf, None, 0, np.array([2.,0]),np.arange(0,13,1.),1.,None)

def VDPOLEasyf(y,t):
    epsilon=1
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLEasyProblem():
    denseOutput = np.array([0.,1.,2.,3.,4.,5.,6.,6.4456913297,7.,8.,9.,10.,11.,12.])    
    return TestProblemDefinition("VDPOLEasy", VDPOLEasyf, None, 0, np.array([2.,0]),denseOutput,1.,None)
    
#ROBER problem

def ROBERf(y,t):
    first_dim = -0.04*y[0]+1e4*y[1]*y[2]
    second_dim = 0.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2
    third_dim = 3e7*y[1]**2
    return np.array([first_dim,second_dim,third_dim])

def ROBERgrad(y,t):
    matrix11=-0.04
    matrix12=1e4*y[2]
    matrix13=1e4*y[1]
    matrix21=0.04
    matrix22=-1e4*y[2]-3e7*y[1]*2
    matrix23=-1e4*y[1]
    matrix31=0
    matrix32=3e7*y[1]*2
    matrix33=0
    return np.array([[matrix11,matrix12,matrix13],[matrix21,matrix22,matrix23],[matrix31,matrix32,matrix33]])

def ROBERProblem():
    base=13*[10.]
    base[0]=0
    denseOutput = np.power(base,range(0,13))
    return TestProblemDefinition("ROBER", ROBERf, ROBERgrad, 0, np.array([1.,0,0]), denseOutput,1.e-6,None)
    

#OREGO problem

def OREGOf(y,t):
    first_dim = 77.27*(y[1]+y[0]*(1-8.375e-6*y[0]-y[1]))
    second_dim = 1/77.27*(y[2]-(1+y[0])*y[1])
    third_dim = 0.161*(y[0]-y[2])
    return np.array([first_dim,second_dim,third_dim])
    
def OREGOgrad(y,t):
    matrix11=77.27*(1-8.375e-6*y[0]*2-y[1])
    matrix12=77.27*(1-y[0])
    matrix13=0
    matrix21=1/77.27*(-y[1])
    matrix22=1/77.27*(-(1+y[0]))
    matrix23=1/77.27
    matrix31=0.161
    matrix32=0
    matrix33=-0.161
    return np.array([[matrix11,matrix12,matrix13],[matrix21,matrix22,matrix23],[matrix31,matrix32,matrix33]])

def OREGOProblem():
    denseOutput = np.arange(0,390,30.)
    return TestProblemDefinition("OREGO", OREGOf, OREGOgrad, 0, np.array([1.,2.,3.]), denseOutput, 1.e-6,None)

#HIRES problem

def HIRESf(y,t):
    first_dim = -1.71*y[0]+0.43*y[1]+8.32*y[2]+0.0007
    second_dim = 1.71*y[0]-8.75*y[1]
    third_dim = -10.03*y[2]+0.43*y[3]+0.035*y[4]
    fourth_dim = 8.32*y[1]+1.71*y[2]-1.12*y[3]
    fifth_dim = -1.745*y[4]+0.43*y[5]+0.43*y[6]
    sixth_dim = -280*y[5]*y[7]+0.69*y[3]+1.71*y[4]-0.43*y[5]+0.69*y[6]
    seventh_dim = 280*y[5]*y[7]-1.81*y[6]
    eighth_dim = -seventh_dim
    return np.array([first_dim,second_dim,third_dim,fourth_dim,fifth_dim,sixth_dim,seventh_dim,eighth_dim])

def HIRESgrad(y,t):
    return np.array([[-1.71,0.43,8.32,0,0,0,0,0],
                     [1.71,-8.75,0,0,0,0,0,0],
                     [0,0,-10.03,0.43,0.035,0,0,0],
                     [0,8.32,1.71,-1.12,0,0,0,0],
                     [0,0,0,0,-1.745,0.43,0.035,0],
                     [0,0,0,0.69,1.71,-0.43-280*y[7],0.69,-280*y[5]],
                     [0,0,0,0,0,280*y[7],-1.81,280*y[5]],
                     [0,0,0,0,0,-280*y[7],1.81,-280*y[5]]])

def HIRESProblem():
    denseOutput = np.array([0,321.8122,421.8122])
    return TestProblemDefinition("HIRES", HIRESf, HIRESgrad, 0, np.array([1.,0,0,0,0,0,0,0.0057]),denseOutput,1.e-4,None)

#E5 problem

def E5grad(y,t):
    A=7.86e-10
    B=1.1e7
    C=1.13e3
    M=1e6
    return np.array([[-A-B*y[2],0,-B*y[0],0],
                     [A,-M*C*y[2],-M*C*y[1],0],
                     [A-B*y[2],-M*C*y[2],-M*C*y[1]-B*y[0],C],
                     [B*y[2],0,B*y[0],-C]])

def E5f(y,t):
    A=7.86e-10
    B=1.1e7
    C=1.13e3
    M=1e6
    first_dim = -A*y[0]-B*y[0]*y[2]
    second_dim = A*y[0]-M*C*y[1]*y[2]
    fourth_dim = B*y[0]*y[2]-C*y[3]
    third_dim = second_dim-fourth_dim
    
    return np.array([first_dim,second_dim,third_dim,fourth_dim])

def E5Plot(ys, times):
    y1=[yt[0] for yt in ys]
    y2=[yt[1] for yt in ys]
    y3=[yt[2] for yt in ys]
    y4=[yt[3] for yt in ys]
    n=len(times)
    plt.plot(np.log10(times[1:n]),np.log10(y1[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y2[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y3[1:n]))
    plt.plot(np.log10(times[1:n]),np.log10(y4[1:n]))
    
    plt.show()

def E5Problem():
    base=8*[10.]
    base[0]=0
    exp = np.arange(-1,14,2)
    #OBS: the first exponent doesn't matter (base =0)
    exp[0]=1
    denseOutput = np.power(base,exp)
    return TestProblemDefinition("E5", E5f, E5grad, 0, np.array([1.76e-3,0,0,0]), denseOutput,1.,1.7e-24)


#BRUSS-2D problem
    
A=0
N=20
step=0
x=0
y=0
alpha=0.1

def initializeBRUSS2DValues(Nval):
    global A,Aperm,N,step,x,y
    N=Nval
    A=five_pt_laplacian_sparse_periodic(N,0,1)
    step=1/(N-1)
    x=np.multiply(step,range(N)*N)
    y=np.multiply(step,np.repeat(range(N),N))
    

def five_pt_laplacian_sparse_periodic(m,a,b):
    """Construct a sparse matrix that applies the 5-point laplacian discretization
       with periodic BCs on all sides."""
    e=np.ones(m**2)
    e2=([1]*(m-1)+[0])*m
    e3=([0]+[1]*(m-1))*m
    h=(b-a)/(m-1)
    A=scipy.sparse.spdiags([-4*e,e2,e3,e,e],[0,-1,1,-m,m],m**2,m**2)
    # Top & bottom BCs:
    A_periodic_top = scipy.sparse.spdiags([e[0:m]],[2*m-m**2],m**2,m**2).transpose()
    A_periodic_bottom = scipy.sparse.spdiags(np.concatenate((np.zeros(m),e[0:m])),[2*m-m**2],m**2,m**2)
    A_periodic = A_periodic_top + A_periodic_bottom
    # Left & right BCs:
    for i in range(m):
        A_periodic[i*m,(i+1)*m-2] = 1.
        A_periodic[(i+1)*m-1,i*m+1] = 1.
    A = A + A_periodic
    A/=h**2
    A = A.tocsr()
    return A

#Here we will use U to obtain the coordinates (x,y)
def BRUSS2DInhom(t):
   Nsq=N**2
   fout = np.zeros(Nsq)
   if t<1.1:
       return fout
   fout = np.add(np.power(x-0.3,2),np.power(y-0.6,2))<=0.01
   fout = 5*fout
   return fout

def BRUSS2Dgradnonsparse(yn,tn):
    U=yn[0:N**2]
    V=yn[N**2:2*N**2]
    df1du = scipy.sparse.spdiags(2*U*V-4.4,0,N**2,N**2)+alpha*A
    df1dv = scipy.sparse.spdiags(U**2,0,N**2,N**2)
    df2du = scipy.sparse.spdiags(3.4-2*U*V,0,N**2,N**2)
    df2dv = scipy.sparse.spdiags(-U**2,0,N**2,N**2)+alpha*A
    left  = scipy.sparse.vstack([df1du,df2du])
    right = scipy.sparse.vstack([df1dv,df2dv])
    final = scipy.sparse.hstack([left, right]).todense()
    return final

def BRUSS2Dgrad(yn,tn):
    U=yn[0:N**2]
    V=yn[N**2:2*N**2]
    df1du = scipy.sparse.spdiags(2*U*V-4.4,0,N**2,N**2)+alpha*A
    df1dv = scipy.sparse.spdiags(U**2,0,N**2,N**2)
    df2du = scipy.sparse.spdiags(3.4-2*U*V,0,N**2,N**2)
    df2dv = scipy.sparse.spdiags(-U**2,0,N**2,N**2)+alpha*A
    left = scipy.sparse.vstack([df1du,df2du])
    right = scipy.sparse.vstack([df1dv,df2dv])
    final = scipy.sparse.hstack([left, right], format='csr')
    return final

def FortBRUSS2Df(y,t):
    '''
    Compiled Fortran brusselator 2D RHS function (faster than python)
    '''
    aux=fnbruss.fnbruss(y,t,N)
    return aux

def PyBRUSS2Df(y,t):
    RHS = np.zeros(2*N**2)
    U=y[0:N**2]
    V=y[N**2:2*N**2]
    RHS[0:N**2] = 1+U**2*V-4.4*U+alpha*A*U+BRUSS2DInhom(t)
    RHS[N**2:2*N**2] = 3.4*U-U**2*V+alpha*A*V
    
    return RHS

def BRUSS2DInitialValue(N):
    initialValue = np.zeros(2*N**2)
    initialValue[0:N**2] = np.multiply(22,np.multiply(y,np.power(1-y,3/2)))
    initialValue[N**2:2*N**2] = np.multiply(27,np.multiply(x,np.power(1-x,3/2)))
    
    return initialValue

def BRUSS2DPlot(ys, times):
    X, Y = np.meshgrid(np.multiply(step,range(N)),np.multiply(step,range(N)))
    for i in range(len(ys)):
        z=ys[i]
        U=np.reshape(z[range(N**2)], (N,N))
        V=np.reshape(z[range(N**2,2*N**2)], (N,N))
        fig = plt.figure()
        fig.suptitle("time : " + str(times[i]))
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(X, Y, U)
        ax.plot_wireframe(X, Y, V, color='r')
    plt.show()

def BRUSS2DProblem():
    initializeBRUSS2DValues(N)
    tf=11.5
    denseOutput = [0,1.5,tf]
    initialValue = BRUSS2DInitialValue(N)
    denseOutput = [0,0.5,1.,1.3,1.4,5.6,6.,6.1,6.2,10]
    return TestProblemDefinition("BRUSS2D_"+str(N), FortBRUSS2Df, BRUSS2Dgrad, 0, initialValue, denseOutput,1.,None)

#KDV problem

def KDVProblem():
    t0, tf = 0, 0.003
    denseOutput = [t0,tf]
    y0 = kdv_init(t0)
    return TestProblemDefinition("kdv", kdv_func, None, t0, y0, denseOutput,1.,None)


def getAllTests():
    '''
    Get all the problem tests that you want to use to test
    (uncomment those that want to be used)
    '''
    tests = []
#     tests.append(VDPOLProblem())
#     tests.append(VDPOLMildProblem())
#     tests.append(VDPOLEasyProblem())
#     tests.append(ROBERProblem())
#     tests.append(OREGOProblem())
#     tests.append(HIRESProblem())
#     tests.append(KDVProblem())
#     tests.append(E5Problem())
    tests.append(BRUSS2DProblem())
#     tests.append(LinearProblem())
    
    return tests

def storeTestsExactSolutions():
    '''
    Stores an exact solution (asking for a very stringent tolerance to a numerical method)
    '''
    for test in getAllTests():
        denseOutput = test.denseOutput
        startTime = time.time()
        exactSolution, infodict = integrate.odeint(test.RHSFunction,test.initialValue, denseOutput, Dfun=None, atol=1e-27, rtol=1e-13, mxstep=100000000, full_output = True)
        print("Store solution for " + test.problemName + "; solution: " + str(exactSolution))
        print("Time : " + str(time.time()-startTime) + " numb steps: " + str(infodict["nst"]))
        np.savetxt(getReferenceFile(test.problemName), exactSolution[1:len(exactSolution)])
        #Use a plot function to visualize results: like BRUSS2DPlot()
            
                
def getReferenceFile(problemName):
    '''
    Get the reference file name for a given problemName (keeps stored solutions tidy)
    '''
    return "reference_" + problemName + ".txt"
      

def inputTuple(k,denseOutput,test,rtol,atol,firstStep,robustness,smoothing,seq,useOptimal):    
    '''
    Gets the input tuple to pass to the solver with all the parameters passed to this function.
    
    If useOptimal is True then the parameters smoothing and seq are ignored and the returned tuple
    sets its values to the optimal values (obtained through exhaustive comparison on multiple
    test problemS)
    '''
    standardTuple = {'func': test.RHSFunction, 'grad': test.RHSGradient, 'y0': test.initialValue, 't': denseOutput
                     ,'full_output': True, 'rtol': rtol, 'atol': atol, 'h0': firstStep, 'mxstep': 10e8, 'robustness': robustness,
                     'smoothing': smoothing,'seq':seq}#, 'nworkers': 1}    
    
    standardOldTuple = {'func': test.RHSFunction, 'y0': test.initialValue, 't': denseOutput
                     ,'full_output': True, 'rtol': rtol, 'atol': atol, 'h0': firstStep, 'mxstep': 10e8}
    
    if(useOptimal):
        midimplicitTuple = standardTuple.copy()
        midimplicitTuple.update({'smoothing': 'gbs','seq':(lambda t: 2*(2*t-1))})
        midsemiimplicitTuple = standardTuple.copy()
        midsemiimplicitTuple.update({'smoothing': 'semiimp' ,'seq':(lambda t: 2*(2*t-1))})
        eulersemiimplicitTuple = standardTuple.copy()
        eulersemiimplicitTuple.update({'smoothing': 'no','seq':(lambda t: 2*(2*t-1))})
#         eulersemiimplicitTuple.update({'smoothing': 'no','seq':seq})

        optimalTuples =[
#             standardTuple
#             ,
#             standardOldTuple
#             ,
#             midimplicitTuple
#             ,
#             midsemiimplicitTuple
#             ,
            eulersemiimplicitTuple
            ,
            standardTuple
        ]
        return optimalTuples[k]
    return standardTuple

def comparisonTest():
    '''
    Mainly: loops over all the tolerances in tol to obtain a comparison plot of the behavior of all the
    algorithms in solverFunctions (in relation to their names, labelsFunction).
    
    It also iterates onto all possible configuration parameters to get different algorithm/parameters
    combinations to plot (to compare how different algorithms behave with different parameters configurations)
    
    Obs: if useOptimal is True, the seq and smoothing parameters are set to the optimal values
    (see inputTuple(...))
    '''
    dense=True
    tol = [1.e-4]#,1.e-5,1.e-7,1.e-8]#,1.e-10,1.e-12]#,1.e-13,1.e-15]
    resultDict={}
    useOptimal = True
    solverFunctions = [
#         ex_parallel.ex_midpoint_explicit_parallel
#         ,
#         ex_parallel_original.ex_midpoint_parallel
#         ,
#         ex_parallel.ex_midpoint_implicit_parallel
#         ,
#         ex_parallel.ex_midpoint_semi_implicit_parallel
#         ,
        ex_parallel.ex_euler_semi_implicit_parallel
#         ,
#         integrate.odeint
        ]
    labelsFunction=[
#         "New Explicit parl"
#         ,
#         "Old Explicit parl"
#         ,
#         "Implicit"
#         ,
#         "SemiImp Midpoint"
#         ,
        "Semi Eul"
#         ,
#         "Scipy int"
        ]

    robustnesses=[3]#, 3, 5, 10, 100]
    smoothings = ['no']#,'semiimp']#,'gbs']
    useGrads = [False]#, False]

    def BD1983(t):
        #First value of the sequence not used
        seq=[0,2,6,10,14,22,34,50,70,98,138,194]
        return seq[t]
    
    seqs = {'2(2t-1)':(lambda t: 2*(2*t-1))}#,'t+1':(lambda t: t+1)}#,'B&D1983':BD1983,'None':None}#, }
    firstStep=0.0005
    for test in getAllTests():
        testProblemResult = []
        for aux in range(0,len(labelsFunction)*len(robustnesses)*len(smoothings)*len(useGrads)*len(seqs)):
            testProblemResult.append([])
        y_ref = np.loadtxt(getReferenceFile(test.problemName))
        denseOutput = test.denseOutput
        if(not dense):
            y_ref=y_ref[-1]
            denseOutput=[denseOutput[0], denseOutput[-1]]
        print(denseOutput)
        print(test.problemName)
        for i in range(len(tol)):
            first =True
            print(tol[i])
            j=0
            labels=[]
            if(test.atol is None):
                atol=test.atolfact*tol[i]
            else:
                atol = test.atol
            rtol=tol[i]
            print("rtol: " + str(rtol) + " atol:" + str(atol))
            for seqStr in seqs:
                seq = seqs[seqStr]
#                 print("sequence " + seqStr)
                for useGrad in useGrads:
                    print("gradient " + str(useGrad))
                    for smoothing in smoothings:
#                         print("smoothing " + str(smoothing))
                        for robustness in robustnesses:
#                             print("robustness " + str(robustness))
                            k=0    
                            for solverFunction in solverFunctions:
                                startTime = time.time()
                                if solverFunction is integrate.odeint:
                                    if(useGrad):
                                        grad = test.RHSGradient
                                    else:
                                        grad = None

                                    ys, infodict = solverFunction(test.RHSFunction,test.initialValue, denseOutput, Dfun= grad, atol=atol, rtol=rtol, mxstep=100000000, full_output = True)
                                    mean_order = 0
                                    fe_seq = np.sum(infodict["nfe"])
                                    mused = infodict["mused"]
#                                     print "1: adams (nonstiff), 2: bdf (stiff) -->" + str(mused)
                                else:
#                                     print("initial guess " + str(first))
#                                     print("iterative "+ str(first))
#                                     print("freeze jac " + str(first))
                                    ex_parallel.setusegradient(useGrad)
                                    aaa=(True and first)
                                    ex_parallel.setfrezeejacobian(True)
                                    ex_parallel.setwork(True)
                                    ex_parallel.setaddinitialguess(False)
                                    ex_parallel.setiterative(True)
                                    if(first):
                                        first=False
                                    functionTuple=inputTuple(k,denseOutput, test,rtol,atol,firstStep,robustness,smoothing,seq,useOptimal)
#                                     yappi.start()
                                    ys, infodict = solverFunction(**functionTuple)
                                #Code to see bottleneck in the code
#                                     yappi.get_func_stats().print_all()
#                                     yappi.clear_stats()                                    
                                    mean_order = infodict["k_avg"]
                                    fe_seq = infodict["fe_seq"]

        
                                finalTime = time.time()
                                fe_tot = np.sum(infodict["nfe"])
                                nsteps = np.sum(infodict["nst"])
                                je_tot = np.sum(infodict["nje"])
                                ys=ys[1:len(ys)]
                                componentwise_relative_error = (ys-y_ref)/y_ref
                                relative_error = [np.linalg.norm(output, 2) for output in componentwise_relative_error]
                                print(relative_error)
                                maximum_relative_error = np.linalg.norm(relative_error)
                                testProblemResult[j].append([finalTime-startTime, maximum_relative_error, fe_tot, nsteps, mean_order, fe_seq, je_tot])
                                print("Done: " + labelsFunction[k] + " time: " + str(finalTime-startTime) +  " rel error: " + str(maximum_relative_error) + " func eval: " + str(fe_tot) + " jac eval: " + str(je_tot) + " func eval seq: " + str(fe_seq)+ " num steps: " + str(nsteps) + " mean_order: " + str(mean_order))
                                print("\n")
#                                 labels.append(labelsFunction[k] +", rob=" + str(robustness) + ", smooth=" + str(smoothing) + ", usegrad=" + str(useGrad) + " , seq = " + seqStr)
                                if(not useOptimal):
                                    labels.append(labelsFunction[k] + ", smooth=" + str(smoothing) + " , seq = " + seqStr)
                                else:
                                    labels.append(labelsFunction[k] + " optim, freeze=" + str(aaa) ) # + ", seq = " + str(seqStr))#", it= " + str(aaa))
                                k+=1
                                j+=1
        
#         getComparisonFactorMetric(testProblemResult, labels, labelsFunction)
        resultDict[test.problemName] = testProblemResult
    plt.show()
           
    return resultDict , labels

def ploteigenvalues(allvaps, testName, tol):
    allvapsreal = [k.real for k in allvaps]
    allvapsim = [k.imag for k in allvaps]
    fig = plt.figure()
    fig.suptitle(testName + " " + str(tol))
    plt.scatter(allvapsreal, allvapsim)
    plt.axvline()
    plt.axhline()

def getReferenceIndex(labels, labelsFunction):
    '''
    Gets the index where in all the algorithm/parameter combinations
    the reference algorithm is
    '''
    j=0
    for label in labels:
        if label.startswith(labelsFunction[-1]):
            return j
        j += 1
            
def getComparisonFactorMetric(testProblemResult, labels, labelsFunction):
    '''
    Gets an extra metric from comparing the last algorithm in the list of algorithms
    labeled in parameter labelsFunction (ideally the scipy algorithm) with all other algorithms,
    calculating the ratio of function evaluations done of the other algorithms with our 
    reference (scipy algorithm).
    '''
    reference = testProblemResult[getReferenceIndex(labels, labelsFunction)]
    xreference = [resRow[1] for resRow in reference]
    yreference=[resRow[2] for resRow in reference]
    
    for i in range(0,len(testProblemResult)):
        res=testProblemResult[i]
        yfunceval=[resRow[2] for resRow in res]
        x=[resRow[1] for resRow in res]
        k=0
        for xval in x:
            yval = yfunceval[k]
            yratio=None
            j=0
            for xvalref in xreference:
                if(xval>=xvalref):
                    if(j==0):
                        yratio=linearExtrapolate(xreference[j+1], yreference[j+1], xvalref, yreference[j], xval, yval)
                        break
                    else:
                        yratio=linearExtrapolate(xvalref, yreference[j], xreference[j-1], yreference[j-1], xval, yval)
                        break 
                    
                j+=1
            if(yratio is None):
                yratio=linearExtrapolate(xreference[j-1], yreference[j-1], xreference[j-2], yreference[j-2], xval, yval)
                
            res[k].append(yratio)
            k+=1

def linearExtrapolate(x1,y1,x2,y2,xextr,ycompare):
    '''
    Linearly extrapolate values using the two (x,y) points and return the ratio of the 
    extrapolated value with the ycompare value
    '''
    yval=y1+(y2-y1)/(x2-x1)*(xextr-x1)
    return ycompare/yval[0]

def plotResults(resultDict, labels):
    '''
    Plot all the results in resultDict. ResultDicts should contain for each test problem a 
    list with all the results for that problem (each problem is plotted in separated windows).
    
    Each problem entry should contain a list with all the different algorithm/parameters combinations
    to be plotted. The labels parameter should contain the title to be shown as legend for each of this
    algorithm/parameters combination.
    
    Each algorithm/parameters combination is a list with all the indicator y-axis values (number 
    of function evaluations, number of steps, mean order, time...) to be plotted (included 
    the x-axis value: relative error). 
    '''
    j=1
    for test in getAllTests():
        testName = test.problemName
        resultTest = resultDict[testName]
        fig= plt.figure(j)
        fig.suptitle(testName)
        for i in range(0,len(resultTest)):
            res=resultTest[i]
            ynumsteps=[resRow[3] for resRow in res]
            yfunceval=[resRow[2] for resRow in res]
            ytime=[resRow[0] for resRow in res]
            ymeanord=[resRow[4] for resRow in res]
#             yratio=[resRow[7] for resRow in res]
            yfuncevalseq=[resRow[5] for resRow in res]
            yjaceval=[resRow[6] for resRow in res]
            x=[resRow[1] for resRow in res]
            plt.subplot(711)
            plt.plot([math.log10(xval) for xval in x],[math.log10(yval) for yval in ytime],label=labels[i], marker="o")
            plt.subplot(712)
            plt.plot([math.log10(xval) for xval in x],yfunceval,label=labels[i], marker="o")
            plt.subplot(714)
            plt.plot([math.log10(xval) for xval in x],ymeanord,label=labels[i], marker="o")
            plt.subplot(717)
            plt.plot([math.log10(xval) for xval in x],ynumsteps,label=labels[i], marker="o")
            plt.subplot(713)
#             plt.plot([math.log10(xval) for xval in x],yratio,label=labels[i], marker="o")
            plt.subplot(715)
            plt.plot([math.log10(xval) for xval in x],yfuncevalseq,label=labels[i], marker="o")
            plt.subplot(716)
            plt.plot([math.log10(xval) for xval in x],yjaceval,label=labels[i], marker="o")
        
        plt.subplot(711)
        plt.legend()
        plt.ylabel("time(log)")
        plt.subplot(712)
        plt.ylabel("fun.ev.")
        plt.subplot(714)
        plt.ylabel("mean order")
        plt.subplot(717)
        plt.ylabel("#steps")
        plt.xlabel("relative error (log10)")
        plt.subplot(713)
        plt.ylabel("fun.ev. ratio")
        plt.subplot(715)
        plt.ylabel("fun.ev.seq")
        plt.subplot(716)
        plt.ylabel("jac.ev.")
        j+=1
        fig.subplots_adjust(left=0.05, top = 0.96, bottom=0.06, right=0.99)
    plt.show()



if __name__ == "__main__":
    #If exact solution hasn't been yet calculated uncomment first line
#     storeTestsExactSolutions()
    resultDict, labels = comparisonTest()
#     plotResults(resultDict, labels)
    print "done"
    
    
