from __future__ import division
from scipy import integrate
import scipy
import numpy as np
from collections import namedtuple
import time
import ex_parallel
import ex_parallel_original
import matplotlib.pyplot as plt
import math
from compare_test import kdv_func, kdv_init, kdv_solout
import fnbruss

"""
From book: Solving Ordinary Differential Equations II,
IV.10 Numerical Experiment, Twelve Test Problems

Each ODE problem is defined with: problemName, right hand side function
(derivative function), jacobian matrix of RHS function, initial time (float),
initial value (np.array), times at which output is wanted, atolfact absolute
tolerance factor-> set to 1. as default (multiplies relative tolerance factor
to make absolute tolerance more stringent), atol absolute tolerance -> set to
None as default (required absolute tolerance for all relative tolerances
wanted).
"""
#TODO: for each problem add plot function to plot results
TestProblemDefinition = namedtuple("TestProblemDefinition", 
            ["problemName","RHSFunction", "RHSGradient","initialTime",
                "initialValue", "denseOutput", "atolfact", "atol"])


#VDPOL problem

#Observation: RHS function can't be nested in VDPOLProblem():
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


#BRUSS-2D problem (Brusselator)
    
A=0
N=10
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
#     denseOutput = [0,0.5,1.,1.3,1.4,5.6,6.,6.1,6.2,10]
    return TestProblemDefinition("BRUSS2D_"+str(N), FortBRUSS2Df, BRUSS2Dgrad, 0, initialValue, denseOutput,1.,None)

#KDV problem

def KDVProblem():
    t0, tf = 0, 0.0003
    denseOutput = [t0,tf]
    y0 = kdv_init(t0)
    return TestProblemDefinition("kdv2", kdv_func, None, t0, y0, denseOutput,1.,None)


def getAllTests():
    '''
    Get all the problem tests that you want to use to test
    (uncomment those that want to be used)
    '''
    tests = []
    tests.append(VDPOLProblem())
#     tests.append(ROBERProblem())
#     tests.append(OREGOProblem())
#     tests.append(HIRESProblem())
#     tests.append(KDVProblem())
#     tests.append(E5Problem())
#     tests.append(BRUSS2DProblem())
    
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
    tol = [1.e-12,1.e-10,1.e-8,1.e-7,1.e-5,1.e-3]
    resultDict={}
    solverFunctions = [
#         ex_parallel.ex_midpoint_implicit_parallel
#         ,
#         ex_parallel.ex_midpoint_semi_implicit_parallel
#         ,
        ex_parallel.ex_euler_semi_implicit_parallel
        ,
        integrate.odeint
        ]
    labelsFunction=[
#         "SemiImp Midpoint"
#         ,
        "Semi Eul"
        ,
        "Scipy int"
        ]

    useGrad = False
    
    for test in getAllTests():
        testProblemResult = []
        for aux in range(0,len(labelsFunction)):
            testProblemResult.append([])
        y_ref = np.loadtxt(getReferenceFile(test.problemName))
        denseOutput = test.denseOutput
        if(not dense):
            y_ref=y_ref[-1]
            denseOutput=[denseOutput[0], denseOutput[-1]]
        print(denseOutput)
        print(test.problemName)
        for i in range(len(tol)):
            print(tol[i])
            labels=[]
            if(test.atol is None):
                atol=test.atolfact*tol[i]
            else:
                atol = test.atol
            rtol=tol[i]
            print("rtol: " + str(rtol) + " atol:" + str(atol))
            
            k=0
            for solverFunction in solverFunctions:
                if solverFunction is integrate.odeint:
                    if(useGrad):
                        grad = test.RHSGradient
                    else:
                        grad = None
                    startTime = time.time()
                    ys, infodict = solverFunction(test.RHSFunction,test.initialValue, denseOutput, Dfun= grad, atol=atol, rtol=rtol, mxstep=100000000, full_output = True)
                    finalTime = time.time()

                    mean_order = 0
                    fe_seq = np.sum(infodict["nfe"])
                    mused = infodict["mused"]
                    print "1: adams (nonstiff), 2: bdf (stiff) -->" + str(mused)
                else:
                    if(useGrad):
                        grad = test.RHSGradient
                    else:
                        grad = None
                    functionTuple = {'func': test.RHSFunction, 'grad': grad, 'y0': test.initialValue, 't': denseOutput
                     ,'full_output': True, 'rtol': rtol, 'atol': atol}
                    startTime = time.time()
                    ys, infodict = solverFunction(**functionTuple)
                    finalTime = time.time()
                  
                    mean_order = infodict["k_avg"]
                    fe_seq = infodict["fe_seq"]


                fe_tot = np.sum(infodict["nfe"])
                nsteps = np.sum(infodict["nst"])
                je_tot = np.sum(infodict["nje"])
                ys=ys[1:len(ys)]
                
                relative_error = np.linalg.norm(ys-y_ref)/np.linalg.norm(y_ref)

                print(relative_error)

                testProblemResult[k].append([finalTime-startTime, relative_error, fe_tot, nsteps, mean_order, fe_seq, je_tot])
                print("Done: " + labelsFunction[k] + " time: " + str(finalTime-startTime) +  " rel error: " + str(relative_error) + " func eval: " + str(fe_tot) + " jac eval: " + str(je_tot) + " func eval seq: " + str(fe_seq)+ " num steps: " + str(nsteps) + " mean_order: " + str(mean_order))
                print("\n")
                
                labels.append(labelsFunction[k]) 
                k+=1
        
        resultDict[test.problemName] = testProblemResult
           
    return resultDict , labels
      

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
            yfunceval=[resRow[2] for resRow in res]
            ytime=[resRow[0] for resRow in res]
            yfuncevalseq=[resRow[5] for resRow in res]
            yjaceval=[resRow[6] for resRow in res]
            x=[resRow[1] for resRow in res]
            
            plt.subplot(411)
            plt.loglog(x,ytime,"s-")
            plt.subplot(412)
            plt.loglog(x,yfunceval,"s-")
            plt.subplot(413)
            plt.loglog(x,yfuncevalseq,"s-")
            plt.subplot(414)
            plt.loglog(x,yjaceval,"s-")

        
        plt.subplot(411)
        plt.legend()
        plt.ylabel("time(log)")
        plt.subplot(412)
        plt.ylabel("fun.ev.")
        plt.subplot(413)
        plt.ylabel("fun.ev.seq")
        plt.subplot(414)
        plt.ylabel("jac.ev.")
        j+=1
#         fig.subplots_adjust(left=0.05, top = 0.96, bottom=0.06, right=0.99)
    plt.show()



if __name__ == "__main__":
    #If exact solution hasn't been yet calculated uncomment first line
#     storeTestsExactSolutions()
    resultDict, labels = comparisonTest()
    plotResults(resultDict, labels)
    print "done"
    
    
