from __future__ import division
from scipy import integrate
import numpy as np
from collections import namedtuple
import time
import ex_parallel
import ex_parallel_original
import matplotlib.pyplot as plt
import math

'''
From book: Solving Ordinary Differential Equations II,
IV.10 Numerical Experiment, Twelve Test Problems
'''

#finalTime has no relation with dense output times 
#(used for testing with stiff problems-> some dense output final time would take really long time)
TestProblemDefinition = namedtuple("TestProblemDefinition", ["problemName","RHSFunction","initialTime","initialValue", "finalTime", "denseOutput"])


#Linear problem

def Linearf(y,t):
#     print("entra \n")
    lam = 3.
    return lam*y

def LinearProblem():    
    return TestProblemDefinition("Linear", Linearf, 0, np.array([1.]), 10., np.arange(0,11,1.))

#VDPOL problem

#OBS: RHS function can't be nested in VDPOLProblem():
#http://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
def VDPOLf(y,t):
    epsilon=1e-6
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLProblem():    
    return TestProblemDefinition("VDPOL", VDPOLf, 0, np.array([2.,0.]),12., np.arange(0,13,1.))

def VDPOLMildf(y,t):
    epsilon=1e-2
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLMildProblem():    
    return TestProblemDefinition("VDPOLMild", VDPOLMildf, 0, np.array([2.,0]),12., np.arange(0,13,1.))

def VDPOLEasyf(y,t):
    epsilon=1
    second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
    return np.array([y[1],second_dim])

def VDPOLEasyProblem():    
    return TestProblemDefinition("VDPOLEasy", VDPOLEasyf, 0, np.array([2.,0]),12., np.arange(0,13,1.))
    
#ROBER problem

def ROBERf(y,t):
    first_dim = -0.04*y[0]+1e4*y[1]*y[2]
    second_dim = 0.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2
    third_dim = 3e7*y[1]**2
    return np.array([first_dim,second_dim,third_dim])
    

def ROBERProblem():
    base=13*[10.]
    base[0]=0
    denseOutput = np.power(base,range(0,13))
    return TestProblemDefinition("ROBER", ROBERf, 0, np.array([1.,0,0]), 100., denseOutput)
    

#OREGO problem

def OREGOf(y,t):
    first_dim = 77.27*(y[1]+y[0]*(1-8.375e-6*y[0]-y[1]))
    second_dim = 1/77.27*(y[2]-(1+y[0])*y[1])
    third_dim = 0.161*(y[0]-y[2])
    return np.array([first_dim,second_dim,third_dim])
    

def OREGOProblem():
    denseOutput = np.arange(0,390,30.)
    return TestProblemDefinition("OREGO", OREGOf, 0, np.array([1.,2.,3.]), 360., denseOutput)

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

def HIRESProblem():
    denseOutput = np.array([0,321.8122,421.8122])
    return TestProblemDefinition("HIRES", HIRESf, 0, np.array([1.,0,0,0,0,0,0,0.0057]),425. ,denseOutput)

#E5 problem

def E5f(y,t):
    A=7.86e-10
    B=1.1e7
    C=1.13e3
    M=1e6
    first_dim = -A*y[0]-B*y[0]*y[2]
    second_dim = A*y[0]-M*C*y[1]*y[2]
    third_dim = A*y[0]-B*y[0]*y[2]-M*C*y[1]*y[2]+C*y[3]
    fourth_dim = B*y[0]*y[2]-C*y[3]
    return np.array([first_dim,second_dim,third_dim,fourth_dim])

def E5Problem():
    base=8*[10.]
    base[0]=0
    exp = range(-1,15,2)
    #OBS: the first exponent doesn't matter (base =0)
    exp[0]=1
    denseOutput = np.power(base,exp)
    return TestProblemDefinition("E5", E5f, 0, np.array([1.76e-3,0,0,0]), 100., denseOutput)

def getAllTests():
    tests = []
    tests.append(VDPOLProblem())
#     tests.append(VDPOLMildProblem())
#     tests.append(VDPOLEasyProblem())
#     tests.append(ROBERProblem())
#     tests.append(OREGOProblem())
#     tests.append(HIRESProblem())
#     tests.append(E5Problem())
#     tests.append(LinearProblem())
    
    return tests

def storeTestsExactSolutions():
    for test in getAllTests():
        denseOutput = test.denseOutput
        exactSolution, infodict = integrate.odeint(test.RHSFunction,test.initialValue, denseOutput, atol=1e-18, rtol=1e-13, mxstep=100000000, full_output = True)
        print("Store solution for " + test.problemName + "; solution: " + str(exactSolution))
        np.savetxt(getReferenceFile(test.problemName), exactSolution[1:len(exactSolution)])
                
def getReferenceFile(problemName):
    return "reference_" + problemName + ".txt"
      

def comparisonTest():
    dense=False
    tol = [1.e-3,1e-5,1e-7,1e-9,1e-11]
    resultDict={}
    solverFunctions = [#ex_parallel.ex_midpoint_explicit_parallel
#          ,ex_parallel_original.ex_midpoint_parallel
            ex_parallel.ex_midpoint_implicit_parallel]
    labelsFunction=[#"New Explicit parallel"
#           ,"Old Explicit parallel"
            "Implicit parallel"]
    robustnesses=[2]#, 3, 5, 10, 100]
    for test in getAllTests():
        testProblemResult = []
        for aux in range(0,len(labelsFunction)*len(robustnesses)):
            testProblemResult.append([])
        y_ref = np.loadtxt(getReferenceFile(test.problemName))
        if(not dense):
            y_ref=y_ref[-1]
        print(test.problemName)
        labels=[]
        for i in range(len(tol)):
            print(tol[i])
            j=0
            for robustness in robustnesses:
                print("robustness " + str(robustness))
                functionTuple = (test.RHSFunction, test.initialValue, test.denseOutput,(),True, tol[i], tol[i],0.5,10e8,robustness)
                if(not dense):
                    denseOutput = test.denseOutput
                    functionTuple = (test.RHSFunction, test.initialValue, [denseOutput[0], denseOutput[-1]],(),True, tol[i], tol[i],0.5,10e8,robustness)
                k=0    
                for solverFunction in solverFunctions:
                    startTime = time.time()
                    ys, infodict = solverFunction(*functionTuple)
                    ys=ys[1:len(ys)]
                    componentwise_relative_error = (ys-y_ref)/y_ref
                    relative_error = [np.linalg.norm(output, 2) for output in componentwise_relative_error]
                    maximum_relative_error = np.max(relative_error)
                    print(relative_error)
                    testProblemResult[j].append([time.time()-startTime, maximum_relative_error, infodict["fe_tot"]])
                    print("Done: " + labelsFunction[k] + ", max rel error: " + str(maximum_relative_error)+ " abs error: " + str(np.linalg.norm(ys-y_ref)) + " rel error: " + str(np.linalg.norm((ys-y_ref)/y_ref))+ " func eval: " + str(infodict["fe_tot"]))
                    labels.append(labelsFunction[k] +" " + str(robustness))
                    k+=1
                    j+=1
   
        resultDict[test.problemName] = testProblemResult
           
    return resultDict , labels

def plotResults(resultDict, labels):
    j=1
    for test in getAllTests():
        testName = test.problemName
        resultTest = resultDict[testName]
        fig= plt.figure(j)
        fig.suptitle(testName)
        plt.subplot(211)
        for i in range(0,len(resultTest)):
            res=resultTest[i]
            yfunceval=[resRow[2] for resRow in res]
            ytime=[resRow[0] for resRow in res]
            x=[resRow[1] for resRow in res]
            plt.subplot(211)
            plt.plot([math.log10(xval) for xval in x],[math.log10(yval) for yval in ytime],label=labels[i], marker="o")
            plt.subplot(212)
            plt.plot([math.log10(xval) for xval in x],yfunceval,label=labels[i], marker="o")
        
        plt.subplot(211)
        plt.legend()
        plt.ylabel("time (log10)")
        plt.xlabel("relative error (log10)")
        plt.subplot(212)
        plt.legend()
        plt.ylabel("function evaluations")
        plt.xlabel("relative error (log10)")
        j+=1
    plt.show()

if __name__ == "__main__":
#     storeTestsExactSolutions()
    resultDict, labels = comparisonTest()
    plotResults(resultDict, labels)
    print "done"
    
    
