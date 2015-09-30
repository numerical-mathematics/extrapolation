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

#finalTime has no relation with dense output times (used for different testing)
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
    denseOutput = [0,np.power(12*[10.],range(0,12))]
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
    denseOutput = [0,np.power(7*[10.],range(1,15,2))]
    return TestProblemDefinition("E5", E5f, 0, np.array([1.76e-3,0,0,0]), 100., denseOutput)

def getAllTests():
    tests = []
#     tests.append(VDPOLProblem())
#     tests.append(VDPOLMildProblem())
    tests.append(VDPOLEasyProblem())
#     tests.append(ROBERProblem())
#     tests.append(OREGOProblem())
#     tests.append(HIRESProblem())
#     tests.append(E5Problem())
#     tests.append(LinearProblem())
    
    return tests

#TO calculate exact solutions switch all tests functions to f(t,y) (instead of f(y,t))
def storeTestsExactSolutions():
    for test in getAllTests():
        #denseOutput = test.denseOutput
        exactSolution, infodict = integrate.odeint(test.RHSFunction,test.initialValue, [test.initialTime, test.finalTime], atol=1e-13, rtol=1e-13, mxstep=100000000, full_output = True)
        print("Store solution for " + test.problemName + "; solution: " + str(exactSolution))
        np.savetxt(getReferenceFile(test.problemName), exactSolution[-1])
                
def getReferenceFile(problemName):
    return "reference_" + problemName + ".txt"
      

def comparisonTest():
    dense=False
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11]
    resultDict={}
    for test in getAllTests():
        testProblemResult = [[],[],[]]
        y_ref = np.loadtxt(getReferenceFile(test.problemName))
        print(test.problemName)
        for i in range(len(tol)):
            print(tol[i])
            functionTuple = (test.RHSFunction, test.initialValue, [test.initialTime, test.finalTime],(),True, tol[i], tol[i],0.5,10e8)
            if(dense):
                functionTuple = (test.RHSFunction, test.initialValue, test.denseOutput,(),True, tol[i], tol[i],0.5,10e8)
            #Explicit parallel current mcr design
            startTime = time.time()
            ys, infodict = ex_parallel.ex_midpoint_explicit_parallel(*functionTuple)
            testProblemResult[0].append([time.time()-startTime, np.linalg.norm((ys[-1]-y_ref)/y_ref), infodict["fe_tot"]])
            print("Done: Explicit parallel current mcr design, abs error: " + str(np.linalg.norm(ys[-1]-y_ref)) + " rel error: " + str(np.linalg.norm((ys[-1]-y_ref)/y_ref))+ " func eval: " + str(infodict["fe_tot"]))
             
            #Explicit parallel old hum design
            startTime = time.time()
            ys, infodict = ex_parallel_original.ex_midpoint_parallel(*functionTuple)
            testProblemResult[1].append([time.time()-startTime, np.linalg.norm((ys[-1]-y_ref)/y_ref), infodict["fe_tot"]])
            print("Done: Explicit parallel old hum design, abs error: " +  str(np.linalg.norm(ys[-1]-y_ref))+ " rel error: " + str(np.linalg.norm((ys[-1]-y_ref)/y_ref))+ " func eval: " + str(infodict["fe_tot"]))
             
            #Implicit parallel
            startTime = time.time()
            ys, infodict = ex_parallel.ex_midpoint_implicit_parallel(*functionTuple)
            testProblemResult[2].append([time.time()-startTime, np.linalg.norm((ys[-1]-y_ref)/y_ref), infodict["fe_tot"]])
            print("Done: Implicit parallel, abs error: " + str(np.linalg.norm(ys[-1]-y_ref)) + " rel error: " + str(np.linalg.norm((ys[-1]-y_ref)/y_ref))+ " func eval: " + str(infodict["fe_tot"]))
            
        resultDict[test.problemName] = testProblemResult
           
    return resultDict

def plotResults(resultDict):
    j=1
    labels=["New Explicit parallel","Old Explicit parallel","Implicit parallel"]
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
    resultDict = comparisonTest()
    plotResults(resultDict)
    print "done"
    
    
