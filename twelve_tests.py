from __future__ import division

'''
From book: Solving Ordinary Differential Equations II,
IV.10 Numerical Experiment, Twelve Test Problems
'''

TestProblemDefinition = namedtuple("TestProblemDefinition", ["problemName","RHSFunction","initialTime","initialValue","denseOutput"])


#VDPOL problem

def VPOLProblem(epsilon):
    def f(y,t):
        second_dim=1/epsilon*(((1-y[0]**2)*y[1])-y[0])
        return np.array([y[1],second_dim])
    
    return TestProblemDefinition("VDPOL", f, 0, np.array([2,0]), range(1,12))

    
#ROBER problem

def ROBERProblem():
    def f(y,t):
        first_dim = -0.04*y[0]+1e4*y[1]*y[2]
        second_dim = 0.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2
        third_dim = 3e7*y[1]**2
        return np.array([first_dim,second_dim,third_dim])
    
    denseOutput = np.power(12*[10],range(0,12))
    return TestProblemDefinition("ROBER", f, 0, np.array([1,0,0]), denseOutput)
    

#OREGO problem

def OREGOProblem():
    def f(y,t):
        first_dim = 77.27*(y[1]+y[0]*(1-8.375e-6*y[0]-y[1]))
        second_dim = 1/77.27*(y[2]-(1+y[0])*y[1])
        third_dim = 0.161*(y[0]-y[2])
        return np.array([first_dim,second_dim,third_dim])
    
    denseOutput = range(30,390,30)
    return TestProblemDefinition("OREGO", f, 0, np.array([1,2,3]), denseOutput)

#HIRES problem

def HIRESProblem():
    def f(y,t):
        first_dim = -1.71*y[0]+0.43*y[1]+8.32*y[2]+0.0007
        second_dim = 1.71*y[0]-8.75*y[1]
        third_dim = -10.03*y[2]+0.43*y[3]+0.035*y[4]
        fourth_dim = 8.32*y[1]+1.71*y[2]-1.12*y[3]
        fifth_dim = -1.745*y[4]+0.43*y[5]+0.43*y[6]
        sixth_dim = -280*y[5]*y[7]+0.69*y[3]+1.71*y[4]-0.43*y[5]+0.69*y[6]
        seventh_dim = 280*y[5]*y[7]-1.81*y[6]
        eighth_dim = -seventh_dim
        return np.array([first_dim,second_dim,third_dim,fourth_dim,fifth_dim,sixth_dim,seventh_dim,eighth_dim])
    
    denseOutput = np.array([321.8122,421.8122])
    return TestProblemDefinition("HIRES", f, 0, np.array([1,0,0,0,0,0,0,0.0057]), denseOutput)

#E5 problem

def E5Problem():
    def f(y,t):
        A=7.86e-10
        B=1.1e7
        C=1.13e3
        M=1e6
        first_dim = -A*y[0]-B*y[0]*y[2]
        second_dim = A*y[0]-M*C*y[1]*y[2]
        third_dim = A*y[0]-B*y[0]*y[2]-M*C*y[1]*y[2]+C*y[3]
        fourth_dim = B*y[0]*y[2]-C*y[3]
        return np.array([first_dim,second_dim,third_dim,fourth_dim])
    
    denseOutput = np.power(12*[10],range(1,15,2))
    return TestProblemDefinition("E5", f, 0, np.array([1.76e-3,0,0,0]), denseOutput)
