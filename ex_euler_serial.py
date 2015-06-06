
from __future__ import division
import numpy as np
import pylab as pl
import math 

def ex_one_step (f, yn, h, p):
  Y = np.zeros((p+1,p+1), dtype=complex)
  T = np.zeros((p+1,p+1), dtype=complex)
  
  for k in range(1,p+1):
    Y[k,0] = yn
    for j in range(1,k+1):
      Y[k,j] = Y[k,j-1] + (h/k)*f(Y[k,j-1])
    T[k,1] = Y[k,k]

  for k in range(2, p+1):
    for j in range(k, p+1):
      T[j,k] = T[j,k-1] + (T[j,k-1] - T[j-1,k-1])/((j/(j-k+1)) - 1)

  return (T[p,p], T[p-1,p-1])

# solve the first order ODE y' = f(y) by extrapolation based on Euler's method
# on the interval [t0, tf] with the intial value y(t0) = y0 and time step size
# h and order p
def ex_euler_serial (f, y0, t0, tf, h, p):
  ts = np.linspace(t0, tf, int(round((tf - t0) / h)))
  ys = np.zeros(len(ts), dtype=complex)
  ys_hat = np.zeros(len(ts), dtype=complex)
  ys[0] = y0

  for i in range(len(ts) - 1):
    ys[i+1], ys_hat[i+1] = ex_one_step(f, ys[i], h, p)

  return (ts, ys, ys_hat)


# Convergence Test 
# example: y' = f(y) = lam*y on [0, 10] --> exact solution is y = y0*e**(lam*t)

# change pramters to see different cases
lam = -2
y0 = 1
p_max = 5
t0 = 0
tf = 5

f = lambda y: lam*y
exact = lambda t: y0*np.exp(lam*t)

hs = np.asarray([2**(-k) for k in range(1, 10)])
err = np.zeros((p_max, len(hs)))

for p in range(p_max):
  for i in range(len(hs)):
    _, ys, _ = ex_euler_serial(f, y0, t0, tf, hs[i], p+1)
    err[p, i] = abs(ys[-1] - exact(tf))

pl.hold('true')
err_line = []
for p in range(p_max):
  line,  = pl.loglog(hs, err[p], 's-') 
  err_line.append(line)

labels = ['order ' + str(k+1) for k in range(p_max)]
pl.legend(err_line, labels, loc=2)

err_order_p,  = pl.loglog(hs, (hs)*(err[1,5]/hs[5]))
pl.title('Ex-Euler')
pl.ylabel('||error||')
pl.xlabel('time step size')

pl.show()




