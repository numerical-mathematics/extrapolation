
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
  ts = np.arange(t0, tf+h, h)
  if ts[-1] > tf:
    ts = np.delete(ts, -1)
  ys = np.zeros(len(ts), dtype=complex)
  ys_hat = np.zeros(len(ts), dtype=complex)
  ys[0] = y0

  for i in range(len(ts) - 1):
    ys[i+1], ys_hat[i+1] = ex_one_step(f, ys[i], h, p)

  return (ts, ys, ys_hat)


# preforms a convergence test solving the IVP y' = f(y) and y(0) = y0 with
# the [method] provided, where the function exact is the solution the IVP.
# The function plots the ||error|| vs time step size, and a reference line
# of the given order to compare. The graph is titled with graph_title.
def convergence_test(f, exact, t0, tf, y0, method, order, graph_title):
  hs = np.asarray([2**(-k) for k in range(1, 10)])
  err = np.zeros(len(hs))

  for i in range(len(hs)):
    ts, ys, _ = method(f, y0, t0, tf, hs[i], order)
    err[i] = abs(ys[-1] - exact(tf))

  pl.hold('true')
  method_err,  = pl.loglog(hs, err, 's-')
  order_line,  = pl.loglog(hs, (hs**order)*(err[5]/hs[5]))
  pl.legend([method_err, order_line], ['Method Error', 'order ' + str(order)], loc=2)
  pl.title(graph_title)
  pl.ylabel('||error||')
  pl.xlabel('time step size')
  pl.show()


# Convergence Test 

#### test 1 ####
lam = -1j
y0 = 1
f = lambda y: lam*y
exact = lambda t: y0*np.exp(lam*t)

t0 = 0
tf = 10
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 1, "ex_euler_serial")
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 2, "ex_euler_serial")
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 3, "ex_euler_serial")
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 4, "ex_euler_serial")
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 5, "ex_euler_serial")
convergence_test(f, exact, t0, tf, exact(t0), ex_euler_serial, 6, "ex_euler_serial")
