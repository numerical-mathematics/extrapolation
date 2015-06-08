# basic implementation of explicit extrapolation with Euler's and midpoint methods
# a convergence test on the example y' = lam*y is preformed at the end of this file

from __future__ import division
import numpy as np
import pylab as pl
import math 

# solve the first order ODE y' = f(y, t) on the interval [t0, tf] with 
# the initial value y(t0) = y0 and time step size h
def euler_method (f, y0, t0, tf, h):
  ts = np.arange(t0, tf+h, h)
  if ts[-1] > tf:
    ts = np.delete(ts, -1)
  ys = np.zeros(len(ts), dtype=complex)
  ys[0] = y0
  for i in range(len(ts) - 1):
    ys[i+1] = ys[i] + h*f(ys[i], ts[i])
  return (ts, ys)

# solve the first order ODE y' = f(y, t) on the interval [t0, tf] with 
# the intial value y(t0) = y0  and time step size h
def midpoint_method (f, y0, t0, tf, h):
  ts = np.arange(t0, tf+h, h)
  if ts[-1] > tf:
    ts = np.delete(ts, -1)

  ys = np.zeros(len(ts), dtype=complex)
  ys[0] = y0
  ys[1] = ys[0] + h*f(ys[0], ts[0]) # use Euler to get second initial value
  for i in range(len(ts) - 2):
    ys[i+2] = ys[i] + 2*h*f(ys[i+1], ts[i+1])
  return (ts, ys)

# preforms a convergence test solving the IVP y' = f(y) and y(0) = y0 with
# the [method] provided, where the function exact is the solution the IVP.
# The function plots the ||error|| vs time step size, and a reference line
# of the given order to compare. The graph is titled with graph_title
def convergence_test(f, exact, t0, tf, y0, method, order, graph_title):
  hs = np.asarray([2**(-k) for k in range(1, 14)])
  err = np.zeros(len(hs))

  for i in range(len(hs)):
    ts, ys = method(f, y0, t0, tf, hs[i])
    err[i] = abs(ys[-1] - exact(tf))

  pl.hold('true')
  method_err,  = pl.loglog(hs, err, 's-')
  order_line,  = pl.loglog(hs, (hs**order)*(err[5]/hs[5]))
  pl.legend([method_err, order_line], ['Method Error', 'order ' + str(order)], loc=2)
  pl.title(graph_title)
  pl.ylabel('||error||')
  pl.xlabel('time step size')

  pl.show()

# Convergence Tests 

#### test 1 ####
lam = -2
f = lambda y,t: lam*abs(y)
y0 = 1
exact = lambda t: y0*np.exp(lam*t)
t0 = 0
tf = 10
convergence_test(f, exact, t0, tf, exact(t0), euler_method, 1, "euler_method")
convergence_test(f, exact, t0, tf, exact(t0), midpoint_method, 2, "midpoint_method")

# #### test 2 ####
f = lambda y,t: 4.*y*float(np.sin(t))**3*np.cos(t)
y0 = 1
exact = lambda t: y0*np.exp((np.sin(t))**4)
t0 = 0
tf = 5
convergence_test(f, exact, t0, tf, exact(t0), euler_method, 1, "euler_method")
convergence_test(f, exact, t0, tf, exact(t0), midpoint_method, 2, "midpoint_method")

# #### test 3 ####
f = lambda y,t: 4.*t*np.sqrt(y)
y0 = 1
exact = lambda t: (1.+t**2)**2
t0 = 0
tf = 5
convergence_test(f, exact, t0, tf, exact(t0), euler_method, 1, "euler_method")
convergence_test(f, exact, t0, tf, exact(t0), midpoint_method, 2, "midpoint_method")

#### test 4 ####
f = lambda y,t:  y/t*np.log(y)
y0 = np.exp(1.)
exact = lambda t: np.exp(2.*t)
t0 = 0.5
tf = 5
convergence_test(f, exact, t0, tf, exact(t0), euler_method, 1, "euler_method")
convergence_test(f, exact, t0, tf, exact(t0), midpoint_method, 2, "midpoint_method")
