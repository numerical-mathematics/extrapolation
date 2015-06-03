# basic implementation of explicit extrapolation with Euler's and midpoint methods
# a convergence test on the example u' = lam*u is preformed at the end of this file

import numpy as np
import pylab as pl
import math 

# solve the first order ODE u' = f(u) on the interval [t0, tf] with 
# the intial value u(t0) = u0 and time step size dt
def euler_method (f, u0, t0, tf, dt):
  ts = np.arange(t0, tf, dt)
  us = np.zeros(len(ts), dtype=complex)
  us[0] = u0
  for i in range(len(ts) - 1):
    us[i+1] = us[i] + dt*f(us[i])
  return (ts, us)

# solve the first order ODE u' = f(u) on the interval [t0, tf] with 
# the intial value u(t0) = u0 and u(t0 + dt) = u1 and time step size dt
def midpoint_method (f, u0, u1, t0, tf, dt):
  ts = np.arange(t0, tf, dt)
  us = np.zeros(len(ts), dtype=complex)
  us[0] = u0
  us[1] = u1
  for i in range(len(ts) - 2):
    us[i+2] = us[i] + 2*dt*f(us[i+1])
  return (ts, us)


# Convergence Test 
# example: u' = f(u) = lam*u on [0, 10] --> exact solution is u = u0*e**(lam*t)

# change pramters lam and u0 to see different cases
lam = -1
u0 = 1

f = lambda u: lam*u
exact = lambda t: u0*np.exp(lam*t)

dts = np.asarray([2**(-k) for k in range(1, 10)])
err_euler = np.zeros(len(dts))
err_midpoint = np.zeros(len(dts))

for i in range(len(dts)):
  ts, us = euler_method(f, u0, 0, 10, dts[i])
  err_euler[i] = abs(us[-1].real - exact(10).real)
  ts, us = midpoint_method(f, u0, us[1], 0, 10, dts[i])
  err_midpoint[i] = abs(us[-1].real - exact(10).real)

pl.hold('true')
err_euler_line,  = pl.loglog(dts, err_euler, 's-') 
err_order_1,  = pl.loglog(dts, dts*(err_euler[5]/dts[5]))
err_midpoint_line,  = pl.loglog(dts, err_midpoint, 's-') 
err_order_2,  = pl.loglog(dts, (dts**2)*(err_midpoint[5]/dts[5]))
pl.legend([err_euler_line, err_midpoint_line], ['Euler Method', 'Midpoint Method'], loc=2)
pl.show()