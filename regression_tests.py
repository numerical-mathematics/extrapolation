from __future__ import division
import numpy as np
import math 
import time
import ex_parallel as ex_p
import fnbod
from compare_test import kdv_init, kdv_func, kdv_solout, burgers_init, burgers_func, burgers_solout


def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref)

def regression_tst(func, y0, t, y_ref, tol_boundary=(0,6), h0=0.5, mxstep=10e4,
        adaptive="order", p=4, solout=(lambda t: t)):
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]

    err = np.zeros(len(tol))
    print ''
    for i in range(len(tol)):
        print tol[i]
        ys, infodict = ex_p.ex_midpoint_parallel(func, y0, t, atol=tol[i], 
            rtol=tol[i], mxstep=mxstep, adaptive="order", full_output=True)
        y = solout(ys[1:len(ys)])
        err[i] = relative_error(y, y_ref)
    return err

def f_1(y,t):
    lam = -1j
    y0 = np.array([1 + 0j])
    return lam*y

def exact_1(t):
    lam = -1j
    y0 = np.array([1 + 0j])
    return y0*np.exp(lam*t)

def f_2(y,t):
    return 4.*y*float(np.sin(t))**3*np.cos(t)
    
def exact_2(t):
    y0 = np.array([1])
    return y0*np.exp((np.sin(t))**4)

def f_3(y,t):
    return 4.*t*np.sqrt(y)
    
def exact_3(t):
    return np.array([(1.+t**2)**2])

def f_4(y,t):
    return y/t*np.log(y)
    
def exact_4(t):
    return np.array([np.exp(2.*t)])

def f_5(y,t):
    return fnbod.fnbod(y,t)

def check(err, err_ref, test_num):
    assert (np.less_equal(err, err_ref*10)).all(), ("REGRESSION TEST " + str(test_num) + " FAILED")

########### RUN TESTS ###########
if __name__ == "__main__":
    # Test 1
    t0, tf = 0, 10
    y0 = exact_1(t0)
    y_ref = exact_1(tf)
    err = regression_tst(f_1, y0, [t0, tf], y_ref)
    err_ref_1 = np.array([5.88838286e-04, 1.81025200e-06, 1.13209455e-08, 7.56285559e-11, 1.38080058e-13, 5.00877951e-14]) 
    check(err, err_ref_1, 1)
    print err

    # Test 2
    t0, tf = 0, 10
    y0 = exact_2(t0)
    y_ref = exact_2(tf)
    err = regression_tst(f_2, y0, [t0, tf], y_ref)
    err_ref_2 = np.array([2.47659081e-04, 3.59278996e-06, 7.14037355e-07, 3.21678516e-09, 1.13278021e-11, 4.61973180e-13]) 
    check(err, err_ref_2, 2)
    print err

    # Test 3
    t0, tf = 0, 10
    y0 = exact_3(t0)
    y_ref = exact_3(tf)
    err = regression_tst(f_3, y0, [t0, tf], y_ref)
    err_ref_3 = np.array([4.57090672e-06, 2.50348696e-08, 1.07432535e-10, 2.77778815e-12, 2.08628331e-14, 2.49640738e-15]) 
    check(err, err_ref_3, 3)
    print err

    # Test 4
    t0, tf = 0.5, 10
    y0 = exact_4(t0)
    y_ref = exact_4(tf)
    err = regression_tst(f_4, y0, [t0, tf], y_ref)
    err_ref_4 = np.array([4.55316677e-03, 6.64359288e-05, 2.14299317e-07, 6.15033099e-10, 1.68573340e-11, 3.53697615e-13]) 
    check(err, err_ref_4, 4)
    print err

    # Test 5
    t0, tf = 0, 0.08
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference.txt")
    err = regression_tst(f_5, y0, [t0, tf], y_ref)
    err_ref_5 = np.array([4.88309757e-01, 1.41317737e-01, 1.40019042e-03, 1.25488605e-04, 2.74782735e-07, 3.21542638e-09]) 
    check(err, err_ref_5, 5)
    print err

    # Test 6
    t0, tf = 0, 0.003
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv.txt")
    err = regression_tst(kdv_func, y0, [t0, tf], y_ref, solout=kdv_solout)
    err_ref_6 = np.array([1.77032382e-05, 8.79188769e-08, 1.14493849e-09, 4.88142372e-12, 7.11948506e-12, 6.82866795e-13]) 
    check(err, err_ref_6, 6)
    print err

    # Test 7
    t0, tf = 0, 3.
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers.txt")
    err = regression_tst(burgers_func, y0, [t0, tf], y_ref, solout=burgers_solout, tol_boundary=(0,4))
    err_ref_7 = np.array([3.48157737e-10, 4.45753764e-11, 6.35315497e-13, 2.45959376e-14]) 
    check(err, err_ref_7, 7)
    print err
