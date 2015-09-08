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
        adaptive="order", p=4, solout=(lambda t: t), nworkers=2):
    tol = [1.e-3,1.e-5,1.e-7,1.e-9,1.e-11,1.e-13]
    a, b = tol_boundary
    tol = tol[a:b]

    err = np.zeros(len(tol))
    print ''
    for i in range(len(tol)):
        print tol[i]
        ys, infodict = ex_p.ex_midpoint_parallel(func, y0, t, atol=tol[i], 
            rtol=tol[i], mxstep=mxstep, adaptive="order", full_output=True, nworkers=nworkers)
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
    return y0*np.exp(np.dot(lam,t))

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

def check(err, err_ref, test_name):
    """
    Checks if err equals err_ref, returns silently if err equals err_ref (matching 10 decimals)
    or raises an exception otherwise
    @param err: calculated error
    @param err_ref: expected error
    @param test_name: tests explanation and/or explanatory name
    """
    np.testing.assert_array_almost_equal(err, err_ref, 10, "REGRESSION TEST " + test_name + " FAILED")
    
    
########### RUN TESTS ###########

def non_dense_tests():
    print("\n Executing non dense tests")
    # Test 1
    t0, tf = 0, 10
    y0 = exact_1(t0)
    y_ref = exact_1(tf)
    err = regression_tst(f_1, y0, [t0, tf], y_ref)
    err_ref_1 = np.array([1.02154130e-03, 1.22710679e-05, 9.25784586e-08, 7.71182340e-10, 4.91119065e-12, 4.44094761e-14]) 
    check(err, err_ref_1, "Test 1: linear ODE")
    print err
 
    # Test 2
    t0, tf = 0, 10
    y0 = exact_2(t0)
    y_ref = exact_2(tf)
    err = regression_tst(f_2, y0, [t0, tf], y_ref)
    err_ref_2 = np.array([8.54505911e-02, 2.48291275e-04, 8.17836269e-07, 2.85969698e-09, 2.54201200e-11, 9.10317032e-13])
    check(err, err_ref_2, "Test 2")
    print err
 
    # Test 3
    t0, tf = 0, 10
    y0 = exact_3(t0)
    y_ref = exact_3(tf)
    err = regression_tst(f_3, y0, [t0, tf], y_ref)
    err_ref_3 = np.array([1.93968837e-05, 9.48240508e-08, 1.21300220e-09, 2.83645194e-10, 1.74748516e-13, 5.88438882e-15]) 
    check(err, err_ref_3, "Test 3")
    print err
 
    # Test 4
    t0, tf = 0.5, 10
    y0 = exact_4(t0)
    y_ref = exact_4(tf)
    err = regression_tst(f_4, y0, [t0, tf], y_ref)
    err_ref_4 = np.array([6.77863684e-03, 9.50343703e-05, 4.90535263e-07, 3.36128205e-09, 2.47615358e-11, 2.92147596e-13]) 
    check(err, err_ref_4, "Test 4")
    print err
  
    # Test 5
    t0, tf = 0, 0.08
    y0 = fnbod.init_fnbod(2400)
    y_ref = np.loadtxt("reference.txt")
    err = regression_tst(f_5, y0, [t0, tf], y_ref)
    err_ref_5 = np.array([4.6785364959e-01, 8.0415058627e-02, 1.6090309508e-03, 1.0235932826e-05, 1.0988144963e-07, 1.1568408676e-09]) 
    check(err, err_ref_5, "Test 5")
    print err
 
    # Test 6
    t0, tf = 0, 0.003
    y0 = kdv_init(t0)
    y_ref = np.loadtxt("reference_kdv.txt")
    err = regression_tst(kdv_func, y0, [t0, tf], y_ref, solout=kdv_solout)
    err_ref_6 = np.array([1.20373187e-05, 7.98230720e-08, 1.63759715e-10, 1.75095038e-12, 3.66764576e-12, 1.69146958e-12]) 
    check(err, err_ref_6, "Test 6")
    print err
 
    # Test 7
    t0, tf = 0, 3.
    y0 = burgers_init(t0)
    y_ref = np.loadtxt("reference_burgers.txt")
    err = regression_tst(burgers_func, y0, [t0, tf], y_ref, solout=burgers_solout, tol_boundary=(0,4))
    err_ref_7 = np.array([6.92934673e-09, 4.45755379e-11, 6.26721092e-13, 2.49897416e-14]) 
    check(err, err_ref_7, "Test 7")
    print err


def dense_tests():
    print("\n Executing dense tests")
    # Test 1 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_1(t0)
    y_ref = exact_1([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_1, y0, t, y_ref)
    err_ref_1 = np.array([1.45770475e-04, 1.14841518e-06, 1.54196490e-06, 3.88285196e-06, 1.26814157e-07, 1.88666778e-08]) 
    check(err, err_ref_1, "Test 1 dense")
    print err
      
    # Test 2 dense 
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_2(t0)
    y_ref = exact_2([[2.5],[5],[7.5],[10]])
    err = regression_tst(f_2, y0, t, y_ref)
    err_ref_2 = np.array([3.17429818e-04, 4.86595063e-05, 1.10295116e-05, 1.99391509e-06, 2.11069640e-07, 9.94570884e-08])
    check(err, err_ref_2, "Test 2 dense")
    print err
  
    # Test 3 dense
    t0=0
    t = [t0,2.5,5,7.5,10]
    y0 = exact_3(t0)
    y_ref = exact_3(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_3, y0, t, y_ref)
    err_ref_3 = np.array([8.99830265e-06, 2.02137818e-06, 4.63896854e-08, 2.05977138e-08, 2.36656866e-09, 1.44506569e-09]) 
    check(err, err_ref_3, "Test 3 dense")
    print err
 
    # Test 4 dense
    t0 = 0.5
    t = [t0,2.5,5,7.5,10]
    y0 = exact_4(t0)
    y_ref = exact_4(np.array([[2.5],[5],[7.5],[10]]))
    err = regression_tst(f_4, y0, t, y_ref)
    err_ref_4 = np.array([4.43565713e-03,   9.09774798e-05, 6.04212715e-07, 2.39230449e-09, 5.32280637e-11, 6.44359887e-11]) 
    check(err, err_ref_4, "Test 4 dense")
    print err
    
    #TODO: add tests 5, 6 and 7 for dense output


if __name__ == "__main__":
    non_dense_tests()
    dense_tests()
    
