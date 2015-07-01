Experiments with a modified version of Ernst Hairer's ODEX extrapolation code
and the original DOP853 code.
The ODEX code has been converted to fixed order and an OpenMP parallel loop added.
There are two versions:

 - odex.f: uses dynamic load balancing
 - odex_load_balanced.f: manually load balanced

The tests in the paper can be run via

    python run_tests.py
