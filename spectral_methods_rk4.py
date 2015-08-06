from __future__ import division
import numpy as np


def kdv_equation(tmax):
    # Grid
    m = 256
    x = np.arange(-m/2,m/2)*(2*np.pi/m)
    dx = x[1]-x[0]
    L = x[-1]-x[0] + dx
    k = 0.4/m**2
    nmax = int(round(tmax/k))
    k = 1.0*tmax/nmax

    # Initial data
    A = 25; B = 16;
    u = 3*A**2/np.cosh(0.5*(A*(x+2.)))**2 + 3*B**2/np.cosh(0.5*(B*(x+1)))**2
    uhat = np.fft.fft(u)

    # This is the order in which numpy's FFT gives the frequencies:
    xi=np.fft.fftfreq(m)*m/(L/(2*np.pi))

    xi3 = 1j*xi**3
    g = -0.5j * k * xi
    E = np.exp(k*xi3/2.)
    E2 = E**2

    for n in range(1,nmax+1):
        # Runge-Kutta stages
        a = g*np.fft.fft(np.real(np.fft.ifft(uhat))**2)
        b = g*np.fft.fft(np.real(np.fft.ifft(E*(uhat+a/2.)))**2)
        c = g*np.fft.fft(np.real(np.fft.ifft(E*uhat + b/2) )**2)
        d = g*np.fft.fft(np.real(np.fft.ifft(E2*uhat + E*c))**2)
        uhat = E2*uhat + (E2*a + 2*E*(b+c) + d)/6.

    return uhat

def burgers_equation(tmax):
    epsilon = 0.1

    # Grid
    m = 64
    x = np.arange(-m/2,m/2)*(2*np.pi/m)
    dx = x[1]-x[0]
    L = x[-1]-x[0] + dx
    k = 2./m
    nmax = int(round(tmax/k))
    k = 1.0*tmax/nmax

    # Initial data
    u = np.sin(x)**2 * (x<0.)
    # u = np.sin(x)**2
    uhat = np.fft.fft(u)

    # This is the order in which numpy's FFT gives the frequencies:
    xi=np.fft.fftfreq(m)*m/(L/(2*np.pi))

    eps_xi2 = epsilon * xi**2.
    g = -0.5j * k * xi
    E = np.exp(-k*eps_xi2/2.)
    E2 = E**2

    nmax = int(round(tmax/k))

    for n in range(1,nmax+1):
        # Runge-Kutta stages
        a = g*np.fft.fft(np.real(np.fft.ifft(uhat))**2)
        b = g*np.fft.fft(np.real(np.fft.ifft(E*(uhat+a/2.)))**2)
        c = g*np.fft.fft(np.real(np.fft.ifft(E*uhat + b/2) )**2)
        d = g*np.fft.fft(np.real(np.fft.ifft(E2*uhat + E*c))**2)
        uhat = E2*uhat + (E2*a + 2*E*(b+c) + d)/6.
        
    return uhat

def relative_error(y, y_ref):
    return np.linalg.norm(y-y_ref)/np.linalg.norm(y_ref*len(y_ref))

# compare the computed reference solutions from create_reference.py 
# with those solution from the RK4 code above
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.hold('true')
    from create_reference import kdv_reference, kdv_solout, burgers_reference, burgers_solout

    N = 256
    x = np.arange(-N/2,N/2)*(2*np.pi/N)
    kdv_ref = np.loadtxt("reference_kdv.txt")
    rk4_ref = np.squeeze(np.real(np.fft.ifft(kdv_equation(0.003))))
    rk4_line, = plt.plot(x, rk4_ref, "s-")
    kdv_line, = plt.plot(x, kdv_ref, "s-")
    plt.legend([rk4_line, kdv_line], ["Rk4", "kdv"], loc=1)
    plt.show()
    print relative_error(kdv_ref, rk4_ref)

    N = 64
    x = np.arange(-N/2,N/2)*(2*np.pi/N)
    burgers_ref = np.loadtxt("reference_burgers.txt")
    rk4_ref = np.squeeze(np.real(np.fft.ifft(burgers_equation(3.))))
    rk4_line, = plt.plot(x, rk4_ref, "s-")
    burgers_line, = plt.plot(x, burgers_ref, "s-")
    plt.legend([rk4_line, burgers_line], ["Rk4", "burgers"], loc=1)
    plt.show()
    print relative_error(burgers_ref, rk4_ref)
