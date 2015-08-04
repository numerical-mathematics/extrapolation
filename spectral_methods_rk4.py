from __future__ import division
import numpy as np

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
    uhat = np.fft.fft(u)

    # This is the order in which numpy's FFT gives the frequencies:
    xi=np.fft.fftfreq(m)*m/(L/(2*np.pi))

    eps_xi2 = epsilon * xi**2.
    g = -0.5j * k * xi
    E = np.exp(-k*eps_xi2/2.)
    E2 = E**2

    nplt = np.floor((tmax/25)/k)
    nmax = int(round(tmax/k))

    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # line, = axes.plot(x,u,lw=3)

    frames = [u.copy()]

    for n in range(1,nmax+1):
        # Runge-Kutta stages
        a = g*np.fft.fft(np.real(np.fft.ifft(uhat))**2)
        b = g*np.fft.fft(np.real(np.fft.ifft(E*(uhat+a/2.)))**2)
        c = g*np.fft.fft(np.real(np.fft.ifft(E*uhat + b/2) )**2)
        d = g*np.fft.fft(np.real(np.fft.ifft(E2*uhat + E*c))**2)
        uhat = E2*uhat + (E2*a + 2*E*(b+c) + d)/6.
        
        t = n*k
        # Plotting
        if np.mod(n,nplt) == 0:
            u = np.squeeze(np.real(np.fft.ifft(uhat)))
            frames.append(u.copy())

    # def plot_frame(i):
    #     line.set_data(x,frames[i])
    #     axes.set_title('t='+str(i*k))

    # matplotlib.animation.FuncAnimation(fig, plot_frame, frames=len(frames), interval=20)

    print t
    U_hat = uhat
    return U_hat

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

    nplt = np.floor((tmax/25)/k)

    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # line, = axes.plot(x,u,lw=3)

    frames = [u.copy()]
    tt = [0]

    for n in range(1,nmax+1):
        # Runge-Kutta stages
        a = g*np.fft.fft(np.real(np.fft.ifft(uhat))**2)
        b = g*np.fft.fft(np.real(np.fft.ifft(E*(uhat+a/2.)))**2)
        c = g*np.fft.fft(np.real(np.fft.ifft(E*uhat + b/2) )**2)
        d = g*np.fft.fft(np.real(np.fft.ifft(E2*uhat + E*c))**2)
        uhat = E2*uhat + (E2*a + 2*E*(b+c) + d)/6.

        t = n*k
        # Plotting
        if np.mod(n,nplt) == 0:
            u = np.squeeze(np.real(np.fft.ifft(uhat)))
            frames.append(u.copy())
            tt.append(t)
    
    # def plot_frame(i):
    #     line.set_data(x,frames[i])
    #     axes.set_title('t='+str(t))
    #     axes.set_xlim((-np.pi,np.pi))
    #     axes.set_ylim((0.,1.))
        
    # matplotlib.animation.FuncAnimation(fig, plot_frame, frames=len(frames), interval=20)

    print t    
    U_hat = uhat
    return U_hat


def relative_error(y, y_ref):
    return np.linalg.norm((y-y_ref)/y_ref)/(len(y)**0.5)

# compare the computed reference solutions from create_reference.py 
# with those solution from the RK4 code above
KdV_ref_hat = np.loadtxt("reference_KdV.txt").view('complex')
KdV_ref = np.squeeze(np.real(np.fft.ifft(KdV_ref_hat)))
U_hat = kdv_equation(0.003)
U = np.squeeze(np.real(np.fft.ifft(U_hat)))
print relative_error(KdV_ref_hat, U_hat)
print relative_error(KdV_ref, U)
    
burgers_ref_hat = np.loadtxt("reference_burgers.txt").view('complex')
burgers_ref = np.squeeze(np.real(np.fft.ifft(burgers_ref_hat)))
U_hat = burgers_equation(3.)
U = np.squeeze(np.real(np.fft.ifft(U_hat)))
print relative_error(burgers_ref_hat, U_hat)
print relative_error(burgers_ref, U)
