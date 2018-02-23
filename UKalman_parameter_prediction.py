import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import copy

def VDpol(X,V,mu,dt):
    ''' Nonlinear discretised function describing the state evolution '''
    Xnew = copy.deepcopy(X) + V
    Xnew[0,:] += dt*X[1,:]
    Xnew[1,:] += -dt*X[1,:]*X[2,:]*X[1,:]
    Xnew[2,:] = X[2,:]
    return Xnew
def measure(X,Nk):
    ''' Nonlinear function describing measurement noise '''
    Mx = copy.deepcopy(X)
    Mx[2,:] = 10
    return Mx + Nk 
def Paugment(Px,Pv,Pn):
    ''' Create the Augmented probability matrix '''
    Pa = np.zeros((Px.shape[0]*3,Px.shape[1]*3))
    Pa[0:3,0:3] = Px
    Pa[3:6,3:6] = Pv
    Pa[6:9,6:9] = Pn
    return Pa

''' Simulation Parameters and initial conditions '''
dt = 0.1 
mu = 0.2
N_step = 200
X = np.zeros((3,N_step))
X_meas = np.zeros((3,N_step))
x0 = 0.0
u0 = 2.0
X[0,0] = x0
X[1,0] = u0
X[2,0] = mu

''' Error Parameters '''
kx = 1e0
kv = 1e-3*dt
kn = 1e-0
Px = np.eye(3)*kx
P0 = copy.deepcopy(Px)
Rv = np.eye(3)*kv
Rn = np.eye(3)*kn
X_meas[:,0:1] = measure(X[:,0:1],np.random.randn(3,1)*kn)

''' Kalman filter initial Properties '''
X_posterior = np.zeros((3,N_step))
X_posterior[0,0] = X_meas[0,0]
X_posterior[1,0] = X_meas[1,0]
X_posterior[2,0] = 2.0
''' Unscented Transform Properties '''
alpha = 1.0
beta = 2.0
kappa = 0.0
N = 9 #number of augmented states
lam = (alpha**2)*(N+kappa) - N
gamma = np.sqrt(N+lam)

''' Unscented Transform Weights '''
Wm = np.zeros((2*N+1,1))
Wc = np.zeros((2*N+1,1))
Wm[0,0] = lam/(N+lam)
Wc[0,0] = lam/(N+lam) + (1-alpha**2 + beta)
Wm[1:2*N+1,0] = 1/(2*N + 2*lam)
Wc[1:2*N+1,0] = 1/(2*N + 2*lam)
N_iter = 50
''' Running simulation/kalman filter prediction steps '''
for k in range(0,N_iter):
    for t in range(0,N_step-1):

        ''' Reseting probability matrices '''
        P_prior = np.zeros((3,3))
        Pyy = np.zeros((3,3))
        Pxy = np.zeros((3,3))
        Pka = Paugment(Px,Rv,Rn)

        ''' Updating the true and measured state of the system '''
        X[:,t+1:t+2] = VDpol(X[:,t:t+1],kv*np.random.rand(3,1),mu,dt)
        X_meas[:,t+1:t+2] = measure(X[:,t+1:t+2],kn*np.random.randn(3,1))

        ''' Creating Sigma Points '''
        XsigmaA = np.zeros((N,2*N+1))
        XsigmaA[0:3,0] = X_posterior[0:3,t]
        Pxsqrt = scipy.linalg.sqrtm(Pka)
        #print(Pxsqrt.shape)
        for i in range(0,N):
            XsigmaA[:,i+1] = XsigmaA[:,0] + gamma*Pxsqrt[:,i]
        for i in range(0,N):
            XsigmaA[:,i+N+1] = XsigmaA[:,0] - gamma*Pxsqrt[:,i]
        
        ''' Evolving Sigma points through the non-linear state update '''
        Xsigmax = VDpol(XsigmaA[0:3,:],XsigmaA[3:6,:],mu,dt)
        x_prior = np.dot(Xsigmax,Wm)
        for i in range(0,2*N+1):
            P_prior += Wc[i,0]*np.dot(Xsigmax[:,i:i+1] - x_prior,(Xsigmax[:,i:i+1] - x_prior).T)

        ''' Evolving Sigma points through non-linear measurement function '''
        Ysigma = measure(Xsigmax,XsigmaA[6:9,:])
        y_prior = np.dot(Ysigma,Wm)
        for i in range(0,2*N+1):
            Pyy += Wc[i,0]*np.dot(Ysigma[:,i:i+1] - y_prior,(Ysigma[:,i:i+1] - y_prior).T)
            Pxy += Wc[i,0]*np.dot(Xsigmax[:,i:i+1] - x_prior,(Ysigma[:,i:i+1] - y_prior).T)
        Kgain = np.dot(Pxy,np.linalg.inv(Pyy))
        X_posterior[:,t+1:t+2] = x_prior + np.dot(Kgain,X_meas[:,t+1:t+2]-y_prior)
        Px = P_prior - np.dot(np.dot(Kgain,Pyy),Kgain.T)

    X_posterior[0,0] = x0
    X_posterior[1,0] = u0
    X_posterior[2,0] = X_posterior[2,N_step-1]
    P_mu0 = Px[2,2]
    Px = P0
    Px[2,2] = P_mu0


    plt.plot(mu*np.ones(N_iter),'k')
    plt.plot(k,X_posterior[2,N_step-1],'r x')
    plt.draw()
    plt.pause(1e-2)
    ''' plt.plot(X_posterior[2,:])
    #plt.plot(X[1,:])
    plt.plot(X_meas[2,:],'.')
    plt.draw()
    plt.pause(100) '''

#print(Paugment(Px,Rv,Rn))
plt.figure()
plt.plot(X_posterior[1,:])
plt.plot(X[1,:])
plt.plot(X_meas[1,:],'.')
plt.figure()
plt.plot(X_posterior[0,:])
plt.plot(X[0,:])
plt.plot(X_meas[0,:],'.')
plt.draw()
plt.pause(100)


