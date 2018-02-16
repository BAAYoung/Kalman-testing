import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import copy
def time_step(x_old,nu,dt):
    xnew = copy.deepcopy(x_old)
    xnew[0,:] = x_old[0,:] + dt*x_old[1,:]
    xnew[1,:] = x_old[1,:] - nu*dt*np.sin(x_old[0,:])

    return xnew
N_step = 100
dt = 0.2
nu = 0.25
sigma_signal = 1.0
X = np.zeros((2,N_step))
X_meas = np.zeros((2,N_step))

X[0,0] = 0.0 + np.random.rand(1)
X[1,0] = 0.0 + np.random.rand(1)

X_meas[:,0] = X[:,0] + sigma_signal*np.random.randn(2)

Pn = np.eye(2)*sigma_signal
xh = X_meas[:,0:1]
P0 = np.eye(2)*sigma_signal
x0 = np.zeros((6,1))
x0[0:2,:] = xh
P0a = np.zeros((6,6))
P0a[0:2,0:2] = P0
P0a[2:4,2:4] = P0*0.01
P0a[4:6,4:6] = Pn
Px = P0a
Px_posterior = np.zeros((4,4))
L = 6
lam = 1e-6 - L
Xsigma = np.zeros((L,2*L+1))

alpha = 1e-3
beta = 2 #optimal for gaussian distributed data
Wm = np.ones((2*L+1,1))/(2*(L+lam))
Wc = np.ones((2*L+1,1))/(2*(L+lam))
Wm[0] = lam/(L+lam)
Wc[0] = lam/(L+lam) +(1-alpha**2 + beta)
for t in range(0,N_step-1): 
    X[:,t+1:t+2] = time_step(X[:,t:t+1],nu,dt)
    X_meas[:,t+1:t+2] = X[:,t+1:t+2] + sigma_signal*np.random.randn(2,1)
    #finding square root of Px:
    Pxsqrt = scipy.linalg.sqrtm(Px)*np.sqrt(L+lam)
    Xsigma[:,0:1] = x0

    for i in range(0,L):
        Xsigma[:,i+1:i+2] = x0 + Pxsqrt[:,i:i+1]
    #change values to match i with PXsqrt    
    for i in range(L,2*L):
        Xsigma[:,i+1:i+2] = x0 - Pxsqrt[:,i-L:i+1-L]
    Xsigma_posterior = time_step(Xsigma[0:4,:],nu,dt)
    x_posterior = np.dot(Xsigma_posterior,Wm)
    for i in range(0,2*L+1):
        Px_posterior += Wc[i]*np.dot(Xsigma_posterior[:,i:i+1] - x_posterior,(Xsigma_posterior[:,i:i+1] - x_posterior).T )
    Ysigma_posterior = np.concatenate((Xsigma_posterior,Xsigma[4:6,:])) #measurement function is an identity matrix mapping:
    y_posterior = np.dot(Ysigma_posterior,Wc)
    #measurement equations:
    Pyy += 


    
plt.figure()
plt.plot(X_meas[0,:])
plt.draw()
plt.pause(0.001)
plt.pause(100)