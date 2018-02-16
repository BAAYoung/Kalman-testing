import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
nu = 0.2
x0 = 0.0
u0 = 1.0
N_step = 100
X = np.zeros((2,N_step))
X_meas = np.zeros((2,N_step))
X[0,0] = x0
X[1,0] = u0
X_meas[0,0] = x0
X_meas[1,0] = u0
A = np.matrix([[1.0, dt],[0.0,1.0-dt*nu]])
H = np.matrix([[1.0, 0.0],[0.0,1.0]])
print(A.shape)
sigma_meas0 = 0.1
sigma_meas1 = 0.1
sigma_acc = 0.2
Q = np.matrix([[0.0,0.0],[0.0,1.0]])*sigma_acc**2
R = np.matrix([[sigma_meas0**2,0.0],[0.0,sigma_meas1**2]])

Pposterior = np.eye(2)*0.001
xposterior = np.zeros((2,N_step))
xposterior[0,0] = x0
xposterior[1,0] = u0

for t in range(0,N_step-1):
    ''' X[0,t+1] = X[0,t] + dt*X[1,t]
    X[1,t+1] = X[1,t] - dt*nu*X[1,t]**2 '''


    X[:,t+1:t+2] =  np.dot(A,X[:,t:t+1])
    X[1,t+1] += np.random.randn(1)*sigma_acc
    X_meas[:,t+1:t+2] = np.dot(H,X[:,t+1:t+2])
    X_meas[0,t+1] += np.random.randn(1)*sigma_meas0
    X_meas[1,t+1] += np.random.randn(1)*sigma_meas1
    #kalman filter prediction:

    xprior = np.dot(A,xposterior[:,t:t+1])
    Pprior = np.dot(np.dot(A,Pposterior),A.T) + Q
    S = np.dot(np.dot(H,Pprior),H.T) + R
    Kgain  = np.dot(np.dot(Pprior,H.T),np.linalg.inv(S))
    xposterior[:,t+1:t+2] = xprior + np.dot(Kgain,(X_meas[:,t+1:t+2] - np.dot(H,xprior)))

    Pposterior = np.dot(np.eye(2) - np.dot(Kgain,H),Pprior)


plt.plot(X[0,:].T)
plt.plot(X_meas[0,:].T,'.')
plt.plot(xposterior[0,:].T)
plt.figure()
plt.plot(X[1,:].T)
plt.plot(X_meas[1,:].T,'.')
plt.plot(xposterior[1,:].T)
plt.draw()
plt.pause(100)
