import numpy as np
import matplotlib.pyplot as plt
def time_step(x_old,nu,dt):
    xnew = np.zeros((2,1))
    xnew[0,0] = x_old[0,0] + dt*x_old[1,0]
    xnew[1,0] = x_old[1,0] - nu*dt*np.sin(x_old[0,0])
    return xnew
N_step = 100
dt = 0.2
nu = 0.25
X = np.zeros((2,N_step))
X[0,0] = 0.0 + np.random.rand(1)
X[1,0] = 0.0 + np.random.rand(1)
for t in range(0,N_step-1): 
    X[:,t+1:t+2] = time_step(X[:,t:t+1],nu,dt)
    X_meas = 
plt.figure()
plt.plot(X[0,:])
plt.draw()
plt.pause(0.001)
plt.pause(100)