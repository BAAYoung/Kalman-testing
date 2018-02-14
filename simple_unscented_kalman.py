import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
nu = 0.2
x0 = 0.0
u0 = 1.0
N_step = 1000
X = np.zeros((2,N_step))
X[0,0] = x0
X[1,0] = u0
print X
for t in range(0,N_step-1):
    X[0,t+1] = X[0,t] + dt*X[1,t]
    X[1,t+1] = X[1,t] - dt*nu*X[1,t]**2

plt.plot(X.T) 
plt.draw()
plt.pause(100)
