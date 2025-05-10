# -*- coding: utf-8 -*-
"""
Created on Sep 15 12:30:18 2021
@author: Ling Yang
A simulation demonstration of PIM for the solution of the heat equation

"""

import numpy as np
import matplotlib.pyplot as plt
from mpimpy import memmatfp

## Define the meshgrid
dx = 0.1
L = 10
x = np.arange(0, L, dx)

dy = 0.1
y = np.arange(0, L, dy)

dt = 0.001
t = np.arange(0, 0.5, dt)
a = 1

U = np.zeros([len(x), len(y)])
X, Y = np.meshgrid(x, y)

## Define the initial condition
D = 1
x_bais = 5.
y_bais = 5.
U = np.exp(-((X-x_bais)**2 + (Y-y_bais)**2)/(D**2)) 

## Define the equation on the direction x and y
A1 = np.diag([-2]*len(x)) + np.diag([1]*(len(x)-1), 1) + np.diag([1]*(len(x)-1), -1)
A2 = np.diag([-2]*len(y)) + np.diag([1]*(len(y)-1), 1) + np.diag([1]*(len(y)-1), -1)

dpe_fp = memmatfp.fpmemdpe(HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

for n in range(len(t)-1): 
    
    # U = U + a*((np.dot(A1, U.T)/(dx**2)).T + np.dot(A2, U)/(dy**2))*dt      ## Ideal solution
    U = U + a*(dpe_fp.MapReduceDot(U, A1.T)/(dx**2) + (dpe_fp.MapReduceDot(U.T, A2.T)/(dy**2)).T)*dt   ## PIM solution

    ## boundary conditions, you can change it to other conditions
    
    U[0] = np.sin(2*np.pi*x)*0               
    U[-1] = np.sin(-2*np.pi*x)*0
    U[:,0] = U[:,0]*0
    U[:, -1] = U[:, -1]*0


fig = plt.figure(figsize=(7,8)) 
ax = fig.add_subplot(projection='3d') 
surf = ax.plot_surface(X, Y, U, alpha = 0.8, cmap=plt.cm.coolwarm) 
ax.set_zlim(0, 1)
fig.colorbar(surf, shrink=0.5, aspect=15)
plt.show()