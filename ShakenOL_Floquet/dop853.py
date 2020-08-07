#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:51:40 2020

@author: german
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import complex_ode

sol = []
def solout(t, y):
    sol.append([t, *y])

def rhs(t, z, omega):
    jj   = np.complex(0.0,1.0)
    x, y = z
    #f = [-jj*y, x]
    f = [-jj*omega*y, jj*omega*x]
    #return [1.0/(t - 10.0 - 1j)]
    return f
 #%%
# Create an `ode` instance to solve the system of differential
# equations defined by `fun`, and set the solver method to 'dop853'.

class myfuncs(object):
    def __init__(self, f, fargs=[]):
        self._f = f
        self.fargs=fargs

    def f(self, t, z):
        return self._f(t, z, *self.fargs)


omega_ = 2.0 #* np.pi


#integrator = 'zvode'
integrator = 'dopri5'
case = myfuncs(rhs, fargs=[omega_] )
solver = complex_ode(case.f)
#solver = complex_ode(rhs)
solver.set_integrator(integrator)


# Function to call at each timestep
solver.set_solout(solout)

# Give the value of omega to the solver. This is passed to
# `fun` when the solver calls it.
#omega = np.empty([1],dtype=np.float64)


# Set the initial value z(0) = z0.
t0 = 0.0
t1 = 2.5
z0 = [1, -1]
solver.set_initial_value(z0, t0)

# Perform the integration
solver.integrate(t1)

#%%
# Convert our solution into a numpy array for convenience
asol = []
asol = np.array(sol)
    
# Plot everything
plt.figure()
plt.plot(np.real(asol[:,0]), np.real(asol[:,1:]), 'b.-', markerfacecolor='b')
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.grid()
    
#plt.figure()
#plt.plot(asol[:,0], asol[:,2], 'b.-', markerfacecolor='b')
#plt.xlabel('t (s)')
#plt.ylabel('y\' (m/s)')
#plt.grid()
    
plt.show()

