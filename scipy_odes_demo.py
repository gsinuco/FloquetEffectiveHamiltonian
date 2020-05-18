#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:13:37 2020

@author: german
"""


import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.integrate import complex_ode


def Schrodinger_RHS(t,psi,E_0,V_0): #ode
    jj =np.complex(0.0,1.0)
    H_0 = np.diag(E_0)
    V   = V_0
    H   = H_0 + V
    dpsidt_RHS = -jj*H@psi
    return dpsidt_RHS



D_H = 64
psi0 = np.ones(D_H)
t0 = 0
t    = np.linspace(t0, t0+0.2, 101)

E_0 = np.random.rand(D_H)
V_0 = np.random.rand(D_H,D_H)

#sol = odeint(Schrodinger_RHS, psi0, t,args=(E_0,V_0))
solver = ode(Schrodinger_RHS).set_integrator('zvode',method='bdf')
solver.set_initial_value(psi0,t0).set_f_params(E_0,V_0)
t1 = t0+0.2
N_t = 101
dt = 0.2/N_t

psi_t = np.zeros([N_t+1,D_H])
psi_t[0,:] = psi0
i_ = 0
while solver.successful() and solver.t<t1:
    i_ = i_+1
    #print(solver.t+dt, solver.integrate(solver.t+dt))
    psi_t[i_,:] = solver.integrate(solver.t+dt)
        

plt.plot(psi_t[0:i_,:])
plt.show()

#%%

b = 0.25
c = 5.0
y0 = [np.pi - 0.1, 0.0,0.0,0.0]
t = np.linspace(0, 10, 101)


sol = odeint(ode_rhs, y0, t)

import matplotlib.pyplot as plt
plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.plot(sol)
plt.show()

