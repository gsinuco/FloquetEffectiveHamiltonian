#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:50:40 2020


@author: German Sinuco

Here we evaluate the Floquet spectrum of an optical lattice periodically driven.


TO DO: 23/05/2020

     0. CAN WE FIND THE REPRESENTATIO OF THE HAMILTONIAN IN MOMENTUM?
     1. FIND THE SPATIAL REPRESENTATION OF THE MOMENTUM OPERATOR:
         - WRITE IT AS A DIAGONAL OF K^2
         - TRANSFORM THE OPERATOR TO THE BASIS OF BLOCH STATES
         - TRANSFORM THE OPERATOR TO THE POSITION REPRESENTATION         
         - SUBSTRACT THE KINETIC ENERGY FROM THE TOTAL EFFECTIVE ENERGY
         - or use the bloch spectrum and a representation of cos(k_0*x + phi_0)
     
     2. CHECK THE ROLE OF THE FOLDING OF THE FLOQUET SPECTRUM. tHIS CAN BE DONE WITH A 
        STATIC SYSTEM, WITH A LONG TIME EVOLUTION.
         - SET A LONG TIME EVOLUTION
         - CALCULATE THE kINIETIC ENERGY, THE TOTAL EFFECTIVE H
         - SHOULD WE FOLD THE KINETIC ENERGY?
     

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import OpticalLatticeModule as OLM

jj = np.complex(0.0,1.0)
            

# Set the domain of the wavefunctions and the potential energy
x_l = -64.0
x_r =  64.0    
N_x =  128*4# number of points in real space
x   =  np.linspace(x_l,x_r,N_x)

# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) cos(2*k_x  x  +  2.0*phase)
V_0      = 20.1 # ~ Band gap
phase_x  = 0.0     #np.pi/3.0
omega    = 1000.0     # in units of omega_0 = (E)/hbar, with E = (2 hbar/lambda_0)/m
phase_t  = 0.0

OL    = OLM.OpticalLattice(V_0,phase_x)#OpticalLatticeModule.OpticalLattice(V_0,k_x,phase)


#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 2.0*np.pi/omega # total time to integrate
t1   = t0 + DT # final time #
N_t  = 256  # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 


solver = ode(OLM.Schrodinger_RHS_position).set_integrator('zvode',method='bdf')

U_T = np.zeros([N_x,N_x],dtype=np.complex)

#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
for j_ in range(N_x):
    psi0 = np.zeros(N_x,dtype=np.complex)
    psi0[j_] = 1.0
            
#    # define the parameters of the integrator
#    # TO DO: TEST THE INTEGRATOR
    solver.set_initial_value(psi0,t0).set_f_params(x,OL)

    psi_t = psi0 # Initialise the time dependent #wavefunction 
    i_ = 0
    while solver.successful() and solver.t<t1  and i_+1<N_t:
        i_ = i_+1            
        psi_t = solver.integrate(solver.t+dt)
    U_T[:,j_] = psi_t

 
lambda_u,U = np.linalg.eig(U_T)
e_u = -np.arctan(np.imag(lambda_u)/np.real(lambda_u))/DT

H_eff_F = np.diag(e_u) # Effective Hamiltonian in the basis Floquet states
    
#Transformation from the Floquet basis to the basis of position
H_eff_x = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U

V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)

#%%

plt.contourf(np.abs(H_eff_F))
plt.show()   


plt.plot(e_u)
plt.show()
    
plt.plot(x,np.diag(H_eff_x))
plt.show()

plt.plot(x[0:64],np.diag(H_eff_x)[0:64])
#plt.contourf(np.abs(U))
#plt.plot(np.abs(U))
plt.show()

