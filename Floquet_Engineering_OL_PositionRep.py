#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 9:12 2020


@author: German Sinuco

Here we evaluate the Floquet spectrum of an optical lattice periodically driven.


TO DO: 30/05/2020

    0. 
    
    1. SOLVE THE SCHRODINGER EQUATION IN THE POSITION REPRESENTATION
     
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
import FloquetEffectiveHamiltonian as FEH
import Bloch_wave as BW

#from OpticalLatticeModule import Bloch_spectrum as BS

jj = np.complex(0.0,1.0)
            
# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) cos(2*k_x  x  +  2.0*phase)
V_0      = 20.1 # ~ Band gap
phase_x  = 0.0     #np.pi/3.0
omega    = 600.0     # in units of omega_0 = (E)/hbar, with E = (2 hbar/lambda_0)/m
phase_t  = 0.0

OL    = OLM.OpticalLattice(V_0,phase_x)#OpticalLatticeModule.OpticalLattice(V_0,k_x,phase)

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 5  # Number of bands to evaluate
L     = 32 # number of unit cells
N_x   = 32 # Number of points in a unit cell
x     = np.linspace(0, L, N_x*L)

BlochSpectrum,RealSpaceBlochWavefun = BW.BlochSpectrum(L,N_x,Bands,1)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceBlochWavefun[:, i], N_x)

plt.plot(k_u,BlochSpectrum,".")
plt.show()

#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 3# number of bands to include in the Schrodinger equation
#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 2.0*np.pi/omega # total time to integrate
t1   = t0 + DT # final time #
N_t  = 128  # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 

#%%
E_0 = BlochSpectrum[0:N_Bands*L]
U_x = RealSpaceBlochWavefun[:,0:N_Bands*L]
plt.contourf(np.log(np.abs(np.transpose(np.conjugate(U_x))@U_x)));
plt.colorbar()
plt.show()
U_T = np.zeros([N_Bands*L,N_Bands*L],dtype=np.complex)

#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

U_T,U_x = FEH.FloquetStroboscopicOperatorV2(L,N_Bands,t0,DT,N_t,x,OL,BlochSpectrum,RealSpaceBlochWavefun)


lambda_u,U,e_u,H_eff_kn,H_eff_x = FEH.EffectiveFloquetHamiltonian(U_T,DT,U_x)

V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)

#%%

plt.contourf(np.abs(np.transpose(np.conjugate(U_x))@U_x))
plt.colorbar()
plt.show()

plt.contourf(np.abs(U_x))
plt.show()


H_eff_F = np.diag(e_u) # Effective Hamiltonian in the basis Floquet states

plt.contourf(np.abs(H_eff_F))
plt.show()   

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U

plt.contourf(np.abs(H_eff_kn))
plt.colorbar()
plt.show()   
#%%
plt.plot(np.diag(np.real(H_eff_kn))[0:e_u.shape[0]])
plt.show()

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.plot(x,np.diag(H_eff_x))
plt.show()

plt.plot(x[0:4*N_x],np.diag(H_eff_x)[0:4*N_x])
plt.show()


plt.plot(x,np.diag(H_eff_x))
plt.show()
