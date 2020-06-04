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
import FloquetEffectiveHamiltonian as FEH
import Bloch_wave as BW

#from OpticalLatticeModule import Bloch_spectrum as BS

jj = np.complex(0.0,1.0)
            

# Set the domain of the wavefunctions and the potential energy
x_l = -255.0
x_r =  255.0    
N_x =  5137# number of points in real space
x   =  np.linspace(x_l,x_r,N_x)

# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) cos(2*k_x  x  +  2.0*phase)
V_0      = 20.1 # ~ Band gap
phase_x  = 0.0     #np.pi/3.0
omega    = 2000.0     # in units of omega_0 = (E)/hbar, with E = (2 hbar/lambda_0)/m
phase_t  = 0.0

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
# define the number of bands to evaluate: make sure it is odd
Bands = 3
# define the number of values of momentum in the Brilloun Zone: make sure is even
L     = 64


#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 7# number of bands to include in the Schrodinger equation
#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 2.0*np.pi/omega # total time to integrate
t1   = t0 + DT # final time #
N_t  = 256  # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 

#%%
#Initialize the class of Optical Lattice
OL    = OLM.OpticalLattice(V_0,phase_x)#OpticalLatticeModule.OpticalLattice(V_0,k_x,phase)
#Initialize the class of the Bloch spectrum
BS    = OLM.Bloch_spectrum(L,Bands)

# get the Bloch Spectrum and wavefunctions in the momentum representatoin
BlochSpectrum,BlochWavefun = OLM.Bloch_spectrum.Spectrum(OL,BS) 

BlochWavefun[0:3,0:3]@np.transpose(np.conjugate(BlochWavefun[3:6,0:3]))
# get the Bloch wavefunctions is the position representation
RealSpaceBlochWavefun,RealSpaceMomentumWavefun           = OLM.Bloch_spectrum.RealSpaceWavefun(BS,x,BlochWavefun) 

#%%
N_x  = 32
BlochSpectrum,RealSpaceBlochWavefun = BW.BlochSpectrum(L,N_x,Bands,3)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceBlochWavefun[:, i], N_x)

plt.plot(k_u,BlochSpectrum,".")
plt.show()
#solver = ode(OLM.Schrodinger_Momentum_RHS).set_integrator('zvode',method='bdf')
#H_0    = BS.H_0[0:N_Bands*(L+1),0:N_Bands*(L+1)] # Hamiltonian in the basis of momentum
#%%
solver = ode(OLM.Schrodinger_RHS).set_integrator('zvode',method='bdf')

E_0 = BlochSpectrum[:,0]
for i_ in range(N_Bands-1):
    E_0 = np.concatenate([E_0,BlochSpectrum[:,i_+1]],axis=0) # two bands



U_k = np.transpose(np.reshape(BlochWavefun[:,0],[L+1,Bands]))
for n in range(N_Bands-1):
    U_k_  = np.transpose(np.reshape(BlochWavefun[:,n+1],[L+1,Bands]))
    U_k = np.concatenate([U_k,U_k_],axis=1)


U_x = np.transpose(np.reshape(RealSpaceBlochWavefun[:,0],[L+1,N_x]))
for n in range(N_Bands-1):
    U_x_  = np.transpose(np.reshape(RealSpaceBlochWavefun[:,n+1],[L+1,N_x]))
    U_x = np.concatenate([U_x,U_x_],axis=1)


plt.contourf(np.log(np.abs(np.transpose(np.conjugate(U_x))@U_x)));
plt.colorbar()
plt.show()


U_T = np.zeros([N_Bands*(L+1),N_Bands*(L+1)],dtype=np.complex)
#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

U_T,U_x = FEH.FloquetStroboscopicOperator(N_Bands,t0,DT,N_t,x,OL,BlochSpectrum,RealSpaceBlochWavefun)



lambda_u,U,e_u,H_eff_kn,H_eff_x = FEH.EffectiveFloquetHamiltonian(U_T,DT,U_x)

V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)
k_u =np.argmax(np.abs(U),axis=0)
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

plt.contourf(np.real(H_eff_kn))
plt.colorbar()
plt.show()   
    
plt.plot(k_u,e_u)
plt.show()

plt.plot(np.diag(np.real(H_eff_kn))[0:e_u.shape[0]])
plt.show()

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.plot(x,np.diag(H_eff_x))
plt.show()

plt.plot(x[0:32],np.diag(H_eff_x)[0:32])
plt.show()

#Transformation from the Boch basis to the basis of momentum
H_eff_k = U_k@H_eff_kn@np.transpose(np.conjugate(U_k))
plt.plot(np.diag(H_eff_k))
plt.show()

