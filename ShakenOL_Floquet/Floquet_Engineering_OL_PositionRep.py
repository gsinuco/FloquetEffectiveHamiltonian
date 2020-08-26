#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BukovSoftware descrbies how to run in parallel the solution of the schrodinger equiation

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
import time 

import OpticalLatticeModule as OLM
import FloquetEffectiveHamiltonian as FEH
import Bloch_wave as BW

jj = np.complex(0.0,1.0)
            
# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) (cos(2*k_x  x  +  2.0*phase) + 1)
V_0      = 5 # ~ Band gap
phase_x  = 0.0  # spatial phase
OL       = OLM.OpticalLattice(V_0,phase_x)

#Driving Frequency
omega    = 5.0  #  Driving frequency
#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 3  # Number of bands to evaluate
L     = 32 # number of unit cells
N_x   = 32 # Number of points in a unit cell
dx    = 1.0/N_x
x     = np.linspace(0, L, N_x*L)

BlochSpectrum,RealSpaceBlochWavefun = BW.BlochSpectrum(L,N_x,Bands,V_0,2.0*phase_x)

k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceBlochWavefun[:, i], N_x)

plt.plot(k_u,BlochSpectrum,".")
#plt.plot(k_u,k_u*k_u,".")
#plt.plot(k_u,2*(k_u+np.pi)**2 + np.pi**2,".")
#plt.plot(k_u,4*(k_u+np.pi)*(k_u+np.pi)+np.pi*np.pi,".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
#plt.ylim(0,11.5)
plt.show()


#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 32    # number of bands to include in the Schrodinger equation
N_t     = 2048#512 # number of time steps

t0   = 0                 # initial time
DT   = 2.0*np.pi/(omega) # total time to integrate
t1   = t0 + DT           # final time #
dt   = DT/N_t            # time step
t    = np.linspace(t0,t0+DT, N_t) 

U_x = np.zeros_like(RealSpaceBlochWavefun[:,0:N_Bands*L])
U_T = np.zeros([N_Bands*L,N_Bands*L],dtype=np.complex)
#%%   
# TO DO:
# The integrator is too slow, it requires a large number of time steps (> 4096)
# for an accurate time-evolution (~10^-3) 
# Find an alternative: maybe https://pydstool.github.io/PyDSTool/Tutorial/Tutorial_Calcium.html
# http://hplgit.github.io/odespy/doc/pub/tutorial/html/main_odespy.html
start = time.time()
phi_t,U_T,U_x = FEH.FloquetStroboscopicOperatorV3(L,N_Bands,t0,DT,N_t,x,OL,BlochSpectrum,RealSpaceBlochWavefun)
lambda_u,U,e_u = FEH.FloquetSpectrum(U_T,DT,U_x)
end = time.time()
print(end - start)

# saving:
#f = open("UnfoldingFloquetL16Nt2048_N_Band3.dat", "w")
#f.write("# phi_t[8:256] time evolution of the phase\n#eigenvalues\n#eigenenergies#\n#transformation\n")        # column names
#np.savetxt(f, phi_t)
#np.savetxt(f, lambda_u.T)
#np.savetxt(f, e_u.T)
#np.savetxt(f, U)

#f.close()


#%%
#################################################################
########  UNFOLDING THE FLOQUET SPECTRUM ########################
#################################################################
# The final value of the phase give us the energy folded
direction = -1
phases,folding_counter,phase_sec_order,phase_sec_order_,phase_sec_order_aux = FEH.UnfoldingFloquetSpectrum(L*N_Bands,dt,N_t,phi_t,direction)
e_u_unfolded = e_u + folding_counter*omega
            
#grad_new_ = np.zeros([L*N_Bands,N_t],dtype=np.float32)
#for i in range(phi_t.shape[1]-2):   
#        grad_new_[:,i+1] = phi_t[:,i+1] - phi_t[:,i]    
    
#folding_counter = np.zeros([phi_t.shape[0]],dtype=np.int32)
#for i in range(phi_t.shape[0]):   
#        folding_counter[i] = np.array(np.where(grad_new_[i,:]<-np.pi)).size
#e_u_unfolded = e_u + folding_counter*omega
 
plt.plot(t[0:N_t-1],np.transpose(phi_t[:,0:N_t-1]),"-")
plt.plot(t[0:N_t],np.transpose(phi_t[:,0:N_t]),"-")
#plt.xlim(250,256)#plt.xlim(500,512)
#plt.xlim(0,0.2)
plt.ylim(-0.01,6.3)
plt.ylabel("phase (rads)")
plt.xlabel("time")
plt.show()
#%%
plt.plot(np.transpose(phases[:,:]),".-")
#plt.plot(t[0:N_t],np.transpose(phases[5 ,:]),".-")
#plt.plot(t[0:N_t],np.transpose(phases[10,:]),"-")
#plt.plot(t[0:N_t],np.transpose(phases[15,:]),"-")
#plt.xlim(2030,2048)
#plt.xlim(0,0.001)
#plt.ylim(0,0.2)
plt.ylabel("phase (rads)")
plt.xlabel("time")
plt.show()
#%%
#################################################################
########  CALCULATE EFFECTIVE HAMILTONIAN #######################
#################################################################
# WITH THE FOLDED SPECTRUM
# Effective Hamiltonian in the basis Floquet states
H_eff_x   = FEH.EffectiveFloquetHamiltonian(e_u,U,U_x)
H_eff_x_u = FEH.EffectiveFloquetHamiltonian(e_u_unfolded,U,U_x)


plt.plot(x,np.diag(H_eff_x/dx) - np.min(np.diag(H_eff_x/dx)),"-")
plt.plot(x,np.diag(H_eff_x_u/dx) - np.min(np.diag(H_eff_x_u/dx)))
plt.ylabel('V(x)', fontsize=15)
plt.xlabel('x', fontsize=15)
#plt.xlim(10,20)
plt.show()

plt.contourf(x,x,np.abs(H_eff_x))
plt.ylabel("x'", fontsize=16)
plt.xlabel("x", fontsize=16)
plt.colorbar()
#plt.xlim(1000,1200)
#plt.ylim(1000,1200)
plt.show()

plt.contourf(x,x,np.abs(H_eff_x_u))
plt.ylabel("x'", fontsize=16)
plt.xlabel("x", fontsize=16)
#plt.xlim(1000,1200)
#plt.ylim(1000,1200)
plt.colorbar()
plt.show()
#%%
plt.plot(BlochSpectrum-np.min(BlochSpectrum),".-")
plt.plot(np.sort(e_u),".-")
#plt.plot(e_u+50)
plt.plot(np.sort(e_u_unfolded),".-")
#plt.xlim(0,15)
plt.ylim(0,92)
plt.xlabel("Bloch state (index)")
plt.ylabel("Energy (omega)")
plt.show()

#%%
# saving:
#f = open("UnfoldingFloquetL32Nt2048_N_Band3.dat", "w")
#f.write("# phi_t[8:256] time evolution of the phase\n#eigenvalues\n#eigenenergies#\n#transformation\n")        # column names
#np.savetxt(f, phi_t)
#np.savetxt(f, lambda_u.T)
#np.savetxt(f, e_u.T)
#np.savetxt(f, U)

#f.close()
#%%
f = open("UnfoldingFloquetL32Nt2048_N_Band3.dat", "r")

phi_t    = np.loadtxt(f, dtype=complex,max_rows=N_Bands*L,unpack=True).T
lambda_u = np.loadtxt(f, dtype=complex,max_rows=N_Bands*L,unpack=True)
e_u      = np.loadtxt(f, dtype=float,  max_rows=N_Bands*L,unpack=True)
U        = np.loadtxt(f, dtype=complex,max_rows=N_Bands*L,unpack=True).T

f.close()