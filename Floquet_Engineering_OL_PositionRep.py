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
import time 

import OpticalLatticeModule as OLM
import FloquetEffectiveHamiltonian as FEH
import Bloch_wave as BW

#from OpticalLatticeModule import Bloch_spectrum as BS

jj = np.complex(0.0,1.0)
            
# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) cos(2*k_x  x  +  2.0*phase)
V_0      = 20.0 # ~ Band gap
phase_x  = 0.0     #np.pi/3.0
omega    = 5.0#2.0*20.0   # the secon factor is the frequency in scaled units                      
phase_t  = 0.0

OL    = OLM.OpticalLattice(V_0,phase_x)#OpticalLatticeModule.OpticalLattice(V_0,k_x,phase)

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 3  # Number of bands to evaluate
L     = 32 # number of unit cells
N_x   = 32 # Number of points in a unit cell
x     = np.linspace(0, L, N_x*L)

BlochSpectrum,RealSpaceBlochWavefun = BW.BlochSpectrum(L,N_x,Bands,V_0,2.0*phase_x)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceBlochWavefun[:, i], N_x)

plt.plot(k_u,BlochSpectrum,".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 1   # number of bands to include in the Schrodinger equation
N_t     = 512 # number of time steps

t0   = 0                 # initial time
DT   = 2.0*np.pi/(omega) # total time to integrate
t1   = t0 + DT           # final time #
dt   = DT/N_t            # time step
t    = np.linspace(t0,t0+DT, N_t) 
U_x = RealSpaceBlochWavefun[:,0:N_Bands*L]
U_T = np.zeros([N_Bands*L,N_Bands*L],dtype=np.complex)
   
#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# TO DO:
# The integrator is too slow, it requires a large number of time steps (> 4096)
# for an accurate time-evolution (~10^-3) 
# Find an alternative: maybe https://pydstool.github.io/PyDSTool/Tutorial/Tutorial_Calcium.html
# http://hplgit.github.io/odespy/doc/pub/tutorial/html/main_odespy.html
start = time.time()
phi_t,U_T,U_x = FEH.FloquetStroboscopicOperatorV3(L,N_Bands,t0,DT,N_t,x,OL,BlochSpectrum,RealSpaceBlochWavefun)
lambda_u,U,e_u,H_eff_kn,H_eff_x = FEH.EffectiveFloquetHamiltonian(U_T,DT,U_x)
end = time.time()
print(end - start)
#%%
phases,folding_counter,phase_sec_order,phase_sec_order_,phase_sec_order_aux = FEH.UnfoldingFloquetSpectrum(L*N_Bands,dt,N_t,phi_t)

            
#grad_new_ = np.zeros([L,N_t],dtype=np.float32)
#for i in range(phi_t.shape[1]-2):   
#        grad_new_[:,i+1] = phi_t[:,i+1] - phi_t[:,i]    
#    
#folding_counter = np.zeros([phi_t.shape[0]],dtype=np.int32)
#for i in range(phi_t.shape[0]):   
#        folding_counter[i] = np.array(np.where(grad_new_[i,:]<-np.pi)).size
 
plt.plot(np.transpose(phi_t[:,:]),".")
plt.show()

plt.plot(np.transpose(phases[:,:]),"-")
plt.show()
#%%

# WITH THE FOLDED SPECTRUM
# Effective Hamiltonian in the basis Floquet states
H_eff_F = np.diag(e_u) 

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))

plt.plot(x,np.diag(H_eff_x))
plt.ylabel('V(x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.show()

plt.contourf(np.abs(H_eff_x))
plt.ylabel('y', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.colorbar()
plt.show()
#%%
# WITH THE UNFOLDED SPECTRUM
# Effective Hamiltonian in the basis Floquet states
e_u_unfolded = e_u + folding_counter
H_eff_F = np.diag(e_u_unfolded) 

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))

plt.plot(x,np.diag(H_eff_x))
plt.ylabel('V(x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.show()

plt.contourf(np.abs(H_eff_x))
plt.ylabel('y', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.colorbar()
plt.show()























#%%
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]))
#plt.plot(e_u_unfolded,".")
#plt.plot(e_u,"-")
#plt.plot(e_u+1,"-")
#plt.plot(e_u+2,"-")
#plt.plot(e_u+3,"-")
#plt.plot(e_u+4,"-")
#plt.plot(e_u+5,"-")
plt.plot(e_u_unfolded,".")
#plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]),linewidth=4)
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()


#%%

H_eff_F = np.diag(e_u_) # Effective Hamiltonian in the basis Floquet states

plt.contourf(np.abs(H_eff_F))
plt.show()   

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U
plt.contourf(np.abs(H_eff_kn))
plt.colorbar()
plt.show()   


plt.plot(BlochSpectrum[0:N_Bands*L])
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()


plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]))
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]])
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+7*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+6*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+5*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+4*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+3*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+2*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]-omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]-2*omega)
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]),linewidth=4)
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.plot(x,np.diag(H_eff_x))
plt.ylabel('V(x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.show()


H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.contourf(np.abs(H_eff_x))
plt.ylabel('y', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.colorbar()
plt.show()

#%%
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "--")
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+12*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+11*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+10*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+9*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+8*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+7*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+6*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+5*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+4*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+3*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+2*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]])
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "-")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()
#%%


phases,swaps,phase_unfolded,phase_sec_order,phase_sec_order_,phase_sec_order_aux = FEH.UnfoldingFloquetSpectrum(L,dt,N_t,phi_t)


plt.plot(np.transpose(phases[31,:]),"-")
#plt.xlim(1013,1017)
#plt.ylim(0,0.1)
plt.show()
#%%
plt.plot(np.transpose(phases[:,:]),"-")
plt.xlim(1013,1017)
plt.ylim(0,0.1)
plt.show()

plt.plot(np.transpose(phase_sec_order[:,:]),"-")
plt.xlim(1013,1017)
plt.ylim(0,0.1)
plt.show()


plt.plot(np.transpose(phases[:,:]),"-")
plt.xlim(1013,1017)
plt.ylim(6.2,6.3)
plt.show()

plt.plot(np.transpose(phase_sec_order[:,:]),"-")
plt.xlim(1013,1017)
plt.ylim(6.2,6.3)
plt.show()

#plt.plot(np.transpose(phase_sec_order[:,:]),"-")
#plt.xlim(390,400)
#plt.ylim(6,6.3)
#plt.show()
#plt.plot(np.transpose(phase_sec_order_[:,:]),"-")
#plt.show()
#plt.plot(np.transpose(phase_sec_order_aux[:,:]),"-")
#plt.show()
#plt.xlim(1010,1024)
#plt.ylim(5.2,5.5)

#plt.ylabel('phase', fontsize=14)
#plt.xlabel('time', fontsize=14)
#plt.show()

#%%
# saving:
f = open("UnfoldingFloquetTEST.dat", "w")
#f.write("# phi_t[8:256] time evolution of the phase\n#eigenvalues\n#eigenenergies#\n#transformation\n")        # column names
np.savetxt(f, phi_t)
np.savetxt(f, lambda_u.T)
np.savetxt(f, e_u.T)
np.savetxt(f, U)

f.close()
#%%
f = open("UnfoldingFloquetTEST.dat", "r")

phi_t    = np.loadtxt(f, dtype=complex,max_rows=L,unpack=True).T
lambda_u = np.loadtxt(f, dtype=complex,max_rows=L,unpack=True)
e_u      = np.loadtxt(f, dtype=float,max_rows=L,unpack=True)
U        = np.loadtxt(f, dtype=complex,max_rows=L,unpack=True).T
#phase_sec_order_aux_ = np.loadtxt(f, dtype=complex,max_rows=511,unpack=True)


#%%
#plt.plot(np.transpose(phases[8,:]), "-")
plt.plot(np.transpose(phi_t[:,:]),"-")
plt.show()
plt.plot(np.transpose(phase_sec_order_aux[:,:]),"-")
#plt.plot(np.transpose(phase_sec_order_aux[4,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[5,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[22,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[19,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[6,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[12,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[8,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[4,i_l:i_r]),".")
#plt.xlim(630,640)
#plt.xlim(790,1024)
#plt.ylim(5,5.6)
plt.ylabel('phase', fontsize=14)
plt.xlabel('time', fontsize=14)
plt.show()




#%%

plt.contourf(np.abs(np.transpose(np.conjugate(U_x))@U_x))
plt.colorbar()
plt.show()

plt.contourf(np.abs(U_x))
plt.show()

e_u_unfolded = phase_unfolded/DT
H_eff_F = np.diag(e_u_unfolded) # Effective Hamiltonian in the basis Floquet states

plt.contourf(np.abs(H_eff_F))
plt.show()   

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U
plt.contourf(np.abs(H_eff_kn))
plt.colorbar()
plt.show()   

#plt.plot(np.diag(np.real(H_eff_kn))[0:e_u.shape[0]])
plt.plot(BlochSpectrum[0:N_Bands*L])
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()


plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]))
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]])
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+7*omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+6*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+5*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+4*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+3*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+2*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]-omega)
#plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]-2*omega)
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]),linewidth=4)
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.plot(x,np.diag(H_eff_x))
plt.ylabel('V(x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.show()


H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
plt.contourf(np.abs(H_eff_x))
plt.ylabel('x', fontsize=15)
plt.xlabel('y', fontsize=15)
plt.colorbar()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
# saving:
#f = open("UnfoldingFloquetTest.dat", "w")
#f.write("# phi_t[32:8192] time evolution of the phase\n#eigenvalues\n#eigenenergies#\n#transformation\n")        # column names
#np.savetxt(f, phi_t.T)
#np.savetxt(f, lambda_u.T)
#np.savetxt(f, e_u.T)
#np.savetxt(f, U)
# loading:
f = open("UnfoldingFloquetTest.dat", "r")

phi_t    = np.loadtxt(f, dtype=complex,max_rows=8192,unpack=True)
lambda_u = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)
e_u      = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)
U        = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)

time = np.linspace(1,8192,8192)
N_t_0 = 0
N_t_  = 1024
#plt.plot(time[N_t_0:N_t_],phi_t[0,N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[1,N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[2,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[3,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[4,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[5,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[6,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[7,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[8,N_t_0:N_t_])
plt.show()

#%%
#plt.plot(e_u+11*omega)#
#plt.plot(e_u+10*omega)
#plt.plot(e_u+9*omega)
#plt.plot(e_u+8*omega)
#plt.plot(e_u+7*omega)
#plt.plot(e_u+6*omega)
#plt.plot(e_u+5*omega)
#plt.plot(e_u+4*omega)
plt.plot(e_u+3*omega)
plt.plot(e_u+2*omega)
plt.plot(e_u+omega)
plt.plot(e_u)
plt.plot(e_u-omega)
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "-", linewidth=4)
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#%%
plt.plot(e_u)
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "-")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#plt.plot(abs(U[:,0]))
plt.plot(abs(U[:,1]))
plt.plot(abs(U[:,2]))
plt.plot(abs(U[:,3]))
plt.plot(abs(U[:,4]))
plt.plot(abs(U[:,5]))

plt.show()