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

#plt.plot(k_u[0:32],BlochSpectrum[0:32],".")
plt.plot(k_u,BlochSpectrum,".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()

#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 1# number of bands to include in the Schrodinger equation
#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 2.0*np.pi/(omega) # total time to integrate
t1   = t0 + DT # final time #
N_t  = 1024#8192  # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 


E_0 = BlochSpectrum[0:N_Bands*L]
E_0 = E_0 - np.min(E_0)
U_x = RealSpaceBlochWavefun[:,0:N_Bands*L]
plt.contourf(np.log(np.abs(np.transpose(np.conjugate(U_x))@U_x)));
plt.colorbar()
plt.show()
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

#E_0 = np.linspace(1,int(N_Bands*L),int(N_Bands*L))#BlochSpectrum[0:int(N_Bands*L)]

#print(np.diag(U_T))
#print(np.exp(jj*E_0*DT))
lambda_u,U,e_u,H_eff_kn,H_eff_x = FEH.EffectiveFloquetHamiltonian(U_T,DT,U_x)

#V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)

end = time.time()
print(end - start)

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
plt.ylabel('y', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.colorbar()
plt.show()

#plt.plot(x[0:4*N_x],np.diag(H_eff_x)[0:4*N_x])
#plt.ylabel('V_eff(x)', fontsize=18)
#plt.xlabel('x', fontsize=16)
#plt.show()


#plt.plot(x,np.diag(H_eff_x))
#plt.show()
#%%

plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "--")
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+12*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+11*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+10*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+9*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+8*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+7*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+6*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+5*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+4*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+3*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+2*omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]]+omega)
plt.plot(np.diag(np.real(H_eff_F))[0:e_u.shape[0]])
plt.plot(BlochSpectrum[0:N_Bands*L]-np.min(BlochSpectrum[0:N_Bands*L]), "-")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)
plt.show()
#%%
#N_t = 8192
time = np.linspace(1,N_t,N_t)
N_t_0 = 0 #8160
N_t_  = N_t-1#127 #8191



phase_sec = phi_t[:,N_t_0:N_t_]

#plt.plot(time[N_t_0:N_t_],phi_t[0,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[1,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[2,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[3,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[4,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[5,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[6,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[7,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[8,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[9,N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[10,N_t_0:N_t_])
#plt.ylabel('phase', fontsize=18)
#plt.xlabel('time', fontsize=16)
#plt.show()

phase_sec[0,:]
phase_sec_=phase_sec
plt.plot(np.transpose(phase_sec[:,:]))
#plt.plot(phase_sec[1,:])
#plt.plot(phase_sec[2,:])
#plt.plot(phase_sec[3,:])
#plt.plot(phase_sec[4,:])
#plt.plot(phase_sec[5,:])
#plt.plot(phase_sec[6,:])
#plt.plot(phase_sec[7,:])
plt.xlim(4720,4730)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()

#print(np.argsort(np.abs(phase_sec[:,0])))
#print(np.argsort(np.abs(phase_sec[:,11])))
index_order_old = np.argsort(np.abs(phase_sec[:,0]))
phase_sec_order  = np.zeros(phase_sec.shape)
phase_sec_order_ = np.zeros(phase_sec.shape)
phase_sec_ = phase_sec[index_order_old,:]
phase_sec = phase_sec_    
index_order_old = np.argsort(np.abs(phase_sec[:,0]))

i_l = 0
i_r = N_t

flag = 0
kinks = []
order_ = np.empty([0,L],dtype=np.int)
swap_counter= np.empty([0],dtype=np.int)
r = 1
for i in range(phase_sec.shape[1]):
    swap_flag = False
    index_order_new = np.argsort(np.abs(phase_sec[:,i]))
    #if(i>530 and i<550): 
        #print(i,index_order_old)
        #print(i,index_order_new)
    #    grad_ = (phase_sec[:,i-1] - phase_sec[:,i-2])/dt
    #    phase_ = np.real(grad_*dt + phase_sec[:,i-1])
    #    print(np.real(phase_sec[:,i-2]))
    #    print(np.real(phase_sec[:,i-1]))
     #   print(np.real(phase_sec[:,i]))
        #print(i,np.any(phase_>2.0*np.pi))
    #    if(np.any(phase_>2.0*np.pi) and (index_order_new[0] != index_order_old[0])):# and ((index_order_new[0] == index_order_old[L-1]))):
    #        print(i,index_order_old)            
     #       print(i,index_order_new)
      #      print(phase_>2.0*np.pi)
    if(index_order_new[0] != index_order_old[0]):
    #if(index_order_new[0] == index_order_old[L-1]):
        index_ = index_order_old[L-1]
        grad_last = (phase_sec[index_,i-1] - phase_sec[index_,i-2])/dt
        phase_future = np.real(grad_last*dt + phase_sec[index_,i-1])
        grad_ = (phase_sec[:,i-1] - phase_sec[:,i-2])/dt
        phase_ = np.real(grad_*dt + phase_sec[:,i-1])
        print(i,phase_future)
        if(phase_future > 2.0*np.pi):
            swap_flag= True
            #print("%3i %7.3f %7.3f %7.3f %7.3f %7.3f " %(i,np.abs(phase_sec[index_,i-2]) , np.abs(phase_sec[index_,i-1]), np.abs(phase_sec[index_,i]),phase_future,2.0*np.pi))
            if(index_order_new[0] == index_order_old[L-1]):
                kinks = np.append(kinks,[np.int(i)])
                order_ = np.append(order_,index_order_new)
                swap_counter = np.append(swap_counter,np.array(np.where(phase_>2.0*np.pi)).size)

            if(index_order_new[0] != index_order_old[L-1]):
                kinks = np.append(kinks,[np.int(i)])
                order_ = np.append(order_,[index_order_new])
                swap_counter = np.append(swap_counter,np.array(np.where(phase_>2.0*np.pi)).size)

        #r += 1
        #if(r == 7): r = 0
    #if(swap_flag==False): 
    phase_sec_order[:,i] = np.sort(np.abs(phase_sec[:,i]))
    index_order_old = index_order_new
    
#order_ = np.reshape(order_,[kinks.size,L])

plt.plot(np.transpose(phase_sec[:,i_l:i_r]))
#plt.plot(phase_sec[1,i_l:i_r])
#plt.plot(phase_sec[2,i_l:i_r])
#plt.plot(phase_sec[3,i_l:i_r])
#plt.plot(phase_sec[4,i_l:i_r])
#plt.plot(phase_sec[5,i_l:i_r])
#plt.plot(phase_sec[6,i_l:i_r])
#plt.plot(phase_sec[7,i_l:i_r])
#plt.plot(phase_sec[8,i_l:i_r])
#plt.plot(phase_sec[9,i_l:i_r])
#plt.plot(phase_sec[10,i_l:i_r])
#plt.plot(phase_sec[11,i_l:i_r])
#plt.plot(phase_sec[12,i_l:i_r])
#plt.plot(phase_sec[13,i_l:i_r])
#plt.plot(phase_sec[14,i_l:i_r])
#plt.plot(phase_sec[15,i_l:i_r])
#plt.plot(phase_sec[16,i_l:i_r])
#plt.plot(phase_sec[17,i_l:i_r])
#plt.xlim(4700,4750)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()

phase_sec_order_ = np.zeros(phase_sec_order.shape)
phase_sec_order_[:] = phase_sec_order[:]
i_l = 0
i_r = N_t
plt.plot(np.transpose(phase_sec_order[:,i_l:i_r]))
#plt.plot(phase_sec_order[1,i_l:i_r])
#plt.plot(phase_sec_order[2,i_l:i_r])
#plt.plot(phase_sec_order[3,i_l:i_r])
#plt.plot(phase_sec_order[4,i_l:i_r])
#plt.plot(phase_sec_order[5,i_l:i_r])
#plt.plot(phase_sec_order[6,i_l:i_r])
#plt.plot(phase_sec_order[7,i_l:i_r])
#plt.plot(phase_sec_order[8,i_l:i_r])
#plt.plot(phase_sec_order[9,i_l:i_r])
#plt.plot(phase_sec_order[10,i_l:i_r])
#plt.plot(phase_sec_order[11,i_l:i_r])
#plt.plot(phase_sec_order[12,i_l:i_r])
#plt.plot(phase_sec_order[13,i_l:i_r])
#plt.plot(phase_sec_order[14,i_l:i_r])
#plt.plot(phase_sec_order[15,i_l:i_r])
#plt.plot(phase_sec_order[16,i_l:i_r])
#plt.plot(phase_sec_order[17,i_l:i_r])
#plt.xlim(4700,4800)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()


#%%
#l = 0
#for i in [0]:#,1]:#range(kinks.size):
    #print(index_order_old,index_order_new)
    #print(np.int(kinks[i]))
    # l += 1
    #i_l = np.int(kinks[i])
    #i_r = np.int(kinks[i+1])
#    phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l,axis=0)
    
    #phase_sec_order_[0,i_l:i_r] = phase_sec_order[1,i_l:i_r]
    #phase_sec_order_[1,i_l:i_r] = phase_sec_order[2,i_l:i_r]
    #phase_sec_order_[2,i_l:i_r] = phase_sec_order[3,i_l:i_r]
    #phase_sec_order_[3,i_l:i_r] = phase_sec_order[4,i_l:i_r]
    #phase_sec_order_[4,i_l:i_r] = phase_sec_order[5,i_l:i_r]
    #phase_sec_order_[5,i_l:i_r] = phase_sec_order[6,i_l:i_r]
    #phase_sec_order_[6,i_l:i_r] = phase_sec_order[0,i_l:i_r]
    
    #i_l = np.int(kinks[i+1])
    #i_r = np.int(kinks[i+2])#phase_sec_order.shape[1]
 #   phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-2,axis=0)
    #phase_sec_order_[0,i_l:i_r] = phase_sec_order[2,i_l:i_r]
    #phase_sec_order_[1,i_l:i_r] = phase_sec_order[3,i_l:i_r]
    #phase_sec_order_[2,i_l:i_r] = phase_sec_order[4,i_l:i_r]
    #phase_sec_order_[3,i_l:i_r] = phase_sec_order[5,i_l:i_r]
    #phase_sec_order_[4,i_l:i_r] = phase_sec_order[6,i_l:i_r]
    #phase_sec_order_[5,i_l:i_r] = phase_sec_order[0,i_l:i_r]
    #phase_sec_order_[6,i_l:i_r] = phase_sec_order[1,i_l:i_r]

    #i_l = np.int(kinks[i+2])
    #i_r = np.int(kinks[i+3])#phase_sec_order.shape[1]
  #  phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-3,axis=0)

    #i_l = np.int(kinks[i+3])
    #i_r = phase_sec_order.shape[1]
   # phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-4,axis=0)
#%%    

#kinks = []
#kinks = np.array([ 722.,  783.,  787.,  793.,  808.,  837.,  840.,  842.,  853.,
#        895.,  935.,  1294., 1354., 1355., 1444.,
#       1567., 1575., 1584., 1587., 1616., 1675., 1680., 1684., 1706.,
#       1790., 1871.])

l=0#
i_l = 0
for i in range(kinks.size-1):
    l += 1
    i_r = np.int(kinks[i])
    phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l,axis=0)
    i_l = i_r
    
if(kinks.size == 1):
    i_r = np.int(kinks[0])


#%%
i_l = i_r
i_r = phase_sec_order.shape[1]
phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l-1,axis=0)

plt.plot(np.transpose(phase_sec[:,:]))
#plt.plot(phase_sec[1,:])
#plt.plot(phase_sec[2,:])
#plt.plot(phase_sec[3,:])
#plt.plot(phase_sec[4,:])
#plt.plot(phase_sec[5,:])
#plt.plot(phase_sec[6,:])
#plt.plot(phase_sec[7,:])
#plt.plot(phase_sec[8,:])
#plt.plot(phase_sec[9,:])
#plt.plot(phase_sec[10,:])
#plt.plot(phase_sec[11,:])
#plt.plot(phase_sec[12,:])
#plt.plot(phase_sec[13,:])
#plt.xlim(400,N_t)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()

plt.plot(np.transpose(phase_sec_order[:,:]))
#plt.plot(phase_sec_order[1,:])
#plt.plot(phase_sec_order[2,:])
#plt.plot(phase_sec_order[3,:])
#plt.plot(phase_sec_order[4,:])
##plt.plot(phase_sec_order[5,:])
#plt.plot(phase_sec_order[6,:])
#plt.plot(phase_sec_order[7,:])
#plt.plot(phase_sec_order[8,:])
#plt.plot(phase_sec_order[9,:])
#plt.plot(phase_sec_order[10,:])
#plt.plot(phase_sec_order[11,:])
#plt.plot(phase_sec_order[12,:])
#plt.plot(phase_sec_order[13,:])
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
#plt.xlim(400,N_t)

plt.show()

plt.plot(np.transpose(phase_sec_order_[:,:]), "-")
#plt.plot(phase_sec_order_[1,:])
#plt.plot(phase_sec_order_[2,:])
#plt.plot(phase_sec_order_[3,:])
#plt.plot(phase_sec_order_[4,:])
#plt.plot(phase_sec_order_[5,:])
#plt.plot(phase_sec_order_[6,:])
#plt.plot(phase_sec_order_[7,:])
#plt.plot(phase_sec_order_[8,:])
#plt.plot(phase_sec_order_[9,:])
#plt.plot(phase_sec_order_[10,:])
#plt.plot(phase_sec_order_[11,:])
#plt.plot(phase_sec_order_[12,:])
#plt.plot(phase_sec_order_[13,:])
plt.xlim(600,800)
#plt.ylim(-0.1,0.8)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()


plt.plot(np.transpose(phase_sec_order_aux[:,:]),"-")
#plt.plot(phase_sec_order_aux[1,i_l:i_r])
plt.xlim(600,800)
plt.ylabel('phase', fontsize=14)
plt.xlabel('time', fontsize=14)
plt.show()

#%%
counter = 0
phase_sec_order_aux = np.copy(phase_sec_order_)
phase_sec_order__   = np.copy(phase_sec_order_)
grad_old = phase_sec_order_[:,1] - phase_sec_order_[:,0]
index_order_old = np.argsort(np.abs(grad_old))
#j = np.ones([phase_sec_order_.shape[0],phase_sec_order_.shape[0]],dtype=np.float64)   
swap_time_old = [0,0,0]
swap_time_new = [0,0,0]
for i in range(phase_sec_order_aux.shape[1]-2):   
    grad_new = phase_sec_order_aux[:,i+2] - phase_sec_order_aux[:,i+1]    
    dgrad = grad_new-grad_old
    index_order_old = np.argsort(np.abs(grad_old))
    index_order_new = np.argsort(np.abs(grad_new))
    #phase_difference = (np.multiply(j,phase_sec_order_[:,i+2]) - np.transpose(np.multiply(j,phase_sec_order_[:,i+2]))-(np.multiply(j,phase_sec_order_[:,i+1]) - np.transpose(np.multiply(j,phase_sec_order_[:,i+1]))))
     #  COMPARE GRADIENTS. IF THEY ARE DIFFERENT, THERE WAS A COLLISION
    if(i+2>4725 and i+2 <4731):
        print(i+2,np.linalg.norm(grad_new-grad_old))
        print(grad_old)
        print(grad_new)
        #print(np.where(np.abs(dgrad)>1E-6))
        print(dgrad)
        #swaps_counter = (np.where(np.abs(dgrad)>1E-6))
        
    swap_flag = False
    if(np.linalg.norm(grad_new-grad_old)<1.999*np.pi): 
        swap_flag = True
        swap_index     = np.argsort(np.abs(dgrad))[L-2:L]
        
    if(np.linalg.norm(grad_new-grad_old)>1.999*np.pi): 
        swaps_counter = np.array((np.where(np.abs(dgrad)>1E-6)))[0,:]
        if(swaps_counter.size>1 and np.mod(swaps_counter.size,2) == 1):            
            #print(i+2,np.linalg.norm(grad_new-grad_old),np.mod(swaps_counter.size,2),swaps_counter.size,swaps_counter)
            #print(dgrad)   
            if(swaps_counter.size>1):
                swap_flag      = True
                swap_index     = np.argsort(np.abs(grad_new-grad_old))[L-3:L-1]
            #print(swap_index)
        
    #if(np.linalg.norm(grad_new-grad_old)>5.6E-4 and swap_flag == True):
    if(np.linalg.norm(grad_new-grad_old)>5.6E-4 and np.linalg.norm(grad_new-grad_old)<1.999*np.pi):
        # THE COLLIDING PARTICLES CHANGE THEIR GRADIENT THE MOST
        #swap_index     = np.argsort(np.abs(grad_new-grad_old))[L-2:L]
        swap_time_new  = [i+2,swap_index[0],swap_index[1]]
        #print(swap_time_old)
        #print(swap_time_new)
        if(swap_time_new[0] == swap_time_old[0]+1 and swap_time_new[2]==swap_time_old[1] and swap_time_new[1]==swap_time_old[2]):
            #print("do nothing")
            a=1
        else:
            #print(i+2,swap_index)
            phase_sec_order_aux[swap_index[0],i+1::] = np.copy(phase_sec_order__[swap_index[1],i+1::]) 
            phase_sec_order_aux[swap_index[1],i+1::] = np.copy(phase_sec_order__[swap_index[0],i+1::]) 
            grad_new = phase_sec_order_aux[:,i+2] - phase_sec_order_aux[:,i+1]
            phase_sec_order__ = np.copy(phase_sec_order_aux)
            swap_time_old  = swap_time_new
    grad_old = grad_new


grad_new_ = np.zeros([L,N_t],dtype=np.float32)
for i in range(phase_sec_order_aux.shape[1]-2):   
    grad_new_[:,i+1] = phase_sec_order_aux[:,i+1] - phase_sec_order_aux[:,i]    

i_l = 0
i_r = N_t


plt.plot(np.transpose(phase_sec_order_aux[:,i_l:i_r]),"-")
#plt.plot(phase_sec_order_aux[1,i_l:i_r])
#plt.xlim(300,512)
plt.ylabel('phase', fontsize=14)
plt.xlabel('time', fontsize=14)
plt.show()

#%%

i_l = 0
i_r = N_t


plt.plot(np.transpose(phase_sec_order_aux[0,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[4,i_l:i_r]),".")
plt.plot(np.transpose(phase_sec_order_aux[8,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[3,i_l:i_r]),".")
#plt.plot(np.transpose(phase_sec_order_aux[13,i_l:i_r]),".")
plt.plot(np.transpose(phase_sec_order_aux[12,i_l:i_r]),".")
#plt.plot(phase_sec_order_aux[1,i_l:i_r])
plt.xlim(296,330)
plt.ylim(0,6.3)
plt.ylabel('phase', fontsize=14)
plt.xlabel('time', fontsize=14)
plt.show()

#%%
# saving:
#f = open("UnfoldingFloquetL32N_t1024.dat", "w")
#f.write("# phi_t[32:8192] time evolution of the phase\n#eigenvalues\n#eigenenergies#\n#transformation\n")        # column names
#np.savetxt(f, phi_t.T)
#np.savetxt(f, lambda_u.T)
#np.savetxt(f, e_u.T)
#np.savetxt(f, U)
#np.savetxt(f, phase_sec_order_aux)
#%% 
f = f = open("UnfoldingFloquetL32N_t1024.dat", "r")

phi_t    = np.loadtxt(f, dtype=complex,max_rows=32768,unpack=True)
lambda_u = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)
e_u      = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)
U        = np.loadtxt(f, dtype=complex,max_rows=32,unpack=True)
phase_sec_order_aux_ = np.loadtxt(f, dtype=complex,max_rows=32736,unpack=True)
#%%
phases = np.copy(phase_sec_order_aux)
for j_ in [25]:#range(L):
    i  = N_t-2      
    grad_ref = phase_sec_order_aux[j_,i] - phase_sec_order_aux[j_,i-1]
    
    for i in range(N_t-2):#range(phase_sec_order_aux.shape[1]-2):   
        grad_new   = phases[j_,N_t-2-(i+1)]   - phases[j_,N_t-2-(i+2)]  
        grad_new_2 = phases[j_,N_t-2-(i+1)-1] - phases[j_,N_t-2-(i+2)-1]  
        # check if there is a jump in teh gradient
        if(np.abs(grad_new-grad_ref) > 1E-7): 
            # print(510-(i+1),grad_ref,grad_new)
            # check if the jump is negative and it's NOT a folding
            if(grad_new<0 and np.abs(grad_new)<1.9*np.pi):
                print("A",N_t-2-(i+1),grad_ref,grad_new)
                #Evaluate gradients after the jump
                grad_new_ = phase_sec_order_aux[:,N_t-2-(i-2)] - phase_sec_order_aux[:,N_t-2-(i-3)]    
                #Identify the phase index with the gradient closest to the previous     one
                #print(np.abs(grad_new_-grad_ref))
                #print(np.argmin(np.abs(grad_new_-grad_ref)))
                #print(phase_sec_order_aux[:,510-(i+1)-1] - phases[0,510-(i+1)])
                #print(np.argmin(np.abs(phase_sec_order_aux[:,510-(i+1)-1] - phases[0,510-(i+1)])))
                new_index_grad = np.argmin(np.abs(grad_new_-grad_ref))            
                new_index_value = np.argmin(np.abs(phase_sec_order_aux[:,N_t-2-(i+1)-1] - phases[j_,N_t-2-(i+1)]))
                print(new_index_grad,new_index_value)
                #print(phase_sec_order_aux[new_index_grad,N_t-2-(i+1)-1] - phases[0,N_t-2-(i+1)],
                #phase_sec_order_aux[new_index_value,N_t-2-(i+1)-1] - phases[0,N_t-2-(i+1)])
                if(new_index_grad==new_index_value):
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_grad,0:N_t-2-(i+1)])
                        #grad_ref = grad_new_[new_index_grad]
                else:
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_value,0:N_t-2-(i+1)])

                #phases[j_,0:N_t-3-i] = np.copy(phase_sec_order_aux[new_index_grad,0:N_t-3-i])
                #grad_ref = grad_new_[new_index_grad]
                # check if the jump is positive
            if(grad_new>0):
                    print("B",N_t-2-(i+1),grad_ref,grad_new,grad_new_2)
                    #Evaluate gradients after the jump
                    grad_new_ = phase_sec_order_aux[:,N_t-4-i] -    phase_sec_order_aux[:,N_t-5-i]
                    grad_new_2 = phase_sec_order_aux[:,N_t-5-i] - phase_sec_order_aux[:,N_t-6-i]
                    new_value = -grad_ref + phases[j_,N_t-2-(i+1)]
                    #Identify the phase index with the gradient closest to the previous one
                    print(grad_new_)
                    #print(grad_new_)
                    print(np.argmin(np.abs(grad_new_-grad_ref)))
                    #print(phase_sec_order_aux[:,N_t-2-(i+1)-1] - phases[0,510-(i+1)])
                    #print(phase_sec_order_aux[:,N_t-2-(i+1)-1], phases[0,510-(i+1)])
                    print(phase_sec_order_aux[:,N_t-2-(i+1)-1], new_value)
                    print(np.argmin(np.abs(phase_sec_order_aux[:,N_t-2-(i+1)-1] -   new_value)))
                    new_index_grad = np.argmin(np.abs(grad_new_-grad_ref))
                    new_index_value = np.argmin(np.abs(phase_sec_order_aux[:,N_t-2-(i+1)-1] - new_value))
                    if(new_index_grad==new_index_value):
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_grad,0:N_t-2-(i+1)])
                        #grad_ref = grad_new_[new_index_grad]
                    else:
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_value,0:N_t-2-(i+1)])
                        #grad_ref = grad_new_[new_index_grad]

        #else:            
            #grad_ref = np.copy(grad_new)

#%%
plt.plot(np.transpose(phases[25,:]), "-")
#plt.plot(np.transpose(phase_sec_order_aux[2,i_l:i_r]),".")
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
#phase_sec_order_aux = np.copy(phase_sec)
grad_new_ = np.zeros([L,N_t],dtype=np.float32)
for i in range(phase_sec_order_aux.shape[1]-2):   
    grad_new_[:,i+1] = phase_sec_order_aux[:,i+1] - phase_sec_order_aux[:,i]    
    
plt.plot(grad_new_[0,:])
plt.plot(grad_new_[1,:])
plt.plot(grad_new_[2,:])
plt.plot(grad_new_[3,:])
plt.plot(grad_new_[4,:])
plt.plot(grad_new_[5,:])
plt.plot(grad_new_[6,:])
#plt.xlim(1000,1200)
#plt.ylim(0,3E-3)

plt.show()
#plt.plot(grad_new_[6,:])
swaps_counter = np.zeros([phase_sec_order_aux.shape[0]],dtype=np.int32)
for i in range(phase_sec_order_aux.shape[0]):   
    swaps_counter[i] = np.array(np.where(grad_new_[i,:]<-np.pi)).size
print(swaps_counter)

phase_unfolded = np.copy(phase_sec_order_aux[:,N_t-2]) + 2*np.pi*swaps_counter





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
r=0
kinks_ = []
for i in range(kinks.size):
    r += 1
    if(r == 7): r = 1
    index_=np.int(kinks[i])
    folding_threshold = np.abs(phase_sec_order_[L-r,index_-1]+phase_sec_order_[L-r,index_]-2*np.pi)
    print(r,index_,np.abs(phase_sec_order_[L-r,index_-1]+phase_sec_order_[L-r,index_]-2*np.pi))
    if(folding_threshold<0.01):
        kinks_ = np.append(kinks_,[np.int(kinks[i])])


l=0
for i in range(kinks_.size-1):
    #print(index_order_old,index_order_new)
    #print(np.int(kinks_[i]))
    l += 1
    i_l = np.int(kinks_[i])
    i_r = np.int(kinks_[i+1])
    phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l,axis=0)

if(kinks_.size == 1):
    i_r = np.int(kinks_[0])

i_l = i_r
i_r = phase_sec_order.shape[1]
phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l-1,axis=0)
    

#i_l =410
#i_r =450
plt.plot(phase_sec_order_[0,i_l:i_r])
#plt.plot(phase_sec_order_[1,i_l:i_r])
#plt.plot(phase_sec_order_[2,i_l:i_r])
#plt.plot(phase_sec_order_[3,i_l:i_r])
#plt.plot(phase_sec_order_[4,i_l:i_r])
plt.plot(phase_sec_order_[5,i_l:i_r])
plt.plot(phase_sec_order_[6,i_l:i_r])
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()

#%%
for i in [4]:#range(kinks_.size-1):
    swaps = []
    l += 1
    i_l = np.int(kinks_[i])+1    
    if(i == kinks_.size-1):
        i_r = phase_sec_order_.shape[1]-1
    else:
        i_r = np.int(kinks_[i+1])-2
    for i_ in range(i_r-i_l):
        grad_i_l = (phase_sec_order_[:,i_l+i_+1-1] - phase_sec_order_[:,i_l+i_-1])
        grad_i = (phase_sec_order_[:,i_l+i_+1] - phase_sec_order_[:,i_l+i_])
        print(i_+i_l,np.linalg.norm(grad_i-grad_i_l))

        if(np.linalg.norm(grad_i-grad_i_l)>0.005):
            swaps = np.append(swaps,[i_+i_l])
            print(i_+i_l,np.linalg.norm(grad_i-grad_i_l))
            print(grad_i)
            #print(phase_sec_order_[:,i_+i_l])
            print(grad_i_l)
            print(grad_i-grad_i_l)
            #print(phase_sec_order_[:,i_+i_l-1])
    i_r = 0
    for i_ in range(swaps.size):
        i_l = np.int(swaps[i_])
        if(i_ < swaps.size-1):
            i_r = np.int(swaps[i_+1])
        else:   
            i_r = phase_sec_order_.shape[1]
        phase_sec_order2 = np.copy(phase_sec_order_)
        phase_sec_order3 = np.copy(phase_sec_order_)
        if(i_==0):
            phase_sec_order_[0,i_l:i_r] =phase_sec_order2[6,i_l:i_r]
            phase_sec_order_[6,i_l:i_r] =phase_sec_order2[0,i_l:i_r]
        if(i_==1):
            phase_sec_order_[0,i_l:i_r] =phase_sec_order2[5,i_l:i_r]
            phase_sec_order_[5,i_l:i_r] =phase_sec_order2[6,i_l:i_r]
            phase_sec_order_[6,i_l:i_r] =phase_sec_order2[0,i_l:i_r]
i_l =410
i_r =450
plt.plot(phase_sec_order_[0,i_l:i_r])
#plt.plot(phase_sec_order_[1,i_l:i_r])
#plt.plot(phase_sec_order_[2,i_l:i_r])
#plt.plot(phase_sec_order_[3,i_l:i_r])
#plt.plot(phase_sec_order_[4,i_l:i_r])
plt.plot(phase_sec_order_[5,i_l:i_r])
plt.plot(phase_sec_order_[6,i_l:i_r])
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()
        

#%%    
#plt.plot(time[N_t_0:N_t_],np.gradient(np.gradient(phase_sec[0,:],1),1))
#plt.plot(time[N_t_0:N_t_],np.gradient(phase_sec[0,:],1))
#plt.plot(time[N_t_0:N_t_],np.gradient(phase_sec[1,:],1))
#plt.plot(time[N_t_0:N_t_],np.gradient(phase_sec[2,:],1))
#plt.plot(time[N_t_0:N_t_],np.gradient(phase_sec[3,:],1))
#plt.plot(time[N_t_0:N_t_],np.gradient(phase_sec[4,:],1))
#plt.show()#%%
plt.plot(time[N_t_0:N_t_],phi_t[0,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[1,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[2,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[3,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[4,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[5,N_t_0:N_t_]/time[N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[6,N_t_0:N_t_]/time[N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[7,N_t_0:N_t_]/time[N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[8,N_t_0:N_t_]/time[N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[9,N_t_0:N_t_]/time[N_t_0:N_t_])
#plt.plot(time[N_t_0:N_t_],phi_t[10,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()
#%%
from scipy.ndimage.interpolation import shift

phase_sec[0,:]
phase_shift = shift(np.real(phase_sec[1,:]),1,cval=phase_sec[1,0])
plt.plot(phase_sec[1,:])
plt.plot(phase_sec[1,:] - phase_shift)
plt.ylabel('phase', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.show()

#%%
time = np.linspace(1,8192,8192)
N_t_0 = 0#8160
N_t_  = 1024#8191
plt.plot(time[N_t_0:N_t_],phi_t[0,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[1,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[2,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[3,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[4,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[5,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[6,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[7,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[8,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[9,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[10,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[11,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[12,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[13,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[14,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[15,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[16,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[17,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[18,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[19,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[20,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[21,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[22,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[23,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[24,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[25,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[26,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[27,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[28,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[29,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[30,N_t_0:N_t_]/time[N_t_0:N_t_])
plt.plot(time[N_t_0:N_t_],phi_t[31,N_t_0:N_t_]/time[N_t_0:N_t_])
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