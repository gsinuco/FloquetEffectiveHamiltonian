#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:15:14 2020

@author: german

"""

import numpy as np
import matplotlib.pyplot as plt
import Bloch_wave as BW

def V_OL_array(i,V_m,k_m,phi_m):
    return lambda x : 0.5*V_m[i]+0.5*V_m[i]*np.cos(k_m[i]*x+2.0*phi_m[i])

def Sequence_AVG(V_m,k_m,phi_m,x,DelatT):
    Vs    = [V_OL_array(i,V_m,k_m,phi_m) for i in range(M)] # Make a list of  OL functions
    T = np.sum(DeltaT)
    V_array = DeltaT[0]/T * Vs[0](x)
    for i in range(DeltaT.size-1):
        V_array += DeltaT[i+1]/T * Vs[i+1](x)
    return V_array
    
def F(m1,m2,j,omega,DeltaT):
    w  = omega
    t1 = np.sum(DeltaT[0:m1])
    t2 = np.sum(DeltaT[0:m2])
    F_ = np.cos(w*j*(t1-t2))*(np.cos(w*j*(DeltaT[m1] - DeltaT[m2])) 
                                           - np.cos(w*j*DeltaT[m1]) 
                                           - np.cos(w*j*DeltaT[m2]) 
                                           + 1.0)
    return F_/(4.0*np.pi*np.pi)

def H_F_3(Vs,x,omega,DelatT):
    
    H_ = np.zeros_like(x)
    
    N = 16
    for j in range(N+1):

        for m1 in range(DeltaT.size):

            #lambda x: np.max(Vs[m1](x))-np.min(Vs[m1](x))
            Vm1 = np.max(Vs[m1](x))-np.min(Vs[m1](x))
            Vm1 = np.max(Vs[m1](x))-np.min(Vs[m1](x))

            for m2 in range(DeltaT.size):                

                Vm2 = np.max(Vs[m2](x))-np.min(Vs[m2](x))

                H_ += 0.25*Vm1*Vm2*F(m1,m2,j+1,omega,DeltaT)*k_m[m2]*k_m[m1]*np.sin(k_m[m2]*x
                               +2*phi_m[m2])*np.sin(k_m[m1]*x+2*phi_m[m1])/(np.power(1.0*(j+1),4.0)) 
    
    return H_/(2*pi*omega)**2             

#%%    
    

jj = np.complex(0.0,1.0)
pi = np.pi



#################################################################
##############  PULSE SEQUENCE ##################################
#################################################################


omega  = 15.0           # Driving frequency =: hbar omega/ [E]
T      = (2.0*pi)/omega # Period of the driving 

# Set the parameter of the sequence of static optical lattices parametrised by
# V(x) = (V_0/2) (cos(k_x  x  +  2.0*phase) + 1)
M      =  1
DeltaT = np.array([T],dtype=np.double) # List of durations
#DeltaT = np.array([T/4.0,     T/4.0, T/4.0,T/4.0],dtype=np.double) # List of durations
V_m    = np.array([20.0],dtype=np.double)
#V_m   =  np.array([20.0,     10.0     , 20.0 ,5.0    ],dtype=np.double) # List of OL depths 
k_m    = np.array([2.0*pi],dtype=np.double)
#k_m   =  np.array([2.0*pi,6.0*pi   , 8.0*pi, 10.0*pi   ],dtype=np.double) # List of OL wavelengths
                                                     # the unit of lenght is the
                                                     # longest spatial period of the 
                                                     # optical lattice (DOUBLE THE LASER).
phi_m =  np.array([0.0],dtype=np.double) # List of phases (of the LASE)                                                     
#phi_m =  np.array([0.0, 0.0   ,0.0,0.0     ],dtype=np.double) # List of phases (of the LASE)
assert(np.sum(DeltaT[0:M])==T),"The sequence of pulses do not fit in one period T" 


Vs    = [V_OL_array(i,V_m,k_m,phi_m) for i in range(M)] # Make a list of  OL functions

L     = 64 # number of unit cells
N_x   = 64 # Number of points in a unit cell
x     = np.linspace(0, L, N_x*L)
#for i in range(M):
#    plt.plot(x[0:128],Vs[i](x)[0:128])

plt.xlabel("x")
plt.title("Optical lattice contributions")
plt.show()

## AVERAGE POTENTIAL ####
V_avg = Sequence_AVG(V_m,k_m,phi_m,x,DeltaT)
plt.plot(x[0:128],V_avg[0:128])
plt.xlabel("x")
plt.title("AVG  lattice")
plt.show()

## CORRECTIONL ####
V_correction = H_F_3(Vs,x,omega,DeltaT)
plt.plot(x[0:128],V_correction[0:128])
plt.xlabel("x")
plt.title("H_F_3  lattice")
plt.show()

#%%
#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 5  # Number of bands to evaluate
dx    = 1.0/N_x
x     = np.linspace(0, L, N_x*L)

N = N_x*L
numVec = Bands*L


# Evaluate the Bloch spectrum of the OL
# with tle longest spatial period
# to define the Brillouin Zone
l, v = BW.get_eigen(N, L, numVec, Vs[0])
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(v[1::, i], N_x)

plt.plot(k_u,l,".")
plt.title("Bloch spectrum during the \n pulse with the longest OL period")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()


for i in range(M):
    l, v = BW.get_eigen(N, L, numVec, Vs[i])
    plt.plot(k_u,l,".")
    plt.title("Bloch spectrum during the \n  pulse No. {}".format(i))
    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.show()


l, v = BW.get_eigen(N, L, numVec, V_avg)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(v[1::, i], N_x)

plt.plot(k_u,l,".")
plt.title("Bloch spectrum of the AVG Hamiltoninan")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()

l, v = BW.get_eigen(N, L, numVec, V_avg + V_correction)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(v[1::, i], N_x)

plt.plot(k_u,l,".")
plt.title("Bloch spectrum of the Effective Hamiltonian")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()
