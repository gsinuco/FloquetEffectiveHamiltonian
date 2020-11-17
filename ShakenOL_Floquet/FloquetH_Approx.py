#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:15:14 2020

@author: german

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time 

import OpticalLatticeModule as OLM
import Bloch_wave as BW


def beta_n_p(n=2.0,p=4.0):
    limits = 1024
    m     = np.linspace(-limits,-1,limits)
    m_    = np.linspace(1,limits,limits)
    beta  = (2*n*n*(1-np.cos(2*pi*m/n)) -2.0*n*(2.0-2.0*np.cos(m*pi*(1.0+1.0/n) )*np.cos(m*pi*(1.0-1.0/n))) )/np.power(m,p)
    beta_ = (2*n*n*(1-np.cos(2*pi*m_/n))-2.0*n*(2.0-2.0*np.cos(m_*pi*(1.0+1.0/n))*np.cos(m_*pi*(1.0-1.0/n))))/np.power(m_,p)
    return (np.sum(beta)+np.sum(beta_))/np.power(2*np.pi*n,2.0)


#def H_F_3(OL):
#    H_F_3 =  beta_n_p(2.0,4.0)*0.5*np.power((V_0/omega)*(-np.sin(k1*x+2.0*phi_0) + g*np.sin(2*k1*x+2.0*phi_1)),2.0)


#%%    
    

jj = np.complex(0.0,1.0)
pi = np.pi
            
# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) (cos(2*k_x  x  +  2.0*phase) + 1)
V_0      = 55   # ~ Band gap
phase_x  = 0.0  # spatial phase
OL       = OLM.OpticalLattice(V_0,phase_x)

#Driving Frequency
omega    = 10.0  #  Driving frequency
#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 3  # Number of bands to evaluate
L     = 64 # number of unit cells
N_x   = 32 # Number of points in a unit cell
dx    = 1.0/N_x
x     = np.linspace(0, L, N_x*L)

BlochSpectrum,RealSpaceBlochWavefun = BW.BlochSpectrum_array(L,N_x,Bands,V_0,2.0*phase_x)


k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceBlochWavefun[1::, i], N_x)
#%%

plt.plot(k_u,BlochSpectrum[1:1+Bands*L],".")
plt.title("Bloch spectrum during the \n first half period  [0,T/2]")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()
plt.plot(k_u,BlochSpectrum[1+Bands*L:1+2*Bands*L],".")
plt.title("Bloch spectrum during the \n second half period  [T/2,T]")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()
plt.plot(k_u,BlochSpectrum[1+2*Bands*L:1+3*Bands*L],".")
plt.title("Bloch spectrum of the AVG lattice")
plt.xlabel("k")
plt.ylabel("Energy")
plt.show()
plt.title("Bloch spectrum of the \n AVG + HF_3 lattice")
plt.xlabel("k")
plt.ylabel("Energy")
plt.plot(k_u,BlochSpectrum[1+3*Bands*L:1+4*Bands*L],".")
plt.show()

#%%
print(beta_n_p(2.0,4.0))


k1    = 2.0*pi # Because the 
phi_0 = 0.0
phi_1 = 0.0
# g is defined by:
#    k1 V1 = g k0 V0
g     = 2.0

#H_F_3 =  beta_n_p(2.0,4.0)*0.5*np.power((V_0/omega)*(-np.sin(k1*x+2.0*phi_0) + g*np.sin(2*k1*x+2.0*phi_1)),2.0)

#np.power((0.25*pi*V_0/omega)*(-np.sin(k1*x+2.0*phi_0) + g*np.sin(2*k1*x+2.0*phi_1)),2.0)
#plt.plot(x,H_F_3)



n = 4
Vs = [BW.V_sin_array(i,V_0,phi_0,omega,phi_0,phi_1) for i in range(n)]
plt.plot(x[0:128],Vs[2](x%1)[0:128])
plt.title("AVG potential energy landscape")
plt.xlabel("x")
plt.show()

plt.plot(x[0:128],Vs[3](x%1)[0:128])
plt.title("AVG + H_F_3 energy landscape")
plt.xlabel("x")
plt.show()
