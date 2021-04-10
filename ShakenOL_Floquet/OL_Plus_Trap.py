#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:15:40 2020

@author: german


Here we:
    
    1. evaluate the spectrum of an optical lattice plus a harmonic trap
    2. Evaluate the floquet spectrum of a driven OL + harmonic trap
    
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time 

import OpticalLatticeModule as OLM
import FloquetEffectiveHamiltonian as FEH
import Bloch_wave as BW


def V_trap(x=np.linspace(-np.pi,np.pi,32,dtype=np.double),omega_trap = 1.0,D = np.pi,L=np.pi):
    
    omega_trap2 = omega_trap*omega_trap
    alpha = 2*omega_trap2* D*L/(D-L) 
    beta  = omega_trap2*D*(2*L-D)/((D-L)**2)
    
    V = np.zeros_like(x)
    
    if D<L :
        dV_D = 2*omega_trap2*D/2
        V_D  = omega_trap2*D*D/2    
        a    = dV_D/(2*(D-L))
        b    = -dV_D*L/(D-L)
        c    = V_D - a*D*D  - b*D 
    
        to_the_left  = np.argwhere(x < -D)
        to_the_right = np.argwhere(x >  D)
        central_section     = np.argwhere(np.abs(x)<=D)
        V[to_the_left]      = a*x[to_the_left]**2-b*x[to_the_left]+c
        V[to_the_right]     = a*x[to_the_right]**2+b*x[to_the_right]+c
        V[central_section]  = omega_trap2*x[central_section]**2/2
    else:
        V = omega_trap2*x
            
    
    
    return V

def Spectrum_trapped_OL(L=32,N_x=16,D=14,omega_trap=0.001,V_0=30.0,phase_x=0.0):
    #k     = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
    
    dx    = 1.0/N_x

    H     = np.zeros([N_x*L,N_x*L],dtype=np.complex)
    x     = np.linspace(-L/2, L/2, N_x*L)


    V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)


    V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
    H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

    dim = H.shape[0]
    H[dim-1,0] = H[0,0] - 2/dx**2
    H[0,dim-1] = H[0,0] - 2/dx**2

    E = np.linalg.eigvalsh(H)
    
    return E,np.diag(V)+2/dx**2
#%%
jj = np.complex(0.0,1.0)

omega_trap = 0.001
V_0        = 30# ~ Band gap
phase_x    = 0.0  # spatial phase
           
# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) (cos(2*k_x  x  +  2.0*phase) + 1)
omega_trap = 0.0
V_0        = 30 # ~ Band gap
phase_x    = 0.0  # spatial phase
OL         = OLM.OpticalLattice(V_0,phase_x)

#Driving Frequency
omega    = 5.0  #  Driving frequency

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 3  # Number of bands to evaluate
L     = 64 # number of unit cells
N_x   = 32 # Number of points in a unit cell
dx    = 1.0/N_x
x     = np.linspace(0, L, N_x*L)

Spectrum,RealSpaceWavefun = BW.Spectrum(L,N_x,Bands,V_0,2.0*phase_x,omega_trap=0)
k_u = np.empty(Bands*L)
for i in range(Bands*L):
    k_u[i] = BW.get_k(RealSpaceWavefun[1:, i], N_x)


for i in [1]:#range(2):
    omega_trap = 0.005*i
    Spectrum2,RealSpaceWavefun = BW.Spectrum(L,N_x,Bands,V_0,2.0*phase_x,omega_trap)
    #plt.plot(k,Spectrum2,".")
    

plt.figure()
plt.plot(Spectrum,".")
plt.plot(Spectrum2,".")
plt.ylabel('Energy', fontsize=18)
#plt.xlim(0,800)
plt.ylim(0,75)
plt.xlabel('k', fontsize=16)
plt.show()


#plt.figure()
#plt.plot(RealSpaceWavefun)



#%%
import numpy as np
import matplotlib.pyplot as plt

omega_trap = 0.001
V_0        = 30# ~ Band gap
phase_x    = 0.0  # spatial phase

L     = 128 # number of unit cells
N_x   = 16 # Number of points in a unit cell
k     = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x

H     = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x     = np.linspace(-L/2, L/2, N_x*L)

D = L/2-2 

E,V_trap_UnitCell = Spectrum_trapped_OL(L,N_x,D,omega_trap,V_0,phase_x)

plt.figure()
plt.plot(x,V_trap_UnitCell)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max

plt.figure()
plt.plot(k,E[0:k_max],".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)


#%%
omega_trap = 0.1
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)

plt.figure()
plt.plot(x,V_trap_UnitCell)

V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E = np.linalg.eigvalsh(H)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max

plt.figure()
plt.plot(k,E[0:k_max],".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)

#&&

L     = 16 # number of unit cells
N_x   = 32 # Number of points in a unit cell
k  = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x = np.linspace(-L/2, L/2, N_x*L)

D = L/2-2 
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)


V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E = np.linalg.eigvalsh(H)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max
plt.plot(k,E[0:k_max],".")


L     = 32 # number of unit cells
N_x   = 32 # Number of points in a unit cell
k  = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x = np.linspace(-L/2, L/2, N_x*L)

D = L/2-2 
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)


V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E = np.linalg.eigvalsh(H)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max
plt.plot(k,E[0:k_max],".")

plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)



L     = 64 # number of unit cells
N_x   = 32 # Number of points in a unit cell
k     = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H     = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x     = np.linspace(-L/2, L/2, N_x*L)

D = L/2-2 
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)


V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E = np.linalg.eigvalsh(H)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max
plt.plot(k,E[0:k_max],".")

plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)



L     = 127 # number of unit cells
N_x   = 64 # Number of points in a unit cell
k  = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x = np.linspace(-L/2, L/2, N_x*L)
D = L/2-2 
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)


V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x))+V_trap_UnitCell)# + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E,psi_E = np.linalg.eigh(H)

k_max = L+1
k = np.linspace(0,k_max,k_max)/k_max
plt.plot(k,E[0:k_max],".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)

#%%

L     = 256 # number of unit cells
N_x   = 32 # Number of points in a unit cell
k  = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x = np.linspace(-L/2, L/2, N_x*L)
V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x)) + 0.5*omega_trap*(x**2))
H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

dim = H.shape[0]
#H[dim-1,0] = H[0,0] - 2/dx**2
#H[0,dim-1] = H[0,0] - 2/dx**2

E = np.linalg.eigvalsh(H)
k = np.linspace(0,400,400)/400
plt.plot(k,E[0:400],".")
plt.ylabel('Energy', fontsize=18)
plt.xlabel('k', fontsize=16)


#%%

plt.figure()
plt.plot(np.abs(psi_E[:,20]))
plt.show()

#%%

omega_trap = 0.001

L     = 32 # number of unit cells
N_x   = 16 # Number of points in a unit cell
k  = (1/(L*N_x))*(np.linspace(0,L*N_x,L*N_x))
dx    = 1.0/N_x
H = np.zeros([N_x*L,N_x*L],dtype=np.complex)
x = np.linspace(-L/2, L/2, N_x*L)

D = 8 
V_trap_UnitCell = V_trap(x,omega_trap,D,L/2)

#V = np.diag(2/dx**2+0.5*V_0 + 0.5*V_0*(np.cos(2.0*np.pi*x+phase_x)) + 0.5*omega_trap*(x**2))
#H = V + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=1) + np.diag(-1/dx**2 * np.ones(x.shape[0]-1),k=-1) 

