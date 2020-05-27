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
#from OpticalLatticeModule import Bloch_spectrum as BS

jj = np.complex(0.0,1.0)
            

# Set the domain of the wavefunctions and the potential energy
x_l = -64.0
x_r =  64.0    
N_x =  1024# number of points in real space
x   =  np.linspace(x_l,x_r,N_x)

# Set the parameter of the static optical lattice parametrised by
# V(x) = (V_0/2) cos(2*k_x  x  +  2.0*phase)
V_0      = 20.1 # ~ Band gap
phase_x  = 0.0     #np.pi/3.0
omega    = 1.0     # in units of omega_0 = (E)/hbar, with E = (2 hbar/lambda_0)/m
phase_t  = 0.0

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum

# define the number of bands to evaluate: make sure it is odd
Bands = 7
# define the number of values of momentum in the Brilloun Zone: make sure is even
L     = 32


#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################
# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 1# number of bands to include in the Schrodinger equation
#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 2.0*np.pi/omega # total time to integrate
t1   = t0 + DT # final time 
N_t  = 512  # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 


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
solver = ode(OLM.Schrodinger_RHS).set_integrator('zvode',method='bdf')
#solver = ode(OLM.Schrodinger_Momentum_RHS).set_integrator('zvode',method='bdf')
#H_0    = BS.H_0[0:N_Bands*(L+1),0:N_Bands*(L+1)] # Hamiltonian in the basis of momentum

t1   = t0 + DT # final time 
dt   = DT/N_t # time step

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

#U_x = np.transpose(np.reshape(RealSpaceMomentumWavefun[:,0],[L+1,N_x]))
#for n in range(N_Bands-1):
#    U_x_  = np.transpose(np.reshape(RealSpaceMomentumWavefun[:,n+1],[L+1,N_x]))
#    U_x = np.concatenate([U_x,U_x_],axis=1)

#%%    
U_T = np.zeros([N_Bands*(L+1),N_Bands*(L+1)],dtype=np.complex)


for j_ in range(N_Bands*(L+1)):
    psi0 = np.zeros(E_0.size,dtype=np.complex)
    psi0[j_] = 1.0#np.complex(0.0,1.0)
            
    # define the parameters of the integrator
    # TO DO: TEST THE INTEGRATOR
    solver.set_initial_value(psi0,t0).set_f_params(E_0,x,OL,U_x)
    #solver.set_initial_value(psi0,t0).set_f_params(H_0,x,OL,U_x)

    
    psi_t = np.zeros([E_0.size],dtype=np.complex) # Initialise the time dependent wavefunction 
    i_ = 0
    if(j_==0):
        f = np.zeros([N_t-1,N_Bands*(L+1)],dtype=complex)
    while solver.successful() and solver.t<t1  and i_+1<N_t:
        i_ = i_+1            
        psi_t = solver.integrate(solver.t+dt)
        if j_ == 0:
            f[i_-1,:] = psi_t
            #if j_ == 0:
                #   psi_t_[i_,:] = p#si_t
    #print(np.abs(psi_t))
    U_T[:,j_] = psi_t
    #print(j_,psi_t)
    
#plt.plot(f[:,0],f[:,1],f[:,0],f[:,2])
#plt.show()

lambda_u,U = np.linalg.eig(U_T)
e_u = -np.arctan(np.imag(lambda_u)/np.real(lambda_u))/DT
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

plt.contourf(np.abs(H_eff_kn))
plt.colorbar()
plt.show()   
    
plt.plot(k_u,e_u)
plt.show()
#plt.plot(e_u[k_u])
plt.plot(E_0)
plt.plot(np.diag(np.real(H_eff_kn)))
plt.show()

#Transformation from the Bloch basis to the basis of position
H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))

#plt.contourf(np.abs(H_eff_x))
#plt.plot(np.abs(U_T))
#plt.show()

plt.plot(x,np.diag(H_eff_x))
#plt.contourf(np.abs(U))
#plt.plot(np.abs(U))
plt.show()


#Transformation from the Boch basis to the basis of momentum
H_eff_k = U_k@H_eff_kn@np.transpose(np.conjugate(U_k))

#plt.contourf(np.abs(H_eff_x))
#plt.plot(np.abs(U_T))
#plt.show()

plt.plot(np.diag(H_eff_k))
#plt.contourf(np.abs(U))
#plt.plot(np.abs(U))
plt.show()

#print(lambda_u)
#print(np.diag(U_T))

#%%
plt.plot(RealSpaceWavefun[0:256,:])
plt.show()

print(np.abs(np.transpose(np.conjugate(RealSpaceWavefun[0:N_x,:]))@RealSpaceWavefun[0:N_x,:]))
print(np.abs(np.transpose(np.conjugate(RealSpaceWavefun[N_x:2*N_x,:]))@RealSpaceWavefun[N_x:2*N_x,:]))
print(np.abs(np.transpose(np.conjugate(RealSpaceWavefun[10*N_x:11*N_x,:]))@RealSpaceWavefun[9*N_x:10*N_x,:]))

# Example of how to plot the Bloch spectrum
k = np.linspace(-np.pi,np.pi,L+1)
plt.plot(k,BlochSpectrum[:,0],k,BlochSpectrum[:,1],k,BlochSpectrum[:,2])
plt.title("Bloch spectrum ")
plt.xlabel('x/a', fontsize=12)
plt.ylabel('Energy [E=(hbar/a)^2/m]', fontsize=12)
plt.show()
plt.savefig('BlochSpectrum.png')


G = np.linspace(-(Bands-1)/2,(Bands-1)/2,Bands)
plt.plot(G,np.real(BlochWavefun[0:Bands,0:3]))
plt.title("Bloch Functions: momentum representation |-pi/a,n>")
plt.xlabel('G = 2 pi q /a', fontsize=12)
plt.ylabel('<G|-pi,n>', fontsize=12)
plt.show()
plt.savefig('BlochWavefun_k.png')

plt.title("Bloch Functions: position representation |-pi/a,n>")
plt.plot(x,np.power(np.abs(RealSpaceWavefun[0:N_x,0]),2))  # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.plot(x,np.power(np.abs(RealSpaceWavefun[0:N_x,1]),2)) # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.plot(x,np.power(np.abs(RealSpaceWavefun[0:N_x,2]),2)) # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.xlabel('x/a', fontsize=12)
plt.ylabel('|<x|-pi,n>|^2', fontsize=12)
plt.show()
plt.savefig('BlochWavefun_x.png')

#%%

n=2
plt.title("Bloch Functions: position representation |-pi,n>")
plt.plot(x,np.power(np.abs(RealSpaceWavefun[0:N_x,n]),2))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()
plt.plot(x,np.power(np.abs(RealSpaceWavefun[N_x:2*N_x,n]),2)) # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()
plt.plot(x,np.power(np.abs(RealSpaceWavefun[2*N_x:3*N_x,n]),2)) # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()
plt.plot(x,np.power(np.abs(RealSpaceWavefun[3*N_x:4*N_x,n]),2),x,np.zeros(N_x)) # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.show()
plt.plot(x,np.power(np.abs(RealSpaceWavefun[4*N_x:5*N_x,n]),2)) # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.show()
#plt.xlabel('x/a', fontsize=12)
#plt.ylabel('<x|-pi,n>', fontsize=12)
#plt.show()

U_x_0 = np.transpose(np.reshape(RealSpaceWavefun[:,n],[L+1,N_x]))

##plt.plot(np.power(np.abs(U_x_[4,:]),2))
##plt.show()
##plt.plot(np.power(np.abs(RealSpaceWavefun[4*N_x:5*N_x,n]),2))
##plt.show()###

plt.contourf(np.power(np.abs(U_x_0),2))#
plt.colorbar()
plt.show()

#U_x_1 = np.transpose(np.reshape(RealSpaceWavefun[:,1],[L+1,N_x]))

#plt.plot(np.power(np.abs(U_x_[4,:]),2))
#plt.show()
#plt.plot(np.power(np.abs(RealSpaceWavefun[4*N_x:5*N_x,n]),2))
#plt.show()

#plt.contourf(np.power(np.abs(U_x_1),2))
#plt.colorbar()
#plt.show()

#U_x_ = np.concatenate([U_x_0,U_x_1],axis=1)

#plt.contourf(np.power(np.abs(U_x_),2))
#plt.colorbar()
#plt.show()

#plt.contourf(np.power(np.abs(U_x),2))
#plt.show()

#plt.plot(x,np.power(np.abs(RealSpaceWavefun[0:N_x,n]),2))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.plot(x,np.power(np.abs(RealSpaceWavefun[int(L/2) * N_x:(int(L/2)+1)*N_x,n]),2))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.plot(x,np.power(np.abs(RealSpaceWavefun[L*N_x:(L+1)*N_x,n]),2))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()


#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

U_T,U_x = FEH.FloquetStroboscopicOperator(N_Bands,t0,DT,N_t,x,OL,BlochSpectrum,RealSpaceBlochWavefun)



lambda_u,U,e_u,H_eff_kn,H_eff_x = FEH.EffectiveFloquetHamiltonian(U_T,DT,U_x)

V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)
k_u =np.argmax(np.abs(U),axis=0)

#%%    

plt.plot(k,BlochSpectrum[:,0],k,BlochSpectrum[:,1],k,BlochSpectrum[:,2])#,k,BlochSpectrum[:,2])
plt.plot(k,e_u[0:L+1])#,k,e_u[(L+1):2*(L+1)],k,e_u[2*(L+1):3*(L+1)])#,k,e_u[2*(L+1):3*(L+1)],k,e_u[3*(L+1):4*(L+1)])
#plt.plot(k,e_u[0:L+1],k,BlochSpectrum[:,0])
plt.show()

#%%
plt.plot(k,np.real(lambda_u))
plt.plot(k,np.imag(lambda_u))
plt.show()
#%%

plt.contourf(np.imag(U_x))
plt.show()

plt.contourf(np.real(U_x))
plt.show()

plt.contourf(np.power(np.abs(U_x),2))
plt.show()

#%%
#RealSpaceWavefun[i_*N_x:(i_+1)*N_x,n]
#plt.contourf(np.imag(np.reshape(RealSpaceWavefun[:,0],[N_x,(2*L+1)])))
#plt.show()

plt.contourf(np.real(U_x))
plt.show()

plt.contourf(np.abs(U_x))
plt.show()


plt.contourf(np.imag(U_T))
plt.show()

plt.contourf(np.real(U_T))
plt.show()

#%%
plt.contourf(H_eff_x)
plt.colorbar()
plt.show()

plt.plot(x[0:int(N_x/12)],V_eff_x[0:int(N_x/12)])
plt.xlabel('x/a', fontsize=12)
plt.ylabel('V_eff(x)', fontsize=12)
plt.show()
    
#%%
n=0
plt.title("Bloch Functions: position representation |-pi,n=0>")
plt.plot(x,np.abs(RealSpaceWavefun[0:N_x,n]))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()
plt.plot(x,np.abs(RealSpaceWavefun[int(L/4)*N_x:(int(L/4)+1)*N_x,n]))
#plt.show()
plt.plot(x,np.abs(RealSpaceWavefun[int(L/2)*N_x:(int(L/2)+1)*N_x,n]))
#plt.show()
plt.plot(x,np.abs(RealSpaceWavefun[int(3*L/4)*N_x:(int(3*L/4)+1)*N_x,n]))
#plt.show()
plt.plot(x,np.abs(RealSpaceWavefun[L*N_x:(L+1)*N_x,n])) # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()

n=1
plt.title("Bloch Functions: position representation |-pi,n=1>")
plt.plot(x,np.abs(RealSpaceWavefun[0:N_x,n]))  # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.plot(x,np.abs(RealSpaceWavefun[int(L/4)*N_x:(int(L/4)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[int(L/2)*N_x:(int(L/2)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[int(3*L/4)*N_x:(int(3*L/4)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[L*N_x:(L+1)*N_x,n])) # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.show()
n=2
#plt.title("Bloch Functions: position representation |-pi,n=2>")
plt.plot(x,np.abs(RealSpaceWavefun[0:N_x,n]))  # ,x,np.imag(RealSpaceWavefun[:,m]))
plt.plot(x,np.abs(RealSpaceWavefun[int(L/4)*N_x:(int(L/4)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[int(L/2)*N_x:(int(L/2)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[int(3*L/4)*N_x:(int(3*L/4)+1)*N_x,n]))
plt.plot(x,np.abs(RealSpaceWavefun[L*N_x:(L+1)*N_x,n])) # ,x,np.imag(RealSpaceWavefun[:,m]))

plt.xlabel('x/a', fontsize=12)
plt.ylabel('<x|-pi,n>', fontsize=12)
plt.show()
#%%

#%%    

k = np.linspace(-np.pi,np.pi,L+1)
plt.plot(k,BlochSpectrum[:,0],k,BlochSpectrum[:,1],k,BlochSpectrum[:,2])#,k,BlochSpectrum[:,2])
plt.plot(k,e_u[0:L+1])#,k,e_u[(L+1):2*(L+1)],k,e_u[2*(L+1):3*(L+1)])#,k,e_u[2*(L+1):3*(L+1)],k,e_u[3*(L+1):4*(L+1)])
#plt.plot(k,e_u[0:L+1],k,BlochSpectrum[:,0])
plt.show()
#%%
plt.contourf(np.abs(np.transpose(np.conjugate(U_x))@U_x))
plt.colorbar()
plt.show()
#%%

plt.plot(x,np.real(U_x[:,1]),x,np.real(U_x[:,5]))#,x,np.imag(U_x[:,8]))
plt.plot(x,np.imag(U_x[:,1]),x,np.imag(U_x[:,5]),x,np.imag(U_x[:,8]))
plt.show()
#%%
plt.contourf(np.abs(np.transpose(np.conjugate(U_x))@U_x)[0:128,0:128])
plt.colorbar()
plt.show()

#%%
#plt.plot(U_x)
