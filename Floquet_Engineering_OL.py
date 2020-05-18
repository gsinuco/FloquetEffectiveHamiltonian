#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:50:40 2020


@author: German Sinuco

Here we evaluate the Floquet spectrum of an optical lattice periodically driven.

"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
from scipy.integrate import ode
#from scipy.integrate import complex_ode


def d(t):
    return 2.0 + np.cos(t)

def phi(t):
    return 0

def V(t):
    return 1.0

def OL_driving(OL,x,t):
    V = np.cos(OL.k_x*x+OL.phi_x)*(OL.V_(t)*np.cos(OL.DK(t)*x+OL.Dphi(t)) - OL.V)-np.sin(OL.k_x*x+OL.phi_x)*OL.V_(t)*np.sin(OL.DK(t)*x+OL.Dphi(t))
    return V

def U_t(t,x,OL,U_kn_x):
    # t : instant of time
    # x : np.linspace(x_l,x_r,N_x)
    # U_kn: Space representation of the Bloch functions
    
    V_x = np.diag(OL_driving(OL,x,t))
    U = np.transpose(np.conjugate(U_kn_x))@V_x@U_kn_x
    return U
    

#def Schrodinger_RHS(psi,t,E_0,V_0): # for odeint
def Schrodinger_RHS(t,psi,E_0,x,OL,U_kn): # for ode
    jj  = np.complex(0.0,1.0)
    H_0 = np.diag(E_0)
    V   = U_t(t,x,OL,U_kn)#7.0*V_0*np.cos(t) 
    H   = H_0 + V
    dpsidt_RHS = -jj*H@psi
    return dpsidt_RHS

class OpticalLattice():
    def __init__(self,V=1.0,k_x=1.0,phi_x=0):                
        self.V     = V      # Depth
        self.k_x   = k_x    # wavevector
        self.phi_x = phi_x  # phase  V cos^2(kx+phi)
    
    def DK(self,t):
        return 1.0/d(t) - self.k_x
    
    def Dphi(self,t):
        return phi(t) - self.phi_x
    
    def V_(self,t):
        return V(t)
        

class Bloch_spectrum(object):

    def __init__(self,L=16,G=3):
        self.L  = L  # Number of lattice periods, assert it is even ==  number of values of momentum
        self.G  = G  # Number of Bands ASSERT G IS ODD
        self.D = G*L+1
        #self.H = np.zeros([G,G],dtype=np.complex)
    
    def Spectrum(OL,self):  
        G  = self.G
        L  = self.L
        V  = OL.V
        Bloch_energy  = []#np.zeros([G],dtype=np.float)
        Bloch_wavefun = np.empty([G,G])
        V_ = np.zeros([G,G],dtype=np.complex)
        K  = np.zeros([G,G],dtype=np.complex)
        
        # FOR EACH QUASIMOMENTUM (m)
        for m in np.linspace(-L/2,L/2,L+1):
            i_ = 0
            for n in np.linspace(-(G-1)/2,(G-1)/2,G):        
                momentum   = (1.0*m/L - n)
                K[i_,i_]   = 0.5*momentum*momentum # KINETIC ENERGY 
                if (i_+1 < G):
                    V_[i_,i_+1] = V#0.125   # POTENTIAL ENERGY         
                    V_[i_+1,i_] = V#0.125   # POTENTIAL ENERGY         
                #if (i_+2 < G):
                #    V_[i_,i_+2] = 3.0*V#0.125   # POTENTIAL ENERGY         
                #    V_[i_+2,i_] = 3.0*V#0.125   # POTENTIAL ENERGY         
                #if (i_+3 < G):
                #    V_[i_,i_+3] = 3.0*V#0.125   # POTENTIAL ENERGY         
                #    V_[i_+3,i_] = 3.0*V#0.125   # POTENTIAL ENERGY         

                i_=i_+1      
                
            H  = K + V_    
            if(m == -L/2):
                Bloch_energy,Bloch_wavefun  = np.linalg.eigh(H)
            else:
                Bloch_energy  = np.concatenate([Bloch_energy,np.linalg.eigh(H)[0]],axis=0)     
                Bloch_wavefun = np.concatenate([Bloch_wavefun,np.linalg.eigh(H)[1]],axis=0)     
            
        Bloch_energy   = np.reshape(Bloch_energy,[L+1,G])            

        return Bloch_energy,Bloch_wavefun
    
    def RealSpaceWavefun(self,x_l,x_r,N_x,BlochWavefun):
    
        Bands = self.G
        L     = self.L
        # BlochWavefun[m,n]
        # Fourier component of the Bloch wavefunction corresponding to the factor
        #
        # exp(i 2 np.pi m/(L a) ) exp(-i 2 np.pi n/a)
        # 
        # i.e. the wavefunction is:
        #
        # psi_{m,n}(x) = exp(i 2 np.pi m/(L a) ) sum_{n}  BlochWavefun[m,n]  exp(-i 2 np.pi n/a)

        i_= 0
        jj = np.complex(0.0,1.0)

        k      = 2.0 * np.pi * np.linspace(-(Bands-1)/2,(Bands-1)/2,Bands)
        x      = np.linspace(x_l,x_r,N_x)
        #kk, xx = np.meshgrid(k, x, sparse=False)
        # array of factors: exp(-i 2 np.pi n/a x)
        #phases = np.exp(-jj*kk*xx)
        #n=0
        #RealSpaceBlochfun = phases@BlochWavefun[n*Bands:(n+1)*Bands,0:Bands]
        #for n in range(L):
        #    RealSpaceBlochfun_ = phases@BlochWavefun[(n+1)*Bands:(n+2)*Bands,0:Bands]
        #    RealSpaceBlochfun  = np.concatenate((RealSpaceBlochfun,RealSpaceBlochfun_),axis=0)
        #print(RealSpaceBlochfun.shape)
            
        RealSpaceBlochfun = np.zeros([N_x*(L+1),Bands],dtype=np.complex)
        for i_ in range(L+1): # loop over the values of momentum in the BZ
            RealSpaceBlochfun_ = np.zeros([N_x,Bands],dtype=np.complex)
            m = i_ - L/2
            for n in range(Bands): # loo over the number of bands
                n_ = n - (Bands-1)/2
                momentum = (1.0*m/L - n_)
                for m in range(Bands): #sum_m exp(-i k_m x) U_{i_,m,n}
                    RealSpaceBlochfun_[:,n] = RealSpaceBlochfun_[:,n] + BlochWavefun[i_*Bands+m,n]*np.exp(-jj*k[m]*x)
                    
                RealSpaceBlochfun_[:,n] = np.exp(jj*momentum*x)*RealSpaceBlochfun_[:,n]     
                
            RealSpaceBlochfun[i_*N_x:(i_+1)*N_x,:] = RealSpaceBlochfun_

            
        #RealSpaceWF =np.zeros([N_x*(L+1),Bands],dtype=np.complex)
        #RealSpaceWF_=np.empty([N_x,Bands],dtype=np.complex)
        #for m in np.linspace(-L/2,L/2,L+1):
        #   i_ = int(m + L/2)
        #   for n in range(Bands): 
               #TO DO: TEST THAT THE MOMENTUM IS GIVEN BY               
        #       momentum   = 2.0*np.pi*(np.power(-1.0,n)*m/(1.0*L) + n)
        #       RealSpaceWF_[:,n]= RealSpaceBlochfun[i_*N_x:(i_+1)*N_x,n]#np.exp(jj*momentum*x)*RealSpaceBlochfun[i_*N_x:(i_+1)*N_x,n]
               
         #  if m==-L/2 :
         #      RealSpaceWF = RealSpaceWF_    
         #  else:
         #      RealSpaceWF = np.concatenate([RealSpaceWF,RealSpaceWF_],axis=0)     
           
        return RealSpaceBlochfun#RealSpaceWF

            

x_l = -4.0
x_r =  4.0    
N_x =  512 # number of points in real space
# Set the parameter of the static optical lattice
#
# V(x) = V_0 cos(2 pi x / a + phase)
#
V_0   = 2.0E-2
a     = 1.0
phase = 0.0

#################################################################
##############  BLOCH SPECTRUM ##################################
#################################################################
# Set the parameter of the Bloch spectrum
Bands = 5
  # define the number of bands to evaluate
L     = 64 # define the number of values of momentum in the Brilloun Zone

OL    = OpticalLattice(V_0,a,phase)
BS    = Bloch_spectrum(L,Bands)

# get the Bloch Spectrum and wavefunctions in the momentum representatoin
BlochSpectrum,BlochWavefun = Bloch_spectrum.Spectrum(OL,BS) 
# get the Bloch wavefunctions is the position representation
RealSpaceWavefun           = Bloch_spectrum.RealSpaceWavefun(BS,x_l,x_r,N_x,BlochWavefun) 



# plotting the Bloch spectrum
k = np.linspace(-np.pi/a,np.pi/a,L+1)
plt.plot(k,BlochSpectrum[:,0])
plt.show()
#%%
# plot one wavefucntion in the position representation
m  = 0 # Band index
n  = 2#int(L/2)  # [0,L] reduced momentum index to

# to do : test that the momentum is givne by
momentum = n/L + m
jj = np.complex(0.0,1.0)
x = np.linspace(x_l,x_r,N_x)
#plt.plot(x,np.real(np.exp(jj*k[n]*x)*RealSpaceWavefun[n*N_x:(n+1)*N_x,0]),x,np.imag(np.exp(jj*k[n]*x)*RealSpaceWavefun[n*N_x:(n+1)*N_x,0]))

#plt.plot(x,np.real(RealSpaceWavefun[n*N_x:(n+1)*N_x,m]),x,np.imag(RealSpaceWavefun[n*N_x:(n+1)*N_x,m]))
m = 0
plt.plot(x,np.real(RealSpaceWavefun[0:N_x,0]))  # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.plot(x,np.real(RealSpaceWavefun[0:N_x,1])) # ,x,np.imag(RealSpaceWavefun[:,m]))
#plt.plot(x,np.real(RealSpaceWavefun[0:N_x,2])) # ,x,np.imag(RealSpaceWavefun[:,m]))

plt.show()

n=0
plt.plot(np.real(BlochWavefun[n*Bands:(n+1)*Bands,m]))

n=int(L/2)
plt.plot(np.real(BlochWavefun[n*Bands:(n+1)*Bands,m]))

n=L
plt.plot(np.real(BlochWavefun[n*Bands:(n+1)*Bands,m]))

plt.show()


## TO DO : DEFINE THE KINETIC ENERGY OPERATOR IN THE BLOCH OR SPACE BASIS
## LATER ON, WE NEED THE KINETIC ENERGY OPERATOR
##
#%%
#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

# Solve the Schrodinger equation for a subset of the bands:
N_Bands = 2 # number of bands to include in the Schrodinger equation
E_0 = BlochSpectrum[:,0]#
for i_ in range(N_Bands-1):
    E_0 = np.concatenate([E_0,BlochSpectrum[:,i_+1]],axis=0) # two bands

U_x = np.zeros([N_x,(L+1)*N_Bands],dtype = np.complex)
for n in range(N_Bands):
    for i_ in range(L+1):
        U_x[:,n*N_Bands+i_] = RealSpaceWavefun[i_*N_x:(i_+1)*N_x,n]
    
# TO DO LOOP THE CONCATENATION BELOW TO OBTAIN THE WAVEFUNCITONS OF MORE THAN 
# BAND
#U_x = np.concatenate([U_x,np.reshape(RealSpaceWavefun[:,1],[N_x,L+1])],axis=1)

# Set the coupling of the Bloch states
# This is the potential energy : <k',n'|V(x,t)|k,n>
# e.g. a random potential looks like
#V_R = np.random.rand(E_0.size,E_0.size) # random coupling
#V_R = 0.000*0.5*(V_R + np.transpose(V_R)) # make the random coupling Hermitian

#set the initial condition and parameters to integrate the Schrodinger equation
#psi0 = np.ones(E_0.size)
t0   = 0   # initial time
DT   = 6.0 # total time to integrate
t1   = t0 + DT # final time 
N_t  = 128   # number of time steps
dt   = DT/N_t # time step
t    = np.linspace(t0,t0+DT, N_t) 
#%%
# SET THE INTEGRATOR
solver = ode(Schrodinger_RHS).set_integrator('zvode',method='bdf')

U_T = np.zeros([N_Bands*(L+1),N_Bands*(L+1)],dtype=np.complex)

for j_ in range(N_Bands*(L+1)):
    psi0 = np.zeros(E_0.size,dtype=np.complex)
    psi0[j_] = 1.0#np.complex(0.0,1.0)


    # define the parameters of the integrator
    # TO DO: TEST THE INTEGRATOR
    
    solver.set_initial_value(psi0,t0).set_f_params(E_0,x,OL,U_x)


    psi_t = np.zeros([N_t+1,E_0.size],dtype=np.complex) # Initialise the time dependent wavefunction 
    psi_t[0,:] = psi0
    i_ = 0
    while solver.successful() and solver.t<t1  and i_+1<N_t:
        i_ = i_+1
        #print(solver.t+dt, solver.integrate(solver.t+dt))
        psi_t[i_,:] = solver.integrate(solver.t+dt)
        
    U_T[:,j_] = psi_t[i_,:]

    #plt.plot(psi_t[0:i_,:])
    #plt.show()
#plt.show()

plt.contourf(np.abs(U_T))
plt.show()

#%%
#################################################################
####  EFFECTIVE FLOQUET HAMILTONIAN OPERATOR ####################
#################################################################

# Integrating the Schrodinger equation, build U_T  = U(t0+T,t0) in the basis of Bloch states
# Find the eigenvalues lambda_u and eigenvectors |u>=sum U^u_{kn}|kn> of U_T
# Find the eigenphases: phi_u = arctan(imag(lambda_u)/real(lambda_u))
# Define the Floquet quasienergies:
#    e_u = hbar*phi_u/T
#
# Build the matrix representation of the Effective Hamiltonian:
#
# <x'|H_eff|x> = sum_u sum_{k'n'} sum_{kn} e_u U^u_{k'n'} U^u_{kn}^* psi(x')_{k'n'} psi(x)_{kn}^*
# 
# where
# U^u_{kn} is the projection of the |u> over the Bloch state |kn>
# psi(x)_{kn} is the position representation of the Bloch state |kn>

# The effective potential energy landscape is given by the diagonal elements of the matrix
#
# <x'|H_eff|x> -  <x'|K|x> 
# where
# K is the kinetic energy operator.
#
# The position representation of these operators can be obtained using:
#
# ????

lambda_u,U = np.linalg.eig(U_T)
#print(lambda_u)
e_u = -np.arctan(np.imag(lambda_u)/np.real(lambda_u))/DT

#plt.plot(k,e_u,k,BlochSpectrum[:,0])
#plt.show()

H_eff_F = np.diag(e_u) # Effective Hamiltonian in the basis Floquet states

#Transformation from the Floquet basis to the basis of Bloch states
H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U

#U_x = np.empty([N_x,N_Bands*(L+1)],dtype=np.complex)

n = 0
#plt.plot(x,np.abs(RealSpaceWavefun[n*N_x:(n+1)*N_x,0:4]))
#plt.show()

#U_x = np.reshape(RealSpaceWavefun[:,1],[N_x,L+1])
#RealSpaceWavefun[:,0],[N_x,L+1])
#for n in range(N_Bands):
#    U_x[:,n*(L+1):(n+1)*(L+1)] = np.reshape(RealSpaceWavefun[:,n],[N_x,L+1])

H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))

V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)

plt.plot(x,V_eff_x)
plt.show()
        