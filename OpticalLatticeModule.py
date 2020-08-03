#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:03:38 2020

@author: german
"""

import numpy as np
import matplotlib.pyplot as plt

class Schrodinger(object):
    def __init__(self, f, fargs=[]):
        self._f = f
        self.fargs=fargs

    def f(self, t, z):
        return self._f(t, z, *self.fargs)


def dl(t):
    
    return 0.5*np.cos(t)

def phi(t):
    return 0.0

def V(OL,t):
    return OL.V

def OL_driving_x(OL,x,t):
    
    V =100.1*np.ones(x.shape)
    #0.1*x*np.cos(OL.omega*t)#-1.2*np.ones(x.shape[0])#0.000#5*np.cos(2.0*np.pi*x+OL.phi_x)
    #V = OL.V*np.cos(OL.k_x*x+OL.phi_x)#0.5*OL.V_(t)*np.cos(2.0*(OL.k_x + OL.DK(t))*x+2.0*(OL.DK(t) + OL.phi_x))
    return V


def OL_driving(OL,x,t):
    
    #V = np.cos(t)*x
    #V = 5.1*x*np.cos(OL.omega*t)#-1.2*np.ones(x.shape[0])#0.000#5*np.cos(2.0*np.pi*x+OL.phi_x)
    V = 0.5*(OL.V_(t)*np.cos(2.0*OL.DK(t)*x+2.0*OL.Dphi(t)) - OL.V)*np.cos(2*2*np.pi*x + 2*OL.phi_x) +  0.5*OL.V_(t)*(1.0 - np.sin(2.0*np.pi*x+2.0*OL.phi_x)*np.sin(2.0*OL.DK(t)*x+2.0*OL.Dphi(t)))
    return V

def U_t(t,x,OL,U_kn_x):
    # t : instant of time
    # x : np.linspace(x_l,x_r,N_x)
    # U_kn: Space representation of the Bloch functions
    
    V_x = np.diag(OL_driving(OL,x,t))
    U   = np.transpose(np.conjugate(U_kn_x))@V_x@U_kn_x
   # U   = np.diag(1.0*np.ones(U_kn_x.shape[1]))
    return U

   # U   = np.abs(np.transpose(np.conjugate(U_kn_x))@U_kn_x)
    
#plt.contourf(np.abs(np.transpose(np.conjugate(U_kn_x))@U_kn_x))
#def Schrodinger_RHS(psi,t,E_0,V_0): # for odeint
def Schrodinger_RHS(t,psi,E_0,x,OL,U_kn): # for ode
    t_   = OL.omega*t
    jj   = np.complex(0.0,1.0)
    H_0  = np.diag(E_0)
    V    = U_t(t_,x,OL,U_kn)
    H    = H_0 + V
    #print(t_)
    dpsidt_RHS = -jj*H@psi
    
    return dpsidt_RHS


def Schrodinger_Momentum_RHS(t,psi,H_0,x,OL,U_kn): # for ode
    t_   = OL.omega*t
    jj   = np.complex(0.0,1.0)
    #H_0  = np.diag(E_0)
    V    = U_t(t_,x,OL,U_kn)
    H    = H_0 + V
    #print(t_)
    dpsidt_RHS = -jj*H@psi
    
    return dpsidt_RHS

def Schrodinger_RHS_position(t,psi,x,OL): # for ode
    t_   = OL.omega*t
    jj   = np.complex(0.0,1.0)
    dx   = np.abs(x[0]-x[1])
    H_0  = (-2.0*np.diag(np.ones([x.shape[0]])) + np.diag(np.ones([x.shape[0]-1]), -1) + np.diag(np.ones([x.shape[0] -1]), 1))/(dx*dx)
    V    = np.diag(OL_driving_x(OL,x,t_))
    H    = H_0 + V
    dpsidt_RHS = -jj*H@psi
    
    return dpsidt_RHS




class OpticalLattice():
    def __init__(self,V=1.0,phi_x=0,omega=1.0):                
        self.V     = V      # Depth
        self.k_x   = 2.0*np.pi
        self.phi_x = phi_x  # phase  V cos^2(kx+phi)
        self.omega = omega  # driving frequency
        
    
    def DK(self,t):
        return 2.0*np.pi/(1.0+dl(t)) - 2.0*np.pi
    
    def Dphi(self,t):
        return phi(t) - self.phi_x
    
    def V_(self,t):
        return V(self,t)
        

class Bloch_spectrum(object):

    def __init__(self,L=16,G=3):
        self.L  = L  # Number of lattice periods, assert it is even ==  number of values of momentum
        self.G  = G  # Number of Bands ASSERT G IS ODD
        self.D = G*(L+1)
        self.H_0 = np.zeros([(L+1)*G,(L+1)*G],dtype=complex)
        #self.H = np.zeros([G,G],dtype=np.complex)
    
    def Spectrum(OL,self):  
        # OL input: Optical lattice class
        # self input:Bloch_spectrum class
        #
        # Bloch_energy[i,n] : energy  of the i-th state of the n-th band 
        # Bloch_wavefun[i*G:(i_+1)*G,n] : wavefunction of the i-th state of the n-th band, with G bands.
        G  = self.G
        L  = self.L
        V  = OL.V
        jj = np.complex(0.0,1.0)
        Bloch_energy  = []#np.zeros([G],dtype=np.float)
        Bloch_wavefun = np.empty([G,G])
        V_ = np.zeros([G,G],dtype=np.complex)
        K  = np.zeros([G,G],dtype=np.complex)
        
        # FOR EACH QUASIMOMENTUM (m)
        for m in np.linspace(-L/2,L/2,L+1):
            i_ = 0
            for n in np.linspace(-(G-1)/2,(G-1)/2,G):        
                momentum   = 2.0*(1.0*m/(L+1) - n) + 0.01*np.pi/(L+1)
                K[i_,i_]   = 0.5*(2.0*np.pi)*(2.0*np.pi)*momentum*momentum # KINETIC ENERGY 
                if (i_+1 < G):
                    V_[i_,i_+1] = 0.25*V*np.exp( 2.0*jj*OL.phi_x) #0.125   # POTENTIAL ENERGY         
                    V_[i_+1,i_] = 0.25*V*np.exp(-2.0*jj*OL.phi_x) #0.125   # POTENTIAL ENERGY         
               # if (i_+2 < G):
               #     V_[i_,i_+2] = 2.25*V#0.125   # POTENTIAL ENERGY         
               #     V_[i_+2,i_] = 2.25*V#0.125   # POTENTIAL ENERGY         
               # if (i_+3 < G):
               #     V_[i_,i_+3] = 0.25*V#0.125   # POTENTIAL ENERGY         
               #     V_[i_+3,i_] = 0.25*V#0.125   # POTENTIAL ENERGY         

                i_=i_+1      
                
            H  = K + V_  
            self.H_0[int((m+int(L/2))*G):int((m+int(L/2)+1)*G),int((m+int(L/2))*G):int((m+int(L/2)+1)*G)] = H
            if(m == -L/2):
                Bloch_energy,Bloch_wavefun  = np.linalg.eigh(H)
            else:
                Bloch_energy  = np.concatenate([Bloch_energy,np.linalg.eigh(H)[0]],axis=0)     
                Bloch_wavefun = np.concatenate([Bloch_wavefun,np.linalg.eigh(H)[1]],axis=0)     
            
        Bloch_energy   = np.reshape(Bloch_energy,[L+1,G])            

        return Bloch_energy,Bloch_wavefun
    
    #def RealSpaceWavefun(self,x_l,x_r,N_x,BlochWavefun):
    def RealSpaceWavefun(self,x,BlochWavefun):
        # self         : Bloch_Spectrum class
        # x            : domanin of the wave function
        # BlochWavefun : Momentum decoposition of the Floquet states
        # 
        # OUT
        #
        # RealSpaceBlochfun[i_*N_x:(i_+1)*N_x,n]: Wavefunction of the i-th state of the n-th band
        
        Bands = self.G
        L     = self.L
        N_x   = x.shape[0]
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

        k      = 2.0* np.pi * np.linspace(-(Bands-1)/2,(Bands-1)/2,Bands)
        #x      = np.linspace(x_l,x_r,N_x)
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
        RealSpaceMomentumfun = np.zeros([N_x*(L+1),Bands],dtype=np.complex)
        for i_ in range(L+1): # loop over the values of momentum in the BZ
            RealSpaceBlochfun_ = np.zeros([N_x,Bands],dtype=np.complex)
            RealSpaceMomentumfun_ = np.zeros([N_x,Bands],dtype=np.complex)
            m = i_ - int(L/2)
            for n in range(Bands): # loo over the number of bands
                n_ = n - (Bands-1)/2
                #print(m)
                if n == 0 :
                    momentum = 2.0*np.pi*(m/(L+1))
                if (n-int(n/2)*2) != 0:
                    if m < 0:
                        momentum = 2.0*np.pi*(m/(L+1)) + 2.0*np.pi*int((n+1)/2) + 0.1*np.pi/(L+1)
                    if m > 0:
                        momentum = 2.0*np.pi*(m/(L+1)) - 2.0*np.pi*int((n+1)/2) + 0.1*np.pi/(L+1)
                    if m == 0: 
                        momentum = -(-2.0*np.pi*(m/(L+1)) - 2.0*np.pi*int((n+1)/2)) + 0.1*np.pi/(L+1)
                if (n-int(n/2)*2) == 0:
                    if m < 0:
                        momentum = 2.0*np.pi*(m/(L+1)) - 2.0*np.pi*int(n/2) + 0.1*np.pi/(L+1)
                    if m > 0:
                        momentum = 2.0*np.pi*(m/(L+1)) + 2.0*np.pi*int(n/2) + 0.1*np.pi/(L+1)
                    if m == 0:
                        momentum = -(2.0*np.pi*(m/(L+1)) + 2.0*np.pi*int(n/2)) + 0.1*np.pi/(L+1)
                momentum = 2.0*momentum  + 0.01*np.pi/(L+1)

                #print(L,i_,n,m,n_,momentum)#,momentum/np.pi)        
                #if n=0 :
                #    momentum = 2.0*np.pi*(m/L) + (n-1)*np.pi
                #momentum = momentum/np.pi 
                #momentum = (1.0*m/L - n_)
                #print(m,n,momentum/np.pi)#,momentum/np.pi)        
                for m_ in range(Bands): #sum_m exp(-i k_m x) U_{i_,m,n}
                    RealSpaceBlochfun_[:,n] = RealSpaceBlochfun_[:,n] + BlochWavefun[i_*Bands+m_,n]*np.exp(-jj*k[m_]*x)
                    
                #RealSpaceBlochfun_[:,n] = momentum*np.ones(x.shape[0])
                RealSpaceBlochfun_[:,n]    = np.exp(jj*momentum*x)*RealSpaceBlochfun_[:,n]     
                RealSpaceMomentumfun_[:,n] = np.exp(jj*momentum*x)

            RealSpaceBlochfun[i_*N_x:(i_+1)*N_x,:]    = RealSpaceBlochfun_
            RealSpaceMomentumfun[i_*N_x:(i_+1)*N_x,:] = RealSpaceMomentumfun_

                        
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
           
        return RealSpaceBlochfun/np.sqrt(x.shape[0]),RealSpaceMomentumfun/np.sqrt(x.shape[0])#RealSpaceWF

    