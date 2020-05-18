#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:03:38 2020

@author: german
"""


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

    