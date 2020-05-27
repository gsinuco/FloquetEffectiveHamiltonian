#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:44:12 2020

@author: german
"""

#%%

import numpy as np
import OpticalLatticeModule as OLM
from scipy.integrate import ode

#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

def FloquetStroboscopicOperator(N_Bands=1,t0=0,DT=1.0,N_t=0,x=0,OL=0,BlochSpectrum=0,RealSpaceWavefun=0):
    # Solve the Schrodinger equation for a subset of the bands:
    #N_Bands = 2 # number of bands to include in the Schrodinger equation


    #set the initial condition and parameters to integrate the Schrodinger equation
    #psi0 = np.ones(E_0.size)
    #t0   = 0   # initial time
    #DT   = 6.0 # total time to integrate
    t1   = t0 + DT # final time 
    #N_t  = 128   # number of time steps
    dt   = DT/N_t # time step
    #t    = np.linspace(t0,t0+DT, N_t) 
    L    = BlochSpectrum.shape[0] - 1
    N_x  = int(RealSpaceWavefun.shape[0]/(L+1))
    


    E_0 = BlochSpectrum[:,0]
    for i_ in range(N_Bands-1):
        E_0 = np.concatenate([E_0,BlochSpectrum[:,i_+1]],axis=0) # two bands

    U_x = np.transpose(np.reshape(RealSpaceWavefun[:,0],[L+1,N_x]))
    for n in range(N_Bands-1):
        U_x_  = np.transpose(np.reshape(RealSpaceWavefun[:,n+1],[L+1,N_x]))
        U_x = np.concatenate([U_x,U_x_],axis=1)
               
    #U_x = np.zeros([N_x,(L+1)*N_Bands],dtype = np.complex)
    #for n in range(N_Bands):
    #    for i_ in range(L+1):
    #        U_x[:,n*N_Bands+i_] = RealSpaceWavefun[i_*N_x:(i_+1)*N_x,n]
    
    # SET THE INTEGRATOR
    solver = ode(OLM.Schrodinger_RHS).set_integrator('zvode',method='bdf')

    U_T = np.zeros([N_Bands*(L+1),N_Bands*(L+1)],dtype=np.complex)
    #psi_t_ = np.zeros([N_t+1,E_0.size],dtype=np.complex) # Initialise the time dependent wavefunction 

    for j_ in range(N_Bands*(L+1)):
        psi0 = np.zeros(E_0.size,dtype=np.complex)
        psi0[j_] = 1.0#np.complex(0.0,1.0)
            
        # define the parameters of the integrator
        # TO DO: TEST THE INTEGRATOR
    
        solver.set_initial_value(psi0,t0).set_f_params(E_0,x,OL,U_x)


        #if j_ == 0 :
        #    psi_t_[0,:] = psi0
        
        psi_t = np.zeros([E_0.size],dtype=np.complex) # Initialise the time dependent wavefunction 
        i_ = 0
        while solver.successful() and solver.t<t1  and i_+1<N_t:
            i_ = i_+1            
            psi_t = solver.integrate(solver.t+dt)
            #if j_ == 0:
            #   psi_t_[i_,:] = psi_t
                                
        U_T[:,j_] = psi_t

    return U_T,U_x#,psi_t_


def EffectiveFloquetHamiltonian(U_T = 0, DT = 1.0,U_x = 0.0):
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
    #index_ = np.int(np.argmax(U,axis=1)/U_T.shape[0])
    
    #print(lambda_u)
    e_u = -np.arctan(np.imag(lambda_u)/np.real(lambda_u))/DT
    
    #plt.plot(k,e_u,k,BlochSpectrum[:,0])
    #plt.show()
    
    H_eff_F = np.diag(e_u) # Effective Hamiltonian in the basis Floquet states
    
    #Transformation from the Floquet basis to the basis of Bloch states
    H_eff_kn = U@H_eff_F@np.transpose(np.conjugate(U))#np.transpose(np.conjugate(U))@H_eff_F@U
    
    #U_x = np.empty([N_x,N_Bands*(L+1)],dtype=np.complex)
    
    #n = 0
    #plt.plot(x,np.abs(RealSpaceWavefun[n*N_x:(n+1)*N_x,0:4]))
    #plt.show()
    
    #U_x = np.reshape(RealSpaceWavefun[:,1],[N_x,L+1])
    #RealSpaceWavefun[:,0],[N_x,L+1])
    #for n in range(N_Bands):
    #    U_x[:,n*(L+1):(n+1)*(L+1)] = np.reshape(RealSpaceWavefun[:,n],[N_x,L+1])
    
    H_eff_x = U_x@H_eff_kn@np.transpose(np.conjugate(U_x))
    
    #V_eff_x = np.diag(H_eff_x) #- np.diag(KineticEnergy)
    
    #plt.plot(x,V_eff_x)
    #plt.show()
    
    return lambda_u,U,e_u,H_eff_kn,H_eff_x
        