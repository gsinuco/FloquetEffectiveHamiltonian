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
from scipy.integrate import complex_ode

#################################################################
########  SCHRODINGER EQUATION ##################################
#################################################################

def cart2polar(lambda_u,N):
            
    e_u = np.zeros(N,dtype=np.float64)
    for i in range(N):
        if (np.imag(lambda_u[i]) < 0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))
        if (np.imag(lambda_u[i]) < 0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+2.0*np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = np.pi/2.0
        if (np.imag(lambda_u[i]) <0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = 3.0*np.pi/2.0
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = 0.0
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = np.pi
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = 0.0

    return e_u

def UnfoldingFloquetSpectrum(L=1,dt=1.0,N_t=0,phi_t=0):
    # Takes the time-evolution of the Floquet phases, defined with the eigenvalues of
    # the time-evolution operator:
    # Calculate U(t,t_0)
    # evaluate the eigenvalues
    # define the phase as
    # phi(t) = atan(imag(eigenvalue)/real(eigenvalues))
    # 
    phase_sec_order_aux = UnfoldingFloquetSpectrum_aux(L,dt,N_t,phi_t)
    phases = np.copy(phase_sec_order_aux)
    for j_ in range(L):
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
                    #print("A",N_t-2-(i+1),grad_ref,grad_new)
                    #Evaluate gradients after the jump
                    grad_new_ = phase_sec_order_aux[:,N_t-2-(i-2)] - phase_sec_order_aux[:,N_t-2-(i-3)]    
                    #Identify the phase index with the gradient closest to the previous     one                
                    new_index_grad = np.argmin(np.abs(grad_new_-grad_ref))            
                    new_index_value = np.argmin(np.abs(phase_sec_order_aux[:,N_t-2-(i+1)-1] - phases[j_,N_t-2-(i+1)]))
                    if(new_index_grad==new_index_value):
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_grad,0:N_t-2-(i+1)])
                        #grad_ref = grad_new_[new_index_grad]
                    else:
                        phases[j_,0:N_t-2-(i+1)] = np.copy(phase_sec_order_aux[new_index_value,0:N_t-2-(i+1)])
                    #grad_ref = grad_new_[new_index_grad]
                # check if the jump is positive                
                if(grad_new>0):
                    #print("B",N_t-2-(i+1),grad_ref,grad_new,grad_new_2)
                    #Evaluate gradients after the jump
                    grad_new_ = phase_sec_order_aux[:,N_t-4-i] -    phase_sec_order_aux[:,N_t-5-i]
                    grad_new_2 = phase_sec_order_aux[:,N_t-5-i] - phase_sec_order_aux[:,N_t-6-i]
                    new_value = -grad_ref + phases[j_,N_t-2-(i+1)]
                    #Identify the phase index with the gradient closest to the previous one
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
 
    return phases


def UnfoldingFloquetSpectrum_aux(L=1,dt=1.0,N_t=0,phi_t=0):
    # Takes the time-evolution of the Floquet phases, defined with the eigenvalues of
    # the time-evolution operator:
    # Calculate U(t,t_0)
    # evaluate the eigenvalues
    # define the phase as
    # phi(t) = atan(imag(eigenvalue)/real(eigenvalues))
    # 
    # HERE I DO PRELIMINAR UNTANGLING, BEFORE GOING STATE BY STATE
    
    time = np.linspace(1,N_t,N_t)
    N_t_0 = 0 
    N_t_  = N_t-1

    phase_sec = phi_t[:,N_t_0:N_t_]
    phase_sec[0,:]
    phase_sec_=phase_sec

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
        if(index_order_new[0] != index_order_old[0]):
            index_ = index_order_old[L-1]
            grad_last = (phase_sec[index_,i-1] - phase_sec[index_,i-2])/dt
            phase_future = np.real(grad_last*dt + phase_sec[index_,i-1])
            grad_ = (phase_sec[:,i-1] - phase_sec[:,i-2])/dt
            phase_ = np.real(grad_*dt + phase_sec[:,i-1])
            #print(i,phase_future)
            if(phase_future > 2.0*np.pi):
                swap_flag= True
                if(index_order_new[0] == index_order_old[L-1]):
                    kinks = np.append(kinks,[np.int(i)])
                    order_ = np.append(order_,index_order_new)
                    swap_counter = np.append(swap_counter,np.array(np.where(phase_>2.0*np.pi)).size)

                if(index_order_new[0] != index_order_old[L-1]):
                    kinks = np.append(kinks,[np.int(i)])
                    order_ = np.append(order_,[index_order_new])
                    swap_counter = np.append(swap_counter,np.array(np.where(phase_>2.0*np.pi)).size)
        phase_sec_order[:,i] = np.sort(np.abs(phase_sec[:,i]))
        index_order_old = index_order_new

    phase_sec_order_ = np.zeros(phase_sec_order.shape)
    phase_sec_order_[:] = phase_sec_order[:]
    
    l=0
    i_l = 0
    for i in range(kinks.size-1):
        l += 1
        i_r = np.int(kinks[i])
        phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l,axis=0)
        i_l = i_r
        
    if(kinks.size == 1):
        i_r = np.int(kinks[0])

    i_l = i_r
    i_r = phase_sec_order.shape[1]
    phase_sec_order_[:,i_l:i_r] = np.roll(phase_sec_order[:,i_l:i_r],-l-1,axis=0)
    
    
    counter = 0
    phase_sec_order_aux = np.copy(phase_sec_order_)
    phase_sec_order__   = np.copy(phase_sec_order_)
    grad_old = phase_sec_order_[:,1] - phase_sec_order_[:,0]
    index_order_old = np.argsort(np.abs(grad_old))

    swap_time_old = [0,0,0]
    swap_time_new = [0,0,0]
    for i in range(phase_sec_order_aux.shape[1]-2):   
        grad_new = phase_sec_order_aux[:,i+2] - phase_sec_order_aux[:,i+1]    
        dgrad = grad_new-grad_old
        index_order_old = np.argsort(np.abs(grad_old))
        index_order_new = np.argsort(np.abs(grad_new))
        #  COMPARE GRADIENTS. IF THEY ARE DIFFERENT, THERE WAS A COLLISION        
        swap_flag = False
        if(np.linalg.norm(grad_new-grad_old)<1.999*np.pi): 
            swap_flag = True
            swap_index     = np.argsort(np.abs(dgrad))[L-2:L]
            
        if(np.linalg.norm(grad_new-grad_old)>1.999*np.pi): 
            swaps_counter = np.array((np.where(np.abs(dgrad)>1E-6)))[0,:]
            if(swaps_counter.size>1 and np.mod(swaps_counter.size,2) == 1):            
                if(swaps_counter.size>1):
                    swap_flag      = True
                    swap_index     = np.argsort(np.abs(grad_new-grad_old))[L-3:L-1]        
        if(np.linalg.norm(grad_new-grad_old)>5.6E-4 and np.linalg.norm(grad_new-    grad_old)<1.999*np.pi):
        # THE COLLIDING PARTICLES CHANGE THEIR GRADIENT THE MOST
            swap_time_new  = [i+2,swap_index[0],swap_index[1]]
            if(swap_time_new[0] == swap_time_old[0]+1 and swap_time_new[2]==swap_time_old[1] and swap_time_new[1]==swap_time_old[2]):
                a=1
            else:
                phase_sec_order_aux[swap_index[0],i+1::] = np.copy(phase_sec_order__[swap_index[1],i+1::]) 
                phase_sec_order_aux[swap_index[1],i+1::] = np.copy(phase_sec_order__[swap_index[0],i+1::]) 
                grad_new = phase_sec_order_aux[:,i+2] - phase_sec_order_aux[:,i+1]
                phase_sec_order__ = np.copy(phase_sec_order_aux)
                swap_time_old  = swap_time_new
        grad_old = grad_new


    grad_new_ = np.zeros([L,N_t],dtype=np.float32)
    for i in range(phase_sec_order_aux.shape[1]-2):   
        grad_new_[:,i+1] = phase_sec_order_aux[:,i+1] - phase_sec_order_aux[:,i]    
    
    return phase_sec_order_aux


def FloquetStroboscopicOperatorV4(L=1,N_Bands=1,t0=0,DT=1.0,N_t=0,x=0,OL=0,BlochSpectrum=0,RealSpaceWavefun=0):
    # Solve the Schrodinger equation for a subset of the bands:
    #N_Bands = 2 # number of bands to include in the Schrodinger equation
    # here we store the solution step by step, to track the time evolution of the
    # quasienergies
    # For each time step, we built U_T and diagonalise
    # then we plot phi vs t


    #set the initial condition and parameters to integrate the Schrodinger equation
    #psi0 = np.ones(E_0.size)
    #t0   = 0   # initial time
    #DT   = 6.0 # total time to integrate
    t1   = t0 + DT # final time 
    #N_t  = 128   # number of time steps
    dt   = DT/N_t # time step
    #t    = np.linspace(t0,t0+DT, N_t) 
    #L    = BlochSpectrum.shape[0]/RealSpaceWavefun.shape[1]
    N_x  = RealSpaceWavefun.shape[0]
    
    E_0 = BlochSpectrum[0:int(N_Bands*L)]
    E_0 = E_0 - np.min(E_0)

    #E_0 = np.linspace(1,int(N_Bands*L),int(N_Bands*L))#BlochSpectrum[0:int(N_Bands*L)]
    U_x = RealSpaceWavefun[:,0:int(N_Bands*L)]

    # SET THE INTEGRATOR
    case = OLM.Schrodinger(OLM.Schrodinger_RHS, fargs=[E_0,x,OL,U_x])

    solver = complex_ode(case.f)
    solver.set_integrator('dopri5')#,method='bdf')
    #U_T = np.zeros([int(N_Bands*L),int(N_Bands*L)],dtype=np.complex)
    # initial condition
    U_T = np.diag(np.ones([int(N_Bands*L)],dtype=np.complex))

    i_ = 0
    t = t0
    lambda_u_t = np.zeros(N_Bands*L,dtype=np.complex)
    phi_t      = np.zeros([N_Bands*L,N_t],dtype=np.complex)
    #while solver.successful() and solver.t<t1  and i_+1<N_t:
    for j_ in range(int(N_Bands*L)):
            solver.set_initial_value(U_T[:,j_],t0)
            solver.integrate(t1)
            U_T[:,j_] = np.array(sol)            
            lambda_u_t,U = np.linalg.eig(U_T)
            phi_t[:,i_] = cart2polar(lambda_u_t,N_Bands*L)
     
    return phi_t,U_T,U_x


def FloquetStroboscopicOperatorV3(L=1,N_Bands=1,t0=0,DT=1.0,N_t=0,x=0,OL=0,BlochSpectrum=0,RealSpaceWavefun=0):
    # Solve the Schrodinger equation for a subset of the bands:
    #N_Bands = 2 # number of bands to include in the Schrodinger equation
    # here we store the solution step by step, to track the time evolution of the
    # quasienergies
    # For each time step, we built U_T and diagonalise
    # then we plot phi vs t


    #set the initial condition and parameters to integrate the Schrodinger equation
    #psi0 = np.ones(E_0.size)
    #t0   = 0   # initial time
    #DT   = 6.0 # total time to integrate
    t1   = t0 + DT # final time 
    #N_t  = 128   # number of time steps
    dt   = DT/N_t # time step
    #t    = np.linspace(t0,t0+DT, N_t) 
    #L    = BlochSpectrum.shape[0]/RealSpaceWavefun.shape[1]
    N_x  = RealSpaceWavefun.shape[0]
    
    E_0 = BlochSpectrum[0:int(N_Bands*L)]
    E_0 = E_0 - np.min(E_0)

    #E_0 = np.linspace(1,int(N_Bands*L),int(N_Bands*L))#BlochSpectrum[0:int(N_Bands*L)]
    U_x = RealSpaceWavefun[:,0:int(N_Bands*L)]

    # SET THE INTEGRATOR
    solver = ode(OLM.Schrodinger_RHS).set_integrator('zvode',method='bdf')
    #U_T = np.zeros([int(N_Bands*L),int(N_Bands*L)],dtype=np.complex)
    # initial condition
    U_T = np.diag(np.ones([int(N_Bands*L)],dtype=np.complex))

    i_ = 0
    t = t0
    lambda_u_t = np.zeros(N_Bands*L,dtype=np.complex)
    phi_t      = np.zeros([N_Bands*L,N_t],dtype=np.complex)
    while solver.successful() and solver.t<t1  and i_+1<N_t:
        for j_ in range(int(N_Bands*L)):
            solver.set_initial_value(U_T[:,j_],t).set_f_params(E_0,x,OL,U_x)
            U_T[:,j_] = solver.integrate(solver.t+dt)        
        lambda_u_t,U = np.linalg.eig(U_T)
        phi_t[:,i_] = cart2polar(lambda_u_t,N_Bands*L)
        i_ = i_+1
     
    return phi_t,U_T,U_x


def FloquetStroboscopicOperatorV2(L=1,N_Bands=1,t0=0,DT=1.0,N_t=0,x=0,OL=0,BlochSpectrum=0,RealSpaceWavefun=0):
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
    #L    = BlochSpectrum.shape[0]/RealSpaceWavefun.shape[1]
    N_x  = RealSpaceWavefun.shape[0]
    
    E_0 = BlochSpectrum[0:int(N_Bands*L)]
    E_0 = E_0 - np.min(E_0)

    #E_0 = np.linspace(1,int(N_Bands*L),int(N_Bands*L))#BlochSpectrum[0:int(N_Bands*L)]
    U_x = RealSpaceWavefun[:,0:int(N_Bands*L)]

    # SET THE INTEGRATOR
    solver = ode(OLM.Schrodinger_RHS).set_integrator('zvode',method='bdf')
    U_T = np.zeros([int(N_Bands*L),int(N_Bands*L)],dtype=np.complex)


    for j_ in range(int(N_Bands*L)):
        psi0 = np.zeros(E_0.size,dtype=np.complex)
        psi0[j_] = 1.0
            
        # define the parameters of the integrator
        # TO DO: TEST THE INTEGRATOR
    
        solver.set_initial_value(psi0,t0).set_f_params(E_0,x,OL,U_x)
        
        psi_t = 1.0
        i_ = 0
        while solver.successful() and solver.t<t1  and i_+1<N_t:
            i_ = i_+1            
            psi_t = solver.integrate(solver.t+dt)
            #if j_ == int(N_Bands*L) - 1:
            #    print(psi_t[j_])
            
                                
        U_T[:,j_] = psi_t

    return U_T,U_x


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
    e_u = np.zeros([lambda_u.shape[0]],dtype=np.float64)
    #e_u = cart2polar(lambda_u,lambda_u.shape[0]):
    for i in range(lambda_u.shape[0]):
        if (np.imag(lambda_u[i]) < 0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))
        if (np.imag(lambda_u[i]) < 0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = -np.arctan(np.imag(lambda_u[i])/np.real(lambda_u[i]))+2.0*np.pi
        if (np.imag(lambda_u[i]) >0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = np.pi/2.0
        if (np.imag(lambda_u[i]) <0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = 3.0*np.pi/2.0
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) > 0):
            e_u[i] = 0.0
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) < 0):
            e_u[i] = np.pi
        if (np.imag(lambda_u[i]) == 0) & (np.real(lambda_u[i]) == 0):
            e_u[i] = 0.0
        
    #e_u = -np.arctan(np.imag(lambda_u)/np.real(lambda_u))
    e_u = e_u/DT
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
        