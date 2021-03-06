#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:18:05 2020

@author: german
"""

import numpy as np
from numpy import cos, sin, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
from progress.bar import Bar


# Returns a Hamiltoninan modulated by V_period L times
def get_H(N, L, V_period):
    H = np.empty(3*N, dtype=np.float)
    I = np.empty(3*N, dtype=int)
    J = np.empty(3*N, dtype=int)

    #x = np.linspace(0, L, N)
    x = np.linspace(-L/2, L/2, N)
    dx = L / N
    if(isinstance(V_period, np.ndarray)):
        H[1::3] = 2/dx**2 + V_period    # diagonal elements of the Hamiltonian
    else:
        H[1::3] = 2/dx**2 + V_period(x) # diagonal elements of the Hamiltonian
    
    H[0::3] = H[2::3] = -1/dx**2 * np.ones_like(x) # off diagonal elements of the H
    I[1::3] = J[1::3] = I[0::3] = I[2::3] = np.arange(0, N)
    J[0::3] = (np.arange(0, N) - 1)%(N)
    J[2::3] = (np.arange(0, N) + 1)%(N)
    
    return coo_matrix((H, (I, J)))

# The eigenspace for each energy is degenerate, with one right and one left moving wave
# for each E. This function changes the basis from having u_k*cos(kx), u_k*sin(k0) to u_k*exp(\pm ikx)
def change_basis(v):
    v_new = np.empty_like(v, dtype = np.complex_)
    v_new[:, 0] = v[:, 0]
    for i in range(1, len(v[0])//2 - 1):
        v_new[:, 2*i - 1] = v[:, 2*i] + 1j*v[:, 2*i-1]
        v_new[:, 2*i]     = v[:, 2*i] - 1j*v[:, 2*i-1]
    return v_new


# The energy eigenstates are also eigenstates of the translation vector T_n, 
# where T_n psi_k(x) = psi_k(a*n+x). With T_n psi_k(x) = exp(ikna)psi_k(x) 
# => k = arg(pis_k(x)/psi(k+a))/a. Here, a = 1, f(a) = f[m]
def get_k(v_i, m):
    return np.angle(v_i[0]/ v_i[m])


# Get eigen-vecs/vals of potential wiht v_period repeated L times,
# and periodic bondaryconditions. Returns numVec values/vectors. 
def get_eigen(N, L, numVec, V_period):
    #print(V_period.shape)
    print("numVec = {}, N = {}".format(numVec, N))
    print("Making hamiltonian")
    H = get_H(N, L, V_period)

    print("Finding eigenvalues")
    l, v = eigsh(H, k=numVec, which="SM")
    v = change_basis(v)
    return l, v

def plot_eigenvals(l, v, m, name):
    fig, ax = plt.subplots()
    numVec = len(l)
    k = np.empty(numVec)
    for i in range(numVec):
        k[i] = get_k(v[:, i], m)

    ax.plot(k, l, ".")
    plt.savefig("figs/" + name +".png")
    return k


# plots the n first vectors of v
def plot_eigenvecs(v, n, N, L, V_period, name):
    fig, ax = plt.subplots(n, sharex=True)
    ax2 = [a.twinx() for a in ax]
    x = np.linspace(0, L, N)
    for i in range(n):
        ax[i].plot(x, v[:, i].real)
        ax[i].plot(x, v[:, i].imag)
        ax2[i].plot(x, V_period(x%1), "k--", alpha=0.5)
    plt.savefig("figs/" + name + ".png")

    
# Potentials to repeat
def V_box(size=0.5):
    V0 = 1.0
    sides = (1-size)/2
    V = np.zeros_like(x)
    mask = np.logical_and(x>sides, x<(sides+size))
    V[mask] = V0*np.ones_like(x[mask])
    return V

def V_sin(freqs,V0,phi):
    #V0 = 10.0
    #return lambda x : np.sum([cos(2*pi*x*i) for i in range(freqs+1)], axis=0)    
    return lambda x : 0.5*V0 + 0.25*V0*(cos(2.0*pi*x+phi)+ cos(4.0*pi*x+phi))
        #0.5*V0*(2.0*cos(2*pi*x+phi)*cos(2*pi*x+phi) + 0.0*cos(2*2*pi*x+phi)*cos(2*2*pi*x+phi)) 

def V_OL_trap(i,V0,omega_trap,phi):
    return lambda x : 0.5*V0 + 0.5*V0*(cos(2.0*pi*x+phi)) + 0.5*omega_trap*(x**2)

def beta_n_p(n=2.0,p=4.0):
    limits = 1024
    m     = np.linspace(-limits,-1,limits)
    m_    = np.linspace(1,limits,limits)
    beta  = (2*n*n*(1-np.cos(2*pi*m/n)) -2.0*n*(2.0-2.0*np.cos(m*pi*(1.0+1.0/n) )*np.cos(m*pi*(1.0-1.0/n))) )/np.power(m,p)
    beta_ = (2*n*n*(1-np.cos(2*pi*m_/n))-2.0*n*(2.0-2.0*np.cos(m_*pi*(1.0+1.0/n))*np.cos(m_*pi*(1.0-1.0/n))))/np.power(m_,p)
    return (np.sum(beta)+np.sum(beta_))/np.power(2*np.pi*n,2.0)



def V_sin_array(freqs,V0,phi,omega,phi_0,phi_1):
    print(freqs)
    if(freqs == 0):
        return lambda x : 0.5*V0 + 0.5*V0*cos(4.0*pi*x+phi) # t \in [0,T/2)
    if(freqs == 1):
        return lambda x : 0.5*V0 + 0.5*V0*cos(2.0*pi*x+phi) # t \in [t/2,T)
    if(freqs == 2):
        return lambda x : 0.5*V0 + 0.25*V0*(cos(4.0*pi*x+phi)+ cos(2.0*pi*x+phi)) # V average
    if(freqs == 3):
        #H_F_3 =  beta_n_p(2.0,4.0)*0.5*np.power((V0/omega)*(-np.sin(2.0*pi*x+2.0*phi_0) + g*np.sin(2.0*2.0*pi*x+2.0*phi_1)),2.0)
        g = 2
        return lambda x :   0.5*V0 + 0.25*V0*(cos(4.0*pi*x+phi)+ cos(2.0*pi*x+phi)) + beta_n_p(2.0,4.0)*0.5*np.power((V0/omega)*(-np.sin(2.0*pi*x+2.0*phi_0) + g*np.sin(2.0*2.0*pi*x+2.0*phi_1)),2.0)


def BlochSpectrum(L=64,m=32,Bands=2,V=1.0,phi = 0):
    #L = 64 # Number of periods in lattice
    #m = 32 # Number of points in a period

    n = 2
    N = m*L
    numVec = Bands*L
    x = np.linspace(0, L, N)

    Vs = [V_sin(i,V,phi) for i in range(n)]
    for i in [n-2]:#range(n-1):
        l, v = get_eigen(N, L, numVec, Vs[i])
    return l,v


def Spectrum(L=64,m=32,Bands=2,V=1.0,phi = 0,omega_trap=0.0):
    #L = 64 # Number of periods in lattice
    #m = 32 # Number of points in a period

    n = 2
    N = m*L
    numVec = Bands*L
    x = np.linspace(0, L, N)

    Vs = [V_OL_trap(i,V,omega_trap,phi) for i in range(n)]
    for i in [n-2]:#range(n-1):
        l, v = get_eigen(N, L, numVec, Vs[i])
    return l,v


def BlochSpectrum_array(L=64,m=32,Bands=2,V=1.0,phi = 0,omega=10):
    #L = 64 # Number of periods in lattice
    #m = 32 # Number of points in a period

    n = 4
    N = m*L
    numVec = Bands*L
    x = np.linspace(0, L, N)
    phi_0 = 0
    phi_1 = 0
    
    l_ =  np.empty([1],dtype=np.double)
    v_ =  np.zeros([1,numVec],dtype=np.complex)
    
    Vs = [V_sin_array(i,V,phi,omega,phi_0,phi_1) for i in range(n)]
    for i in range(n):
        l, v = get_eigen(N, L, numVec, Vs[i])
        #plot_eigenvals(l, v, m, "vals_{}".format(i))        
        #plot_eigenvecs(v, 4, N, L, Vs[i], "vecs_{}".format(i))
        l_ = np.append(l_,l,axis=0)
        v_ = np.append(v_,v,axis=0)
        #plt.plot(Vs[i](x%1))
        #plt.show()
    return l_,v_



#L     = 32
#N_x   = 32
#Bands = 3
#V     = 30
#phi   = pi/4.0
#l,v = BlochSpectrum(L,N_x,Bands,V,phi)
