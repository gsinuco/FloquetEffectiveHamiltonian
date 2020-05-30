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


# Returns a hameltoninan modulated by V_period L times
def get_H(N, L, V_period):
    H = np.empty(3*N, dtype=np.float)
    I = np.empty(3*N, dtype=int)
    J = np.empty(3*N, dtype=int)

    x = np.linspace(0, L, N)
    dx = L / N

    H[1::3] = 2/dx**2 + V_period(x%1) # diagonal elements of the Hamiltonian
    H[0::3] = H[2::3] = -1/dx**2 * np.ones_like(x) # off diagonal elements of the H
    I[1::3] = J[1::3] = I[0::3] = I[2::3] = np.arange(0, N)
    J[0::3] = (np.arange(0, N) - 1)%N
    J[2::3] = (np.arange(0, N) + 1)%N
    
    return coo_matrix((H, (I, J)))

# The eigenspace for each energy is degenerate, with one right and one left moving wave
# for each E. This changes the pasis from having u_k*cos(kx), u_k*sin(k0) to u_k*exp(\pm ikx)
def change_basis(v):
    v_new = np.empty_like(v, dtype = np.complex_)
    v_new[:, 0] = v[:, 0]
    for i in range(1, len(v[0])//2 - 1):
        v_new[:, 2*i - 1] = v[:, 2*i] + 1j*v[:, 2*i-1]
        v_new[:, 2*i] = v[:, 2*i] - 1j*v[:, 2*i-1]
    return v_new


# The energy eigenstates are also eigenstates of the translation vector T_n, 
# where T_n psi_k(x) = psi_k(a*n+x). With T_n psi_k(x) = exp(ikna)psi_k(x) 
# => k = arg(pis_k(x)/psi(k+a))/a. Here, a = 1, f(a) = f[m]
def get_k(v_i, m):
    return np.angle(v_i[0]/ v_i[m])


# Get eigen-vecs/vals of potential wiht v_period repeated L times,
# and periodic bondaryconditions. Returns numVec values/vectors. 
def get_eigen(N, L, numVec, V_period):
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
    V0 = 5
    sides = (1-size)/2
    V = np.zeros_like(x)
    mask = np.logical_and(x>sides, x<(sides+size))
    V[mask] = V0*np.ones_like(x[mask])
    return V

def V_sin(freqs):
    V0 = 5
    return lambda x : np.sum([V0*cos(2*pi*x*i) for i in range(freqs+1)], axis=0)


def BlochSpectrum(L=64,m=32,Bands=2,n=1):
    #L = 64 # Number of periods in lattice
    #m = 32 # Number of points in a period
    N = m*L
    numVec = Bands*L
    #n = 1
    Vs = [V_sin(n)]#[V_sin(i) for i in range(n)]
    for i in [0]:#range(n):
        l, v = get_eigen(N, L, numVec, Vs[i])
        k_u = plot_eigenvals(l, v, m, "vals_{}".format(i))
        #plot_eigenvecs(v, 4, N, L, Vs[i], "vecs_{}".format(i))
    return l,v
    
L     = 64
N_x   = 32
Bands = 3
n     = 4
l,v = BlochSpectrum(L,N_x,Bands,n)
