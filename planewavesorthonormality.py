#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 06:47:12 2020

@author: german
"""

# orthonormality of plane waves

import numpy as np

x_l = -100.0
x_r =  100.0
N_x =  128
x   =  np.linspace(x_l,x_r,N_x)

f = np.empty([1,1],dtype=float)
for dlambda in np.linspace(1E-3,10,128):
    #dlambda = 
    lambda1 = 1.7625
    lambda2 = lambda1 + dlambda

    x_l =  -10000.0#/dlambda
    x_r =   10000.0#/dlambda
    N_x =  2048
    x   =  np.linspace(x_l,x_r,N_x)
    
    
    k1 = 2*np.pi/lambda1
    k2 = 2*np.pi/lambda2
    f1 = np.exp(1j*x*k1)/np.sqrt(x.shape[0])
    f2 = np.exp(1j*x*k2)/np.sqrt(x.shape[0])
    #f = [dlambda,np.abs(np.vdot(f1,f2))]
    
    #print(np.vdot(f2,f2))
    #print(np.vdot(f1,f1))
    print(np.abs(np.vdot(f1,f2)))