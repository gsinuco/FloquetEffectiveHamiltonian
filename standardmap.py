#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:23:52 2020

@author: german
"""

import numpy as np
import matplotlib.pyplot as plt

r_old = np.empty([2],dtype=np.float64)
r_new = np.empty([2],dtype=np.float64)

N_iterations  = 128
K  = 0.971635
N = 16
r_trajectory = np.empty([2,N_iterations],dtype=np.float64)

#Set Initial condition
for j in range(N):
    for k in range(N):
        r_old[0]  = (j-1)*2.0*np.pi/N
        r_old[1]  = (k-1)*2.0*np.pi/N 
        for i in range(N_iterations):
            r_new[0] = r_old[0] + K*np.sin(r_old[1])
            r_new[1] = r_old[1] + r_new[0]
            r_new = np.mod(r_new,2.0*np.pi)
            r_old = np.copy(r_new)
            r_trajectory[:,i] = np.copy(r_new) 
            plt.plot(r_trajectory[1,:]-np.pi,r_trajectory[0,:]-np.pi, ".",ms=1)
            plt.xlim(-np.pi,np.pi)
            plt.ylim(-np.pi,np.pi)
            #plt.xlim(-np.pi,2*np.pi)
            #plt.ylim(-np.pi,2*np.pi)
   #         plt.xlabel("q")
   #         plt.ylabel("theta")