#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:01:44 2020

@author: german
"""

import numpy as np
import matplotlib.pyplot as plt

N_x = 1024
x = np.linspace(-8.0,8.0,N_x)
V = np.empty([N_x,2],dtype=np.complex)

V[:,0] = np.cos(2*np.pi*x+np.pi/3.314) + np.cos(2.0*2*np.pi*x+np.pi/5.314)
V[:,1] = np.cos(3.75*2*np.pi*x+np.pi/3.314) + np.cos(4.0*2*np.pi*x+np.pi/5.314)

#plt.plot(x,V)
#plt.show()
for i in [0,1]:
    V_F = np.fft.fft(V[:,i])/N_x
    k   = np.fft.fftfreq(N_x)/(16.0/N_x)
    #plt.plot(k,np.real(V_F),k,np.imag(V_F))
    #plt.show()

    V_F_shifted = np.fft.fftshift(V_F)
    k_shifted = (np.linspace(-N_x/2,N_x/2-1,N_x))*(64.0/N_x)

    plt.plot(k_shifted[int(N_x/2)-100:int(N_x/2)+100],np.real(V_F_shifted)[int(N_x/2)-100:int(N_x/2)+100],k_shifted[int(N_x/2)-100:int(N_x/2)+100],np.imag(V_F_shifted)[int(N_x/2)-100:int(N_x/2)+100])
    plt.show()
    
    
    #H_m[,i]



