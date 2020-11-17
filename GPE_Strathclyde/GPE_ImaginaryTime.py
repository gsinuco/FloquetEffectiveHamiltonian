#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:41:08 2020

@author: german

Imaginary time evolution of the 1D GPE

Code adapted from the original matlab by Elmar Haller's group @ Strathclyde

"""

import numpy as np
import matplotlib.pyplot as pyplot


#clear all;
#clf;

#%*****************************
#% Settings
#%*****************************

#% Physcial Constants
au    = 1.6605389e-27;      #%
mCs   = 133*au;             #%Atomic mass of cesium; 
hbar  = 1.0545718e-34;      #%Plancks constant
pi    = 4.0*np.arctan(1.0);

#% dipole traps
wr = 2*pi*50;
wz = 2*pi*1;               #% trap frequencies

#% harmonic oscillator width
Lr  = np.sqrt(hbar/(mCs*wr));     
Lz  = np.sqrt(hbar/(mCs*wz));    

#% scattering length
a0    = 0.52917721092e-10;  #% Bohr Radius
aScat = 200*a0;             #% scattering length
g1D   = 2*hbar*hbar*aScat/(mCs*Lr*Lr); 


#%*****************************
#% Spatial grid
#%*****************************
N  = 10000;
M  = 300; 
Nz = 2*M+1;
dz = 5.0e-7; 
zV = np.linspace(-M,M,601)*dz;#(-M:1:M)*dz; 


#%*****************************
#% Momentum grid
#%*****************************
dk = pi/(M*dz); 
kV = np.linspace(-M,M,601)*dk;#(-M:1:M)*dk; 

#%*****************************
#% Potential
#%*****************************
V = 0.5*mCs*wz*wz*zV*zV/hbar; 

#%*****************************
#% Wavefunction
#%*****************************
psi_0 = np.sqrt(N/Lz)*np.power(1.0/pi,0.25)*np.exp(-zV*zV/(10*Lz*Lz)); 

#figure(1)
pyplot.plot(zV*1e6,abs(psi_0))
pyplot.plot(zV*1e6,V)
pyplot.xlabel("Position (um)")
pyplot.ylabel("Energy/$k_B$ (mK)")
pyplot.show()
#%*****************************
#% Time
#%*****************************
dt  = 1E-6;    #% time steps
Nt  = 200000; #% number of steps                


#%*****************************
#% Loop over time
#%*****************************
Nframe = 50000; #%Data saved every Nframe steps

tV   = 0; 
psi  = psi_0; 



denseMatrix = np.zeros([int(Nt/Nframe),Nz]); #%Initialization
timeVect    = np.zeros([1,Nt]);
#%%
for itime in range(Nt+1): #%Time-stepping with split-step Fourier method

    #% get current potential
    psi   = psi*np.exp(-0.5*np.complex(0.0,1.0)*dt*(V+(g1D/hbar)*abs(psi)*abs(psi)));
    psi_k = np.fft.fftshift(np.fft.fft(psi)/Nz);
    psi_k = psi_k*np.exp(-0.5*dt*np.complex(0.0,1.0)*(hbar/mCs)*kV*kV);
    psi   = np.fft.ifft(np.fft.ifftshift(psi_k))*Nz;
    psi   = psi*np.exp(-0.5*np.complex(0.0,1.0)*dt*(V+(g1D/hbar)*abs(psi)*abs(psi)));
    
   # if np.mod(itime,500)== 0:  
   #     denseMatrix[itime,::] = abs(psi)*abs(psi); 
   #     timeVect[itime] = dt*(itime-1);
    
        
    if np.mod(itime,Nframe) == 0:
    #   % plot result
    #    figure(2)
    #    yyaxis left
         pyplot.plot(zV*1E6, abs(psi))
    #     pyplot.xlabel("Position (um)")
    #     pyplot.ylabel("Energy/$k_B$ (mK)")
    #     pyplot.show()

#subplot(1,3,1); %Plot potential
#plot(x,V,'k'); 
#xlabel('x (m)'); 
#ylabel('V (J/hbar)');

#subplot(1,3,2); %Plot initial and final density
#plot(x,abs(psi_0).^2,'k',x,abs(psi).^2,'b');

#legend('\psi(x,0)','\psi(x,T)');
##xlabel('x (m)');
#ylabel('|\psi|2 (m{-1})');

#subplot(1,3,3); % Plot spacetime evolution as pcolor plot
#dt_large=dt*double(Nt/Nframe);
#pcolor(x,dt_large*(1:1:Nframe),spacetime); 
#shading interp;
#xlabel('x (m)'); 
#ylabel('t (s)');

 
