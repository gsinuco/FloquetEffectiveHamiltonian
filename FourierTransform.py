#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:01:44 2020

@author: german
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linespace(-65,65,128)
V = np.cos(2*np.pi*x)

V_F = np.fft(V)
plt.plot(np.fft.fftshift(V_F))


