#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:16:24 2020

@author: german
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name='cython test_module', 
      ext_modules=cythonize("cython_test.pyx"))