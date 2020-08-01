#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:22:02 2020

@author: german
"""

# cython code in cython_text.pyx
# compiled with cython_setup.py ==> python cython_setup.py build_ext --inplace ==> cython_test
# we run the compiled module cython_test

#import cython_test_module
import pyximport; pyximport.install()
import cython_test

cython_test.say_hello()
