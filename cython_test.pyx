#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:35:54 2020

@author: german
"""
def say_hello(name):
    print("Hello %s!" % name)

#import time
#import cython
#import n
#from threading import Thread 

#def busy_sleep(n):
#    while n>0:
#        n-=1

#def busy_sleep_nogil(n):
#    with nogil:
#        busy_sleep(n)
    
#cdef inline void busy_sleep_(int n) nogil:
#    while n>0:
#        n-=1
    
#N= 99999999
#start = time.time()
#busy_sleep(N)
#busy_sleep(N)
#end = time.time()
#print("sequential: ",end - start)

#start = time.time()
#t1 = Thread(target=busy_sleep,args=(N,))
#t2 = Thread(target=busy_sleep,args=(N,))
#t1.start()
#t2.start()
#t1.join()
#t2.join()
#end = time.time()
#print("Threaded: ",end - start)
