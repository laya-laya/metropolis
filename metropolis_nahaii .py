#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 00:33:34 2018

@author: layaparkavousi
"""

import numpy as np
import matplotlib.pyplot as plt
import random
def f(x):
    return np.exp(-x**2)

delta=0.35
N = 20000
x = np.arange(N,dtype=np.float)

x[0] = 0.5
counter = 0
for i in range(0, N-1):
        
    x_next = x[i]+(delta*random.uniform(-1,1))
    if np.random.random_sample() < min(1, f(x_next)/f(x[i])):
        x[i+1] = x_next
        counter = counter + 1
    else:
        x[i+1] = x[i]
        
print("acceptance ratio is ", counter/float(N))

"""
plt.hist(x,bins = 15, color='purple')
plt.xlabel("x")
plt.ylabel("Probability distribution")
"""

T = np.zeros(int(N/170))
z = np.zeros(int(N/170))

for t in range (0,int(N/170)):
    asum = 0
    bsum = 0
    csum =0
    c = 0
    for i in range (0,N-t):
        asum = asum + (x[i]*x[i+t])
        bsum = bsum + x[i]
        csum= csum + x[i+t]
        c = c + 1
        
    bsum =bsum/c
    csum = csum/c
    asum = asum/c
    
    sigma= np.var(x)
    z[t] = (asum-(bsum*csum))/sigma
    T[t] = t

    
plt.plot(T,(z))
plt.xlabel("j")
plt.ylabel("C(j)")
plt.title("acceptance ratio = 0.9")
