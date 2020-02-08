#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:48:16 2018

@author: layaparkavousi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 00:33:34 2018

@author: layaparkavousi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

def f(x):
    return np.exp(-x**2)

delta=0.35
N = 40000
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
mu, sigma = 0.0, 1 # mean and standard deviation

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

bin_heights, bin_borders, _ = plt.hist(x, label='histogram' , bins = 25)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[2., 0., 2.])

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 4000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit')
plt.xlabel("x")
plt.ylabel("Probability distribution")
plt.legend()
