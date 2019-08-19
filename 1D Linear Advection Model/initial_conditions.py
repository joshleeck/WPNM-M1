# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 03/05/2019 - Coded skeleton

Extended based on a practical by Dr Hilary Weller
"""

import numpy as np

def create_x(x_min, dx, nx):
    '''
    Return the spatial values from x_min in steps of dx for nx spatial steps
    in an array
    '''
    x = np.zeros(nx, dtype = 'float')
    for i in range(nx):
        x[i] = x_min + i*dx

    return x
    
def initial_conditions_1(x):
    '''
    Return the first set of initial conditions, a sine wave
    '''
    phi = np.zeros(len(x), dtype = 'float')
    
    for i in range(len(phi)):
        if(x[i] < 0.5):
            phi[i] = 0.5*(1-np.cos(4*np.pi*x[i]))
        else:
            phi[i] = 0
            
    return phi

def initial_conditions_2(x):
    '''
    Return the second set of initial conditions, a step function
    '''
    
    phi = np.zeros(len(x), dtype = 'float')
    
    for i in range(len(phi)):
        if(x[i] < 0.5):
            phi[i] = 1
        else:
            phi[i] = 0
            
    return phi