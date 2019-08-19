# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Plotting routines moved into plotting.py

Extended based on a practical by Dr Hilary Weller
"""
import matplotlib.pyplot as plt

font = {'size': 14}
plt.rc('font', **font)

def plot_solution(x, phi, phiNumerical, phiAnalytical, scheme):
    plt.figure(figsize=(10,10))
    plt.plot(x, phi, label = 'Initial Conditions', color='b')
    plt.plot(x, phiNumerical, label='Numerical Solution', color='r',linestyle='-',linewidth=2.0)
    plt.plot(x, phiAnalytical, label='Analytical Solution', linestyle='-',color='g')
    plt.legend(loc='upper right')
    plt.title(scheme)
    plt.axhline(0, linestyle=':', color='black')
    plt.xlabel('$x$')
    plt.ylim(-0.2,1.5)
    plt.ylabel(r'$\phi$')
    plt.tight_layout()

def plot_error_norm(x, L2, phi_error, scheme):
    print ('L2 Error Norm= ', L2)
    plt.figure(figsize=(10,10))
    plt.plot(x, phi_error)
    plt.axhline(0, linestyle=':', color='black')
    plt.title(scheme)
    plt.xlabel('$x$')
    plt.ylabel('L2 Error Norm')
    plt.show()