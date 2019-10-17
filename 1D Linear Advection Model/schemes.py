# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 07/04/2019 - Coded explicit schemes
         03/05/2019 - Added implicit schemes
         04/05/2019 - Added finite volume schemes
         06/08/2019 - Added semi-lagrangian scheme
         04/09/2019 - Added higher order interpolation for SL schemes and monotone limiter

Extended based on a practical by Dr Hilary Weller
"""

import numpy as np
import scipy.linalg as la
import math

def FTCS(phi, c, nt):
    '''
    Performs FTCS scheme
    '''
    nx = len(phi)

    # New time-step array for phi
    phiNew = np.zeros(len(phi), dtype='float')

    # FTCS for all time steps
    for it in range(nt):
        for j in range(0, nx):
            phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])
        phi = phiNew.copy()

    return phi

def CTCS(phi, c, nt):
    '''
    Performs FTCS scheme for first time step, CTCS scheme for the all other time steps
    '''
    nx = len(phi)

    # New time-step array for phiNew
    phiNew = np.zeros(len(phi), dtype='float')

    # FTCS for first time-step
    for j in range(nx):
        phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])

    phiOld = phi.copy()
    phi = phiNew.copy()

    # CTCS for all other time-steps
    for it in range(1, int(nt)):
        for j in range(nx):
            phiNew[j] = phiOld[j] - c*(phi[(j+1)%nx] - phi[(j-1)%nx])

        phiOld = phi.copy()
        phi = phiNew.copy()

    return phi

def FTBS(phi, c, nt):
    '''
    Performs FTBS scheme
    '''
    nx = len(phi)

    # New time-step array for phiNew
    phiNew = np.zeros(len(phi), dtype='float')

    # FTBS for all time steps
    for it in range(int(nt)):
        for j in range(0, nx):
            phiNew[j] = phi[j] - c*(phi[j] - phi[(j-1)%nx])
        phi = phiNew.copy()

    return phi

def CTBS(phi, c, nt):
    '''
    Performs FTCS scheme for first time step, CTBS scheme for the all other time steps
    '''
    nx = len(phi)

    # New time-step array for phiNew
    phiNew = np.zeros(len(phi), dtype='float')

    # FTCS for first time-step
    for j in range(nx):
        phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])

    phiOld = phi.copy()
    phi = phiNew.copy()

    # CTBS for all other time-steps
    for it in range(1, int(nt)):
        for j in range(nx):
            phiNew[j] = phiOld[j] - 2*c*(phi[j] - phi[(j-1)%nx])

        phiOld = phi.copy()
        phi = phiNew.copy()

    return phi

def BTBS(phi, c, nt):
    '''
    Performs BTBS scheme using matrix techniques
    '''
    nx = len(phi)

    # Array representing BTBS
    M = np.zeros([nx, nx])

    # Periodic boundary conditions
    M[0, 0] = 1+c
    M[0, -1] = -c
    M[-1, -1] = 1+c
    M[-1, -2] = -c
    for i in range(1, nx-1):
        M[i, i-1] = -c
        M[i, i] = 1+c

    # BTBS for all time steps
    for it in range(int(nt)):
        # Equivalent to solving system of linear equations Ax = b for x (ie. [A phi^n+1 = phi^n] for phi^n+1)
        phi=la.solve(M, phi)

    return phi

def BTCS(phi, c, nt):
    '''
    Performs BTCS scheme using matrix techniques
    '''
    nx = len(phi)

    # Array representing BTCS
    M = np.zeros([nx, nx])

    # Periodic boundary conditions
    M[0, 0] = 1
    M[0, 1] = c/2
    M[0, -1] = -c/2
    M[-1, -1] = 1
    M[-1, 0] = c/2
    M[-1, -2] = -c/2
    for i in range(1, nx-1):
        M[i, i-1] = -c/2
        M[i, i] = 1
        M[i, i+1] = c/2

    # BTCS for all time steps
    for it in range(int(nt)):
        # Equivalent to solving system of linear equations Ax = b for x (ie. [A phi^n+1 = phi^n] for phi^n+1)
        phi=la.solve(M, phi)

    return phi

def CNCS(phi, c, nt):
    '''
    Performs CNCS scheme using matrix techniques
    '''
    nx = len(phi)

    # Array representing BTCS
    M = np.zeros([nx, nx])
    M2 = np.zeros([nx, nx])

    # Periodic boundary conditions
    M[0, 0] = 1; M2[0, 0] = 1
    M[0, 1] = c/4; M2[0, 1] = -c/4
    M[0, -1] = -c/4; M2[0, -1] = c/4
    M[-1, -1] = 1; M2[-1, -1] = 1
    M[-1, 0] = c/4; M2[-1, 0] = -c/4
    M[-1, -2] = -c/4; M2[-1, -2] = c/4
    for i in range(1, nx-1):
        M[i, i-1] = -c/4; M2[i, i-1] = c/4
        M[i, i] = 1; M2[i, i] = 1
        M[i, i+1] = c/4; M2[i, i+1] = -c/4

    # CNCS for all time steps
    for it in range(int(nt)):
        # Equivalent to solving system of linear equations Ax = Bb for x (ie. [A phi^n+1 = B phi^n] for phi^n+1)
        phi=la.solve(M, np.matmul(M2,phi))

    return phi

def CTCS_AD(phi, c, d, nt):
    '''
    Performs FTCS scheme for first time step, CTCS scheme for the all other time steps

    Artificial diffusion is enabled with diffusion coefficient, d
    '''
    nx = len(phi)

    # New time-step array for phi
    phiNew = np.zeros_like(phi)

    # FTCS for advection and artificial diffusion for first time-step
    for j in range(nx):
        phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx]) \
                    + d*(phi[(j+1)%nx] - 2*phi[j%nx] + phi[(j-1)%nx])

    phiOld = phi.copy()
    phi = phiNew.copy()

    # CTCS for advection and artificial diffusion for all other time-steps
    for it in range(1, int(nt)):
        for j in range(nx):
            phiNew[j] = phiOld[j] - c*(phi[(j+1)%nx] - phi[(j-1)%nx]) \
                        + 2*d*(phiOld[(j+1)%nx] - 2*phiOld[j%nx] + phiOld[(j-1)%nx])

        phiOld = phi.copy()
        phi = phiNew.copy()

    return phi

def LW(phi, c, nt):
    '''
    Performs the Lax-Wendroff scheme
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    # LW for all time steps
    for it in range(int(nt)):
        for j in range(nx):
            # Different formulations but mathematically equivalent
            #phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx]) \
            #            + 0.5*c*c*(phi[(j+1)%nx]-2*phi[j]+phi[(j-1)%nx])
            phiNew[j] = phi[j] - c*(0.5*(1+c)*phi[j] + 0.5*(1-c)*phi[(j+1)%nx] \
                        - 0.5*(1+c)*phi[(j-1)%nx] - 0.5*(1-c)*phi[j])

        phi = phiNew.copy()

    return phi

def WB(phi, c, nt):
    '''
    Performs the Warming and Beam scheme
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    # WB for all time steps
    for it in range(nt):
        for j in range(nx):
            phiNew[j] = phi[j] - c*(0.5*(3-c)*phi[j] - 0.5*(1-c)*phi[(j-1)%nx] \
                        - 0.5*(3-c)*phi[(j-1)%nx] + 0.5*(1-c)*phi[(j-2)%nx])

        phi = phiNew.copy()

    return phi

def TVD(phi, c, nt):
    '''
    Performs TVD scheme
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    # TVD for all time steps
    for it in range(nt):
        for j in range(nx):
            #LW and WB combination
            phiH_plus = 0.5*(1+c)*phi[j] + 0.5*(1-c)*phi[(j+1)%nx]
            phiL_plus = 0.5*(3-c)*phi[j] - 0.5*(1-c)*phi[(j-1)%nx]
            phiH_minus = 0.5*(1+c)*phi[(j-1)%nx] + 0.5*(1-c)*phi[j]
            phiL_minus = 0.5*(3-c)*phi[(j-1)%nx] - 0.5*(1-c)*phi[(j-2)%nx]
            phiNew[j] = phi[j] - c*(VLLimFunc(phi,j)*phiH_plus + (1-VLLimFunc(phi,j))*phiL_plus   \
                        - VLLimFunc(phi,(j-1)%nx)*phiH_minus - (1-VLLimFunc(phi,(j-1)%nx))*phiL_minus)

        phi = phiNew.copy()

    return phi

def SL1(phi, u, dt, dx, nt):
    '''
    Performs SL scheme, with linear interpolation
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    for it in range(nt):
        for j in range(nx):
            # Starting from any point, compute corresponding departure index using back trajectory (constant u)
            dep_index=(j-u*dt/float(dx))%nx
            # Compute integer grid indexes which bound departure index (they are (j-p)%nx and (j-p-1)%nx)
            p=math.floor(u*dt/float(dx))
            # Compute weightage for bounds of departure index based on how far away bounds are (linear interpolation)
            alpha=(j-p-dep_index)%nx
            # Compute new value of phi at index based on a linear combination (interpolation)
            # of values from index bounding departure index
            phiNew[j]=(1-alpha)*phi[int(j-p)%nx] + alpha*phi[int(j-p-1)%nx]

        phi = phiNew.copy()

    return phi

def SL2(phi, u, dt, dx, nt):
    '''
    Performs SL scheme, with quadratic interpolation
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    for it in range(nt):
        for j in range(nx):
            # Starting from any point, compute corresponding departure index using back trajectory (constant u)
            dep_index=(j-u*dt/float(dx))%nx
            # Compute integer nearest grid indexes which bound departure index
            p=math.floor(u*dt/float(dx))
            # Compute weightage for bounds of departure index based on how far away bounds are (quadratic interpolation)
            alpha=(j-p-dep_index)%nx
            # Compute new value of phi at index based on a quadratic interpolation
            # of values from index bounding departure index
            phiNew[j]=alpha/2*(1+alpha)*phi[int(j-p-1)%nx] + (1-alpha**2)*phi[int(j-p)%nx] \
            - alpha/2*(1-alpha)*phi[int(j-p+1)%nx]

        phi = phiNew.copy()

    return phi

def SL3(phi, u, dt, dx, nt, monotone=False):
    '''
    Performs SL scheme, with cubic interpolation
    '''
    nx = len(phi)

    phiNew = np.zeros(len(phi), dtype='float')

    for it in range(nt):
        for j in range(nx):
            # Starting from any point, compute corresponding departure index using back trajectory (constant u)
            dep_index=(j-u*dt/float(dx))%nx
            # Compute nearest integer grid indexes which bound departure index
            p=math.floor(u*dt/float(dx))
            # Compute weightage for bounds of departure index based on how far away bounds are (cubic interpolation)
            alpha=(j-p-dep_index)%nx
            # Compute new value of phi at index based on a cubic interpolation
            # of values from index bounding departure index
            phiNew[j]=-(alpha*(1-alpha**2))/6*phi[int(j-p-2)%nx] + alpha*(1+alpha)*(2-alpha)/2*phi[int(j-p-1)%nx] \
            + (1-alpha**2)*(2-alpha)/2*phi[int(j-p)%nx] - alpha*(1-alpha)*(2-alpha)/6*phi[int(j-p+1)%nx]

            # Impose local monotonicity for cubic interpolation, as done at ECMWF
            if monotone==True:
                phimax=max(phi[int(j-p-2)%nx], phi[int(j-p-1)%nx], phi[int(j-p)%nx], phi[int(j-p+1)%nx])
                phimin=min(phi[int(j-p-2)%nx], phi[int(j-p-1)%nx], phi[int(j-p)%nx], phi[int(j-p+1)%nx])
                phiNew[j]=max(phimin, min(phimax, phiNew[j]))

        phi = phiNew.copy()

    return phi

#=========================================================================
#utility functions
#=========================================================================
def VLLimFunc(phi, j):
    '''
    Van Leer Limiter Function used in TVD scheme, weights the high order flux and low order flux
    '''
    nx = len(phi)

    num = phi[j] - phi[(j-1)%nx]
    denom = phi[(j+1)%nx] - phi[j]

    # Catch special case when gradient is too small (close to 0)
    if np.abs(denom) < 1e-10:
        r = 0
    else:
        r = num/float(denom)

    return ((r + np.abs(r))/(1 + np.abs(r)))


def check_params(x_min, x_max, nx, nt, u, T, K):
    '''
    Checking of input parameters and returns specific error.
    '''
    if type(x_min) not in [float, int]:
        raise Exception("x_min must be a float or integer.")
    if type(x_max) not in [float, int]:
        raise Exception("x_max must be a float or integer.")
    if type(nx) not in [int]:
        raise Exception("nx must be an integer.")
    if type(nt) not in [int]:
        raise Exception("nt must be an integer.")
    if type(u) not in [float, int]:
        raise Exception("u must be a float or integer.")
    if type(T) not in [float, int]:
        raise Exception("T must be a float or integer.")
    if type(K) not in [float, int]:
        raise Exception("K must be a float or integer.")