# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/jostbr/shallow-water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Moved array allocation and initial conditions to initial_conditions.py


Extended based on a practical by Prof Pier Luigi Vidale
"""

import xarray as xr
import numpy as np
from config import *

# ==================================================================================
# ==================== Allocating arrays and initial conditions ====================
# ==================================================================================
def allocate_arrays():
    # Initializing coordinates with one extra zonal grid point in u and one extra meridional grid point in v
    u_coords = {'x': np.arange(-0.5*dx, L_x+0.5*dx, dx), 'y': np.arange(0, L_y, dy)}
    v_coords = {'x': np.arange(0, L_x, dx), 'y': np.arange(-0.5*dy, L_y+0.5*dy, dy)}
    eta_coords = {'x': np.arange(0, L_x, dx), 'y': np.arange(0, L_y, dy)}

    # Initialising 2D xarray dataarrays for storing a single time step (used during integration) and next time step
    eta = xr.DataArray(np.zeros([N_y, N_x]), dims=('y', 'x'), coords={'y': eta_coords['y'], 'x': eta_coords['x']})
    u = xr.DataArray(np.zeros([N_y, N_x+1]), dims=('y', 'x'), coords={'y': u_coords['y'], 'x': u_coords['x']})
    v = xr.DataArray(np.zeros([N_y+1, N_x]), dims=('y', 'x'), coords={'y': v_coords['y'], 'x': v_coords['x']})
    eta_np1 = xr.DataArray(np.zeros([N_y, N_x]), dims=('y', 'x'), coords={'y': eta_coords['y'], 'x': eta_coords['x']})
    u_np1 = xr.DataArray(np.zeros([N_y, N_x+1]), dims=('y', 'x'), coords={'y': u_coords['y'], 'x': u_coords['x']})
    v_np1 = xr.DataArray(np.zeros([N_y+1, N_x]), dims=('y', 'x'), coords={'y': v_coords['y'], 'x': v_coords['x']})

    return eta, u, v, eta_np1, u_np1, v_np1

def ic_eta(eta):
    # Initial condition for eta.
    #eta[:, :] = 0.0
    eta[:, :] = 0.1*np.exp(-((eta.x-L_x/2)**2/(2*(0.05e+6)**2) + (eta.y-L_y/2)**2/(2*(0.05e+6)**2))) #initial positive height perturbation
    #eta[:, :] = -0.1*np.exp(-((eta.x-L_x/2)**2/(2*(0.05e+6)**2) + (eta.y-L_y/2)**2/(2*(0.05e+6)**2))) #initial negative height perturbation
    return eta

def allocate_optional_arrays(u, v, eta, param_string):
    # Define friction array if friction is enabled.
    if (use_friction is True):
        n=10 #number of damping gridpoints
        kappa_u = np.zeros_like(u)
        kappa_v = np.zeros_like(v)
        kappa_eta = np.zeros_like(eta)
        #kappa is constant at domain edge
        # for i in range(n):
        #    kappa_u[:, [i,-i-1]] = kappa_0
        #    kappa_v[[i,-i-1], :] = kappa_0
        #    kappa_eta[[i,-i-1], :] = kappa_0
        #    kappa_eta[:, [i,-i-1]] = kappa_0

        #kappa is a function of distance (linear) from domain edge
        for i in range(n):
            kappa_u[:, [i,-i-1]] = kappa_0*(1-i/n)
            kappa_v[[i,-i-1], :] = kappa_0*(1-i/n)
            kappa_eta[[i,-i-1], :] = kappa_0*(1-i/n)
            kappa_eta[:, [i,-i-1]] = kappa_0*(1-i/n)
        param_string += "\nkappa_0 = {:g}".format(kappa_0)
    else:
        kappa_u = 0
        kappa_v = 0
        kappa_eta = 0

    # Define wind stress arrays if wind is enabled.
    if (use_wind is True):
        tau_x = -tau_0*np.cos(np.pi*u.y/L_y)
        tau_y = 0
        param_string += "\ntau_0 = {:g}\nrho_0 = {:g} kg/m^3".format(tau_0, rho_0)
    else:
        tau_x = 0
        tau_y = 0

    # Define coriolis array if coriolis is enabled.
    if (use_beta == True) and (use_coriolis == False):
        raise Exception('use_coriolis must be set to True before use_beta set to True')

    if (use_coriolis is True):
        L_R = np.sqrt(g*H)/f_0                         # Rossby deformation radius
        if (use_beta is True):
            plane = (f_0 + beta*eta.y)                 # Varying coriolis parameter (beta-plane)
        else:
            plane = f_0 + 0*eta.y                      # Constant coriolis parameter (f-plane)

        param_string += "\nf_0 = {:g}".format(f_0)
        param_string += "\nRossby radius: {:.1f} km".format(L_R/1000)
        param_string += "\n================================================================\n"
    else:
        plane = 0*eta.y

    # Define source array if source is enabled.
    if (use_source is True):
        sigma = 0.0001*np.exp(-((eta.x-L_x/2)**2/(2*(1e+5)**2) + (eta.y-L_y/2)**2/(2*(1e+5)**2)))*eta
    else:
        sigma = np.zeros((N_y, N_x))*eta

    # Define sink array if sink is enabled.
    if (use_sink is True):
        w = np.ones((N_y, N_x))*sigma
    else:
        w = np.zeros((N_y, N_x))*eta

    return kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, param_string