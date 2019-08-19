# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/jostbr/shallow-water

Author: Joshua Lee (MSS/CCRS)
Updated: 16/08/2019 - Moved config variables to config.py

Extended based on a practical by Prof Pier Luigi Vidale
"""

import numpy as np

# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical prameters ---------------
L_x = 1e+6              # Length of domain in x-direction [m]
L_y = 1e+6              # Length of domain in y-direction [m]
g = 9.81                # Acceleration of gravity [m/s^2]
H = 100                # Depth of fluid [m]
f_0 = 1e-4              # Fixed part of coriolis parameter [1/s]
beta = 1e-11            # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0          # Density of fluid [kg/m^3)]
tau_0 = 0.2             # Amplitude of wind stress [kg/ms^2]
kappa_0 = 1e-4          # Linear drag coefficient [1/s]
use_coriolis = True     # True if you want coriolis force
use_beta = True         # True if you want variation in coriolis
use_friction = False     # True if you want linear drag
use_wind = False         # True if you want wind stress
use_source = False      # True if you want mass source into the domain
use_sink = False        # True if you want mass sink out of the domain

# --------------- Computational parameters ---------------
N_x = 200                            # Number of grid points in x-direction
N_y = 200                            # Number of grid points in y-direction
dx = L_x/N_x                        # Grid spacing in x-direction
dy = L_y/N_y                        # Grid spacing in y-direction
dt = 0.1*min(dx, dy)/np.sqrt(g*H)   # Time step (defined from the CFL condition)
time_step = 1                       # For counting time loop steps
max_time_step = 1000                # Total number of time steps in simulation
method = 'linear SL'              # Available options are "eulerian", "linear SL", and "cubic SL"

anim_interval = 50                  # How often to sample for animation