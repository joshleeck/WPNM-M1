# -*- coding: utf-8 -*-
"""
Reference code structure: https://github.com/jostbr/shallow-water
Reference code: https://github.com/gabrielmpp/shallow_water

Author: Joshua Lee (MSS/CCRS)
Updated: 04/07/2019 - Adapted original source code to be compatible with code structure
                        . Amended non-linear conservation equation to fully linear set of SWE
                        . Introduced Arakawa-C grid
                        . Introduced top level switches to control various options
                        . Fixed compatibility with plotting routines
        13/08/2019 - Added new features
                        . Coriolis terms (f-plane or beta-plane or none)
                        . Wind stress term
                        . Friction term
                        . Linear and cubic interpolation for semi-lagrangian method
        15/08/2019 - Tidying up code
        21/08/2019 - Reduced SL to first order, shorter run time (optional second order SL in separate setup)
        11/10/2019 - Added plotting for energy timeseries
        11/10/2019 - Modified friction term to relaxation/damping term
        11/10/2019 - Added radiating BC option

Extended based on a practical by Prof Pier Luigi Vidale

Script that solves that solves the 2D shallow water equations using finite
differences where the momentum and continuity equations are taken to be linear.
The model supports switching on/off various terms, but in its most complete form,
the model solves the following set of equations:

    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    d(eta)/dt + H*(du/dx + dv/dy) = sigma - w

where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, a forward-backward time scheme (Matsuno, 1966)
is used.

The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)

where dx, dy is the grid spacing in the x- and y-direction respectively, g is
the acceleration of gravity and H is the resting depth of the fluid.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import plotting
import schemes as sch
from utils import *
from initial_conditions import *
from config import *

# --------------- Writing string with parameter choices -------------
param_string = "\n================================================================"
param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
param_string += "\nboundary conditions = {}".format(use_BC)
param_string += "\nmethod = {}".format(method)
param_string += "\ng = {:g}\nH = {:g}".format(g, H)
param_string += "\ndx = {:.2f} m\ndy = {:.2f} m\ndt = {:.2f} s".format(dx, dy, dt)

# Allocate arrays
eta, u, v, eta_np1, u_np1, v_np1 = allocate_arrays()
# Sampling variables.
eta_list = []; u_list = []; v_list = []                     # Lists to contain eta and u,v for animation
KE_list = []; PE_list = []                                  # Lists to contain KE and PE for plotting

# Set initial conditions for eta
eta = ic_eta(eta)

# Allocate optional arrays
kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, param_string = allocate_optional_arrays(u, v, eta, param_string)

# Print parameters to screen
print(param_string)

# ============================= Parameter stuff done ===============================

t_0 = time.process_time()  # For timing the computation loop

# ==================================================================================
# ========================= Main computation ==========================
# ==================================================================================
print('Running simulation...')

if method == 'eulerian':
    func = lambda x, u, v, t: x  # Identity function for the eulerian version
    sch.eulerian(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, KE_list, PE_list, func, use_BC)

elif method == 'linear SL':
    func = linear_depart
    sch.SL(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, KE_list, PE_list, func, use_BC)

elif method == 'cubic SL':
    func = cubic_depart
    sch.SL(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, KE_list, PE_list, func, use_BC)

else:
    raise Exception("Method must be a valid option: 'eulerian', 'linear SL', 'cubic SL'")

# ============================= Computation done ================================
print("Main computation loop done!\nExecution time: {:.2f} s".format(time.process_time() - t_0))
print("\nVisualising results...")

# ==================================================================================
# ================================= Plotting =======================================
# ==================================================================================

X, Y = np.meshgrid(eta.x.values, eta.y.values)             # Meshgrid for plotting
X = np.transpose(X)                  # To get plots right
Y = np.transpose(Y)                  # To get plots right
eta_anim = plotting.eta_animation(X, Y, eta_list, anim_interval*dt)
eta_surf_anim = plotting.eta_animation3D(X, Y, eta_list, anim_interval*dt)
quiv_anim = plotting.velocity_animation(X, Y, u_list, v_list, anim_interval*dt, N_x)
energy_ts = plotting.plot_energy(KE_list, PE_list)

# ============================ Done with visualization =============================

print("\nVisualisation done!")
plt.show()
