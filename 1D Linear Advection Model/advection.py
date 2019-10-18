# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 07/04/2019 - Coded skeleton
         03/05/2019 - Added exception catching and error handling
         06/08/2019 - Added new schemes
         15/08/2019 - Moved plotting routines into plotting.py
         04/09/2019 - Added higher order interpolation for SL schemes and monotone limiter
         17/10/2019 - Added Robert-Asselin filter and Robert-Asselin-Williams filter for CTCS scheme

Extended based on a practical by Dr Hilary Weller
"""
import plotting
import schemes as sch
import initial_conditions as ic
import error_analysis as ea
import matplotlib.pyplot as plt

#=========================================================================
#                      Alter the parameters here
#=========================================================================
x_min = 0   #float or integer (starting value of x)
x_max = 1   #float or integer (ending value of x)
nx = 100    #integer (number of gridpoints)
nt = 100   #integer (number of timesteps)
u = 0.2     #float or integer (wind speed)
T = 1       #float or integer (end time)
K = 0.5e-3    #float or integer (diffusivity constant)
use_RA_filter = False # Robert-Asselin filter used only in CTCS

# Which scheme do you want to use?
# Explicit scheme options
"""
FTCS:    Forward in time, Centred in space
CTCS:    Centred in time, Centred in space [Also known as leapfrog]
FTBS:    Forward in time, Backward in space
CTBS:    Centred in time, Backward in space
"""
# Implicit scheme options
"""
BTBS:    Backward in time, Backward in space
BTCS:    Backward in time, Centred in space
CNCS:    Crank-Nicolson, Centred in space
"""
# Extra scheme options
"""
CTCS_AD: Centred in time, Centred in space with Artificial Diffusion
LW:      Lax-Wendroff
WB:      Warming and Beam 
TVD:     Total Variation Diminishing
SL1:     Semi-Lagrangian with Linear Interpolation
SL2:     Semi-Lagrangian with Quadratic Interpolation
SL3:     Semi-Lagrangian with Cubic Interpolation
SL3QM:   Semi-Lagrangian with Cubic Interpolation with Quasi-Monotone
"""
# Pick scheme here
scheme = 'FTCS'

# Which initial condition do you want to use?
# Options
"""
A:       A sine wave
B:       A step function
"""

# Pick initial condition here
init_cond = 'A'

#=========================================================================
# Derived constants
#=========================================================================
# Check parameters are valid
sch.check_params(x_min, x_max, nx, nt, u, T, K)

dx = (x_max - x_min)/nx
dt = T/nt
c = dt*u/dx  # CFL criterion is a necessary but not sufficient for stability of the numerical method
d = K*dt/dx**2 # Non-dimensional diffusion coefficent used in CTCS_AD
print('dt=', dt)
print('dx=', dx)
print('c=', c)
print('d=', d)

#=========================================================================
# Initial conditions
#=========================================================================
x = ic.create_x(x_min, dx, nx)
if init_cond == 'A':
    # Initialise sine wave
    phi = ic.initial_conditions_1(x)
elif init_cond == 'B':
    # Initialise step function
    phi = ic.initial_conditions_2(x)
else:
    raise Exception("Options for initial condition are: 'A' or 'B'")
# Print initial total mass
ea.compute_mass(phi, 'Initial')

#=========================================================================
# Analytic solution with periodic boundaries
#=========================================================================
if init_cond == 'A':
    phiAnalytical = ic.initial_conditions_1((x-u*T)%(x_max-x_min))
elif init_cond == 'B':
    phiAnalytical = ic.initial_conditions_2((x-u*T)%(x_max-x_min))

#=========================================================================
# Numerical solution with periodic boundaries
#=========================================================================
# Some standard explicit schemes
if scheme == 'FTCS':
    phiNumerical = sch.FTCS(phi, c, nt)
elif scheme == 'CTCS':
    phiNumerical = sch.CTCS(phi, c, nt, use_RA_filter)
elif scheme == 'FTBS':
    phiNumerical = sch.FTBS(phi, c, nt)
elif scheme == 'CTBS':
    phiNumerical = sch.CTBS(phi, c, nt)

# Some standard implicit schemes
elif scheme == 'BTBS':
    phiNumerical = sch.BTBS(phi, c, nt)
elif scheme == 'BTCS':
    phiNumerical = sch.BTCS(phi, c, nt)
elif scheme == 'CNCS':
    phiNumerical = sch.CNCS(phi, c, nt)

# Extra schemes (artificial diffusion; extended from CTCS)
elif scheme == 'CTCS_AD':
    phiNumerical = sch.CTCS_AD(phi, c, d, nt)

# Extra schemes (finite volume methods)
elif scheme == 'LW':
    phiNumerical = sch.LW(phi, c, nt)
elif scheme == 'WB':
    phiNumerical = sch.WB(phi, c, nt)
elif scheme == 'TVD':
    phiNumerical = sch.TVD(phi, c, nt)

# Extra schemes (semi-langrangian)
elif scheme == 'SL1':
    phiNumerical = sch.SL1(phi, u, dt, dx, nt)
elif scheme == 'SL2':
    phiNumerical = sch.SL2(phi, u, dt, dx, nt)
elif scheme == 'SL3':
    phiNumerical = sch.SL3(phi, u, dt, dx, nt)
elif scheme == 'SL3QM':
    phiNumerical = sch.SL3(phi, u, dt, dx, nt, monotone=True)

else:
    raise Exception("Option for scheme is not valid, check again. Make sure input is a string.")

#=========================================================================
# Error analysis
#=========================================================================
L2, phi_error = ea.L2ErrorNorm(phiNumerical, phiAnalytical)
# Print final total mass
ea.compute_mass(phiNumerical, 'Final')

#=========================================================================
# Plotting
#=========================================================================
plotting.plot_solution(x, phi, phiNumerical, phiAnalytical, scheme)
plotting.plot_error_norm(x, L2, phi_error, scheme)