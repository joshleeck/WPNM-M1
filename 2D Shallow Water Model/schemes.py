# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/gabrielmpp/shallow_water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Moved numerical schemes to schemes.py
         21/08/2019 - Reduced SL to first order, shorter run time (optional second order SL in separate setup)

Extended based on a practical by Prof Pier Luigi Vidale
"""

from utils import *
import numpy as np

# Note that the eulerian method solves the full set of linearised SWE
# Note that the semi lagrangian methods solves the full set of non-linear SWE

# Eulerian method (Forward-backward time stepping, centred space)
def eulerian(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, func):

    while (time_step < max_time_step):
        #============== Odd time steps =============
        if time_step % 2 != 0:
            # ----------------- Computing eta values at next time step -------------------
            # Without source/sink
            eta_np1[:, :] = func(eta[:, :], u[:, :], v[:, :], dt) - H*dt*divergence(u, v, dx, dy)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*w

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v, 'zonal')

            u_np1 = zonal_boundary(u_np1)

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt)  - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u, v, 'meridional')

            v_np1 = meridional_boundary(v_np1)

            # -------------------------- Done with u and v -----------------------------

            eta = eta_np1.copy()
            u = u_np1.copy()
            v = v_np1.copy()

            # Store eta and (u, v) every anim_interval time step for animations.
            if (time_step % anim_interval == 0):
                print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
                print("Step: \t{} / {}".format(time_step, max_time_step))
                print("Mass: \t{}\n".format(np.sum(eta.values)))
                u_int = u.interp(x=eta.x, y=eta.y, method='linear')
                v_int = v.interp(x=eta.x, y=eta.y, method='linear')
                eta_int = eta.interp(x=eta.x, y=eta.y, method='linear')
                u_list.append(u_int.values)
                v_list.append(v_int.values)
                eta_list.append(eta_int.values)

            time_step += 1
        # ============== Even timesteps =============
        elif time_step % 2 == 0:
            # ----------------- Computing eta values at next time step -------------------
            eta_np1[:, :] = func(eta[:, :], u[:, :], v[:, :], dt) - H*dt*divergence(u, v, dx, dy) # Without source/sink

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*w

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt)  - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u, v, 'meridional')

            v_np1 = meridional_boundary(v_np1)

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v, 'zonal')

            u_np1 = zonal_boundary(u_np1)
            # -------------------------- Done with u and v -----------------------------

            eta = eta_np1.copy()
            u = u_np1.copy()
            v = v_np1.copy()

            # Store eta and (u, v) every anim_interval time step for animations.
            if (time_step % anim_interval == 0):
                print("Time: \t{:.2f} hours".format(time_step*dt/3600))
                print("Step: \t{} / {}".format(time_step, max_time_step))
                print("Mass: \t{}\n".format(np.sum(eta.values)))
                u_int = u.interp(x=eta.x, y=eta.y, method='linear')
                v_int = v.interp(x=eta.x, y=eta.y, method='linear')
                eta_int = eta.interp(x=eta.x, y=eta.y, method='linear')
                u_list.append(u_int.values)
                v_list.append(v_int.values)
                eta_list.append(eta_int.values)

            time_step += 1


# Semi Lagrangian
def SL(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, func):

    while (time_step < max_time_step):
        # ============== Odd time steps =============
        if time_step % 2 != 0:
            # ----------------- Computing eta values at next time step -------------------
            # Without source/sink
            eta_np1[:, :] = func(eta[:, :], u[:, :], v[:, :], dt) - (H+eta.values)*dt*divergence(u, v, dx, dy)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*w

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]  # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate

            # Without friction, wind stress, rotation
            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*(H+eta.interp(x=u.x, y=eta.y, method='linear')[:, 1:-1]))

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v, 'zonal')

            u_np1 = zonal_boundary(u_np1)

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            # Without friction, wind stress, rotation
            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt) - g*dt*height_div \

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= dt*kappa*v[1:-1, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*(H+eta.interp(x=eta.x, y=v.y, method='linear')[1:-1, :]))

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u_np1, v, 'meridional')

            v_np1 = meridional_boundary(v_np1)

            # -------------------------- Done with u and v -----------------------------
            eta = eta_np1.copy()
            u = u_np1.copy()
            v = v_np1.copy()

            # Store eta and (u, v) every anim_interval time step for animations.
            if (time_step % anim_interval == 0):
                print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
                print("Step: \t{} / {}".format(time_step, max_time_step))
                print("Mass: \t{}\n".format(np.sum(eta.values)))
                u_int = u.interp(x=eta.x, y=eta.y, method='linear')
                v_int = v.interp(x=eta.x, y=eta.y, method='linear')
                eta_int = eta.interp(x=eta.x, y=eta.y, method='linear')
                u_list.append(u_int.values)
                v_list.append(v_int.values)
                eta_list.append(eta_int.values)

            time_step += 1
        # ============== Even timesteps =============
        elif time_step % 2 == 0:
            # ----------------- Computing eta values at next time step -------------------
            # Without source/sink
            eta_np1[:, :] = func(eta[:, :], u[:, :], v[:, :], dt) - (H+eta.values)*dt*divergence(u, v, dx, dy)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*w

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            # Without friction, wind stress, rotation
            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= dt*kappa*v[1:-1, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*(H+eta.interp(x=eta.x, y=v.y, method='linear')[1:-1, :]))

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u, v, 'meridional')

            v_np1 = meridional_boundary(v_np1)

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]  # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate

            # Without friction, wind stress, rotation
            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v_np1[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*(H+eta.interp(x=u.x, y=eta.y, method='linear')[:, 1:-1]))

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v_np1, 'zonal')

            u_np1 = zonal_boundary(u_np1)

            # -------------------------- Done with u and v -----------------------------

            eta = eta_np1.copy()
            u = u_np1.copy()
            v = v_np1.copy()

            # Store eta and (u, v) every anim_interval time step for animations.
            if (time_step % anim_interval == 0):
                print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
                print("Step: \t{} / {}".format(time_step, max_time_step))
                print("Mass: \t{}\n".format(np.sum(eta.values)))
                u_int = u.interp(x=eta.x, y=eta.y, method='linear')
                v_int = v.interp(x=eta.x, y=eta.y, method='linear')
                eta_int = eta.interp(x=eta.x, y=eta.y, method='linear')
                u_list.append(u_int.values)
                v_list.append(v_int.values)
                eta_list.append(eta_int.values)

            time_step += 1