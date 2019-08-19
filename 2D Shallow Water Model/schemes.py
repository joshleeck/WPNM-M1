# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/gabrielmpp/shallow_water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Moved numerical schemes to schemes.py

Extended based on a practical by Prof Pier Luigi Vidale
"""

from utils import *
import numpy as np

# The eulerian method solves the full set of linearised SWE
# The semi lagrangian methods solves the full set of non-linear SWE

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
                eta_np1[:, :] += dt*func(sigma)

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*func(w)

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
def SL(time_step, max_time_step, u_nm1, v_nm1, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, func):

    while (time_step < max_time_step):
        # ============== Odd time steps =============
        if time_step % 2 != 0:
            # ----------------- Compute wind at next half time step via extrapolation ----------
            u_ext = extrapolate_wind(u, u_nm1)
            v_ext = extrapolate_wind(v, v_nm1)
            # Estimate midpt of back trajectory (equation 7.20 in Dale Durran's book)
            #from extrapolated wind, interpolate to get wind at midpt
            u_midpt = find_value_at_midpt(u_ext[:,:], u[:, :], v[:, :], dt)
            v_midpt = find_value_at_midpt(v_ext[:,:], u[:, :], v[:, :], dt)

            # Iterate a few times to possibly get better estimates of wind at midpt
            #for iter in range(4):
            #    u_midpt = find_value_at_midpt(u_ext[:, :], u_midpt[:, :], v_midpt[:, :], dt)
            #    v_midpt = find_value_at_midpt(v_ext[:, :], u_midpt[:, :], v_midpt[:, :], dt)

            # ----------------- Computing eta values at next time step -------------------
            # Without source/sink
            eta_np1[:, :] = func(eta[:, :], u_midpt[:, :], v_midpt[:, :], dt) - \
                            dt*H*func(divergence(u, v, dx, dy), u_midpt[:, :], v_midpt[:, :], dt)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*func(sigma, u_midpt[:, :], v_midpt[:, :], dt)

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*func(w, u_midpt[:, :], v_midpt[:, :], dt)

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for u at next time step --------------
            height_div = eta[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u_midpt[:, :], v_midpt[:, :], dt) - dt*func(g*height_div, u_midpt[:, :], v_midpt[:, :], dt)

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*func(kappa*u[:, 1:-1], u_midpt[:, :], v_midpt[:, :], dt)

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*func(tau_x/(rho_0*H), u_midpt[:, :], v_midpt[:, :], dt)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*func(coriolis(plane, u, v, 'zonal'), u_midpt[:, :], v_midpt[:, :], dt)

            u_np1 = zonal_boundary(u_np1)

            # ------------ Computing values for v at next time step --------------
            height_div = eta[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u_midpt[:, :], v_midpt[:, :], dt) - dt*func(g*height_div, u_midpt[:, :], v_midpt[:, :], dt)

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= dt*func(kappa*v[1:-1, :], u_midpt[:, :], v_midpt[:, :], dt)

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*func(tau_y/(rho_0*H), u_midpt[:, :], v_midpt[:, :], dt)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*func(coriolis(plane, u_np1, v, 'meridional'), u_midpt[:, :], v_midpt[:, :], dt)

            v_np1 = meridional_boundary(v_np1)

            # -------------------------- Done with u and v -----------------------------

            u_nm1 = u.copy()
            v_nm1 = v.copy()
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
            # ----------------- Compute wind at next half time step via extrapolation ----------
            u_ext = extrapolate_wind(u, u_nm1)
            v_ext = extrapolate_wind(v, v_nm1)
            # Estimate wind at midpt of back trajectory from extrapolated wind
            u_midpt = find_value_at_midpt(u_ext[:,:], u[:, :], v[:, :], dt)
            v_midpt = find_value_at_midpt(v_ext[:,:], u[:, :], v[:, :], dt)

            # Iterate a few times to possibly get better estimates of wind at midpt
            #for iter in range(4):
            #    u_midpt = find_value_at_midpt(u_ext[:, :], u_midpt[:, :], v_midpt[:, :], dt)
            #    v_midpt = find_value_at_midpt(v_ext[:, :], u_midpt[:, :], v_midpt[:, :], dt)

            # ----------------- Computing eta values at next time step -------------------

            # Without source/sink
            eta_np1[:, :] = func(eta[:, :], u_midpt[:, :], v_midpt[:, :], dt) - \
                            dt*H*func(divergence(u, v, dx, dy), u_midpt[:, :], v_midpt[:, :], dt)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*func(sigma, u_midpt[:, :], v_midpt[:, :], dt)

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*func(w, u_midpt[:, :], v_midpt[:, :], dt)

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for v at next time step --------------
            height_div = eta[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u_midpt[:, :], v_midpt[:, :], dt) - dt*func(g*height_div, u_midpt[:, :], v_midpt[:, :], dt)

            # Add friction if enabled.
            if (use_friction is True):
                v_np1[1:-1, :] -= dt*func(kappa*v[1:-1, :], u_midpt[:, :], v_midpt[:, :], dt)

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*func(tau_y/(rho_0*H), u_midpt[:, :], v_midpt[:, :], dt)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*func(coriolis(plane, u, v, 'meridional'), u_midpt[:, :], v_midpt[:, :], dt)

            v_np1 = meridional_boundary(v_np1)

            # ------------ Computing values for u at next time step --------------
            height_div = eta[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u_midpt[:, :], v_midpt[:, :], dt) - dt*func(g*height_div, u_midpt[:, :], v_midpt[:, :], dt)

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, 1:-1] -= dt*func(kappa*u[:, 1:-1], u_midpt[:, :], v_midpt[:, :], dt)

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*func(tau_x/(rho_0*H), u_midpt[:, :], v_midpt[:, :], dt)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*func(coriolis(plane, u, v_np1, 'zonal'), u_midpt[:, :], v_midpt[:, :], dt)

            u_np1 = zonal_boundary(u_np1)

            # -------------------------- Done with u and v -----------------------------

            u_nm1 = u.copy()
            v_nm1 = v.copy()
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