# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/gabrielmpp/shallow_water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Moved numerical schemes to schemes.py
         21/08/2019 - Reduced SL to first order, shorter run time (optional second order SL in separate setup)
         11/10/2019 - Added computation of KE and PE
         11/10/2019 - Modified friction term to relaxation/damping term
         11/10/2019 - Added radiating BC option

Extended based on a practical by Prof Pier Luigi Vidale
"""

from utils import *
import numpy as np

# Note that the eulerian method solves the full set of linearised SWE
# Note that the semi lagrangian methods solves the full set of non-linear SWE

# Eulerian method (Forward-backward time stepping, centred space)
def eulerian(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, KE_list, PE_list, func, use_BC):

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

            # Add friction if enabled.
            if (use_friction is True):
                eta_np1[:, :] -= dt*kappa_eta*eta[:, :]

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]
                u_np1[:, :] -= dt*kappa_u*u[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v, 'zonal')

            if use_BC == 'reflecting':
                u_np1 = zonal_boundary(u_np1)
            elif use_BC == 'radiating':
                u_np1 = zonal_radiating(u_np1, u, g, H, dx, dt)

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt)  - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]
                v_np1[:, :] -= dt*kappa_v*v[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u_np1, v, 'meridional')

            if use_BC == 'reflecting':
                v_np1 = meridional_boundary(v_np1)
            elif use_BC == 'radiating':
                v_np1 = meridional_radiating(v_np1, v, g, H, dy, dt)

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

            if (use_friction is True):
                eta_np1[:, :] -= dt*kappa_eta*eta[:, :]


            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt)  - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]
                v_np1[:, :] -= dt*kappa_v*v[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u, v, 'meridional')

            if use_BC == 'reflecting':
                v_np1 = meridional_boundary(v_np1)
            elif use_BC == 'radiating':
                v_np1 = meridional_radiating(v_np1, v, g, H, dy, dt)


            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]

            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]
                u_np1[:, :] -= dt*kappa_u*u[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v_np1, 'zonal')

            if use_BC == 'reflecting':
                u_np1 = zonal_boundary(u_np1)
            elif use_BC == 'radiating':
                u_np1 = zonal_radiating(u_np1, u, g, H, dx, dt)

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

        # Store kinetic energy and potential energy at each timestep
        u_temp = u.interp(x=eta.x)
        v_temp = v.interp(y=eta.y)
        KE_list.append(np.sum(dx*dy*0.5*rho_0*H*(u_temp**2 + v_temp**2)))
        PE_list.append(np.sum(dx*dy*0.5*rho_0*g*(eta**2)))


# Semi Lagrangian
def SL(time_step, max_time_step, eta_np1, u_np1, v_np1, eta, u, v, H, g, rho_0,
                 kappa_u, kappa_v, kappa_eta, tau_x, tau_y, plane, sigma, w, dt, dx, dy,
                 use_friction, use_wind, use_coriolis, use_source, use_sink,
                 anim_interval, u_list, v_list, eta_list, KE_list, PE_list, func, use_BC):

    while (time_step < max_time_step):
        # ============== Odd time steps =============
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

            if (use_friction is True):
                eta_np1[:, :] -= dt*kappa_eta*eta[:, :]

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]  # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate

            # Without friction, wind stress, rotation
            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]
                u_np1[:, :] -= dt*kappa_u*u[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v, 'zonal')

            if use_BC == 'reflecting':
                u_np1 = zonal_boundary(u_np1)
            elif use_BC == 'radiating':
                u_np1 = zonal_radiating(u_np1, u, g, H, dx, dt)

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            # Without friction, wind stress, rotation
            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt) - g*dt*height_div \

            # Add friction if enabled.
            if (use_friction is True):
                #v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]
                v_np1[:, :] -= dt*kappa_v*v[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u_np1, v, 'meridional')

            if use_BC == 'reflecting':
                v_np1 = meridional_boundary(v_np1)
            elif use_BC == 'radiating':
                v_np1 = meridional_radiating(v_np1, v, g, H, dy, dt)

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
            eta_np1[:, :] = func(eta[:, :], u[:, :], v[:, :], dt) - H*dt*divergence(u, v, dx, dy)

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :] -= dt*w

            if (use_friction is True):
                eta_np1[:, :] -= dt*kappa_eta*eta[:, :]

            # ----------------------------- Done with eta --------------------------------

            # ------------ Computing values for v at next time step --------------
            height_div = eta_np1[:, :].diff('y')/dy
            height_div['y'] = v.y.values[1:-1]

            # Without friction, wind stress, rotation
            v_np1[1:-1, :] = func(v[1:-1, :], u[:, :], v[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #v_np1[1:-1, :] -= kappa*dt*v[1:-1, :]
                v_np1[:, :] -= dt*kappa_v*v[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                v_np1[1:-1, :] += dt*tau_y/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                v_np1[1:-1, :] -= dt*coriolis(plane, u, v, 'meridional')

            if use_BC == 'reflecting':
                v_np1 = meridional_boundary(v_np1)
            elif use_BC == 'radiating':
                v_np1 = meridional_radiating(v_np1, v, g, H, dy, dt)

            # ------------ Computing values for u at next time step --------------
            height_div = eta_np1[:, :].diff('x')/dx
            height_div['x'] = u.x.values[1:-1]  # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate

            # Without friction, wind stress, rotation
            u_np1[:, 1:-1] = func(u[:, 1:-1], u[:, :], v_np1[:, :], dt) - g*dt*height_div

            # Add friction if enabled.
            if (use_friction is True):
                #u_np1[:, 1:-1] -= dt*kappa*u[:, 1:-1]
                u_np1[:, :] -= dt*kappa_u*u[:, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, 1:-1] += dt*tau_x/(rho_0*H)

            # Use coriolis term if enabled.
            if (use_coriolis is True):
                u_np1[:, 1:-1] += dt*coriolis(plane, u, v_np1, 'zonal')

            if use_BC == 'reflecting':
                u_np1 = zonal_boundary(u_np1)
            elif use_BC == 'radiating':
                u_np1 = zonal_radiating(u_np1, u, g, H, dx, dt)

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

        # Store kinetic energy and potential energy at each timestep
        u_temp = u.interp(x=eta.x)
        v_temp = v.interp(y=eta.y)
        KE_list.append(np.sum(dx*dy*0.5*rho_0*H*(u_temp**2 + v_temp**2)))
        PE_list.append(np.sum(dx*dy*0.5*rho_0*g*(eta**2)))