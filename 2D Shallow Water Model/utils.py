# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/gabrielmpp/shallow_water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Moved utility functions to utils.py

Extended based on a practical by Prof Pier Luigi Vidale
"""

from scipy import interpolate

def divergence(u, v, dx, dy):
    # Take the difference between u or v at index j+1 and j
    dudx = u.diff('x')/dx
    dvdy = v.diff('y')/dy
    # Fix coordinates to match eta
    dudx['x'] = dvdy.x
    dvdy['y'] = dudx.y
    return dudx+dvdy

def zonal_boundary(u):
    # Set u=0 at eastern and western boundaries
    u[:,[0,-1]] = 0
    return u

def meridional_boundary(v):
    # Set v=0 at northern and southern boundaries
    v[[0,-1],:] = 0
    return v

def coriolis(plane, u, v, component):
    '''
    Function to compute the Coriolis term (either f-plane or beta-plane)

    Here we use the interp method from the xarray library to compute the v values in the u grid (and vice-versa)

    Note that u (v) is interpolated to v (u) grid based on the surrounding 4 u (v) points in the Arakawa C-grid
    '''
    if component == 'zonal':
        coriolis_force = plane.interp(y=u.y, method='linear')*v[:, :].interp(x = u.x[1:-1], y = u.y, method = 'linear')
    elif component == 'meridional':
        coriolis_force = plane.interp(y=v.y, method='linear')*u[:, :].interp(x = v.x, y = v.y[1:-1], method = 'linear')

    return coriolis_force

def linear_depart(array, u, v, dt):
    '''
    Method for finding the departure points
    '''
    # Finding departure points
    x_dep = array.x - dt*u.interp(x=array.x, y=array.y)
    y_dep = array.y - dt*v.interp(x=array.x, y=array.y)

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain
    x_dep = x_dep.where(x_dep >= array.x.min(), array.x.min())
    x_dep = x_dep.where(x_dep <= array.x.max(), array.x.max())
    y_dep = y_dep.where(y_dep >= array.y.min(), array.y.min())
    y_dep = y_dep.where(y_dep <= array.y.max(), array.y.max())

    # Interpolating array at the departure points
    array.values = array.interp(x=x_dep, y=y_dep, method = 'linear').values

    return array

def cubic_depart(array, u, v, dt):
    '''
    Method to obtain a fully third order semi-lagrangian scheme
    following Dale Durran's book
    '''
    # Finding the midpoint of the back trajectory (we only consider half time-step)
    x_mid = array.x - 0.5*dt*u.interp(x=array.x, y=array.y)
    y_mid = array.y - 0.5*dt*v.interp(x=array.x, y=array.y)

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain
    x_mid = x_mid.where(x_mid >= array.x.min(), array.x.min())
    x_mid = x_mid.where(x_mid <= array.x.max(), array.x.max())
    y_mid = y_mid.where(y_mid >= array.y.min(), array.y.min())
    y_mid = y_mid.where(y_mid <= array.y.max(), array.y.max())

    # Computing u and v by linear interpolation in the midpoints
    u_mid = u.interp(x=x_mid.x, y=y_mid.y, method = 'linear')
    v_mid = v.interp(x=x_mid.x, y=y_mid.y, method = 'linear')

    # Determining the departure point based on the wind at the midpoint of back trajectory instead of at the initial endpoint
    x_dep = array.x - dt*u_mid
    y_dep = array.y - dt*v_mid

    # Performing a cubic interpolation using 4 closest points to departure point
    # interpolate.interp2d returns a function which can be called on departure points
    interp = interpolate.interp2d(array.x.values, array.y.values, array.values, kind='cubic')
    array.values = interp(x_dep.x.values, y_dep.y.values)

    return array

def extrapolate_wind(wind_n, wind_nm1):
    """
    This is required to get the wind at time n+1/2 to find midpoint of back trajectory, as advised by Dale Durran.
    """
    wind_ext = 3/2*wind_n - 1/2*wind_nm1
    return wind_ext

def find_value_at_midpt(array, u, v, dt):
    # Finding the midpoint of the back trajectory (we only consider half time-step)
    x_mid = array.x - 0.5*dt*u.interp(x=array.x, y=array.y)
    y_mid = array.y - 0.5*dt*v.interp(x=array.x, y=array.y)

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain
    x_mid = x_mid.where(x_mid >= array.x.min(), array.x.min())
    x_mid = x_mid.where(x_mid <= array.x.max(), array.x.max())
    y_mid = y_mid.where(y_mid >= array.y.min(), array.y.min())
    y_mid = y_mid.where(y_mid <= array.y.max(), array.y.max())

    # Performing a cubic interpolation using 4 closest points to departure point
    # interpolate.interp2d returns a function which can be called on departure points
    interp = interpolate.interp2d(array.x.values, array.y.values, array.values, kind='cubic')
    array.values = interp(x_mid.x.values, y_mid.y.values)

    return array