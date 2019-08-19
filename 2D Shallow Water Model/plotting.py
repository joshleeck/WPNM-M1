# -*- coding: utf-8 -*-
"""
Reference code: https://github.com/jostbr/shallow-water

Author: Joshua Lee (MSS/CCRS)
Updated: 15/08/2019 - Tidying up code

Extended based on a practical by Prof Pier Luigi Vidale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def eta_animation(X, Y, eta_list, frame_interval):
    """
    Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also needs to specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list.
    """
    fig, ax = plt.subplots(1, 1)
    plt.xlabel("x [km]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [km]", fontname = "serif", fontsize = 12)
    pmesh = plt.pcolormesh(X/1000., Y/1000., eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r)
    plt.colorbar(pmesh, orientation = "vertical")

    # Update function for quiver animation.
    def update_eta(num):
        ax.set_title("Surface elevation $\eta$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh.set_array(eta_list[num][:-1, :-1].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)

    # Need to return anim object to see the animation
    return anim

def velocity_animation(X, Y, u_list, v_list, frame_interval):
    """
    Function that takes in the domain x, y (2D meshgrids) and a lists of 2D arrays
    u_list, v_list and creates an quiver animation of the velocity field (u, v). To get
    updating title one also needs to specify time step dt between each frame in the simulation,
    the number of time steps between each eta in eta_list.
    """
    fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
    plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
    plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
    q_int = 4
    Q = ax.quiver(Y[::q_int, ::q_int]/1000.0, X[::q_int, ::q_int]/1000.0, u_list[0][::q_int,::q_int], v_list[0][::q_int,::q_int],
        scale=0.02, scale_units='inches')
    #qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s", labelpos = "E", coordinates = "figure")

    # Update function for quiver animation.
    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19)
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = animation.FuncAnimation(fig, update_quiver,
        frames = len(u_list), interval = 10, blit = False)
    fig.tight_layout()

    # Need to return anim object to see the animation
    return anim

def eta_animation3D(X, Y, eta_list, frame_interval):
    """
    Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. A 3D surface plot of eta will be created.
    To get updating title one also needs to specify time step dt between each frame in
    the simulation, the number of time steps between each eta in eta_list.
    """
    fig = plt.figure(figsize = (8, 8), facecolor = "white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap = plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X/1000, Y/1000, eta_list[num], cmap = plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19, y=1.04)
        ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
        ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
        ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
        ax.set_xlim(X.min()/1000, X.max()/1000)
        ax.set_ylim(Y.min()/1000, Y.max()/1000)
        ax.set_zlim(-0.05, 0.05)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf,
        frames = len(eta_list), interval = 10, blit = False)

    # Need to return anim object to see the animation
    return anim