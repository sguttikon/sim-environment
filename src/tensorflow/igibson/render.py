#!/usr/bin/env python3

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def draw_floor_map(floor_map, plt_ax, map_plt):
    """
    Render the scene floor map
    :param ndarray floor_map: environment scene floor map
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :return matplotlib.image.AxesImage: updated plot of scene floor map
    """

    height, width = floor_map.shape
    orign_x, orign_y = 0, 0

    # offset the map to display correctly w.r.t origin
    x_max = width/2 + orign_x
    x_min = -width/2 + orign_x
    y_max = height/2 + orign_y
    y_min = -height/2 + orign_y
    extent = [x_min, x_max, y_min, y_max]

    if map_plt is None:
        # draw floor map
        map_plt = plt_ax.imshow(floor_map, cmap=plt.cm.binary, origin='lower', extent=extent)
    else:
        # do nothing
        pass
    return map_plt

def draw_particles_pose(particles, weights, particles_plt, resolution):
    """
    Render the particle poses on the scene floor map
    :param ndarray particles: estimates of particle pose
    :param ndarray weights: corresponding weights of particle pose estimates
    :param matplotlib.collections.PathCollection: plot of particle position color coded according to weights
    :param int resolution: map resolution to scale (x, y)
    :return matplotlib.collections.PathCollection: updated plot of particles
    """

    pos = particles[:, 0:2] / resolution # [num, x, y]
    color = cm.rainbow(weights)

    if particles_plt is None:
        # render particles positions with color
        particles_plt = plt.scatter(pos[:, 0], pos[:, 1], s=10, c=color, alpha=.4)
    else:
        # update existing particles positions and color
        particles_plt.set_offsets(pos)
        particles_plt.set_color(color)

    return particles_plt

def draw_robot_pose(robot_pose, color, plt_ax, position_plt, heading_plt, resolution):
    """
    Render the robot pose on the scene floor map
    :param ndarray robot_pose: ndarray representing robot position (x, y) and heading (theta)
    :param str color: color used to render robot position and heading
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :param matplotlib.patches.Wedge position_plt: plot of robot position
    :param matplotlib.lines.Line2D heading_plt: plot of robot heading
    :param int resolution: map resolution to scale (x, y)
    :return tuple(matplotlib.patches.Wedge, matplotlib.lines.Line2D): updated position and heading plot of robot
    """

    x, y, heading = robot_pose

    x = x / resolution
    y = y / resolution

    heading_len  = robot_radius = 1.0
    xdata = [x, x + (robot_radius + heading_len) * np.cos(heading)]
    ydata = [y, y + (robot_radius + heading_len) * np.sin(heading)]

    if position_plt == None:
        # render robot position and heading with color
        position_plt = Wedge((x, y), robot_radius, 0, 360, color=color, alpha=0.5)
        plt_ax.add_artist(position_plt)
        heading_plt, = plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
    else:
        # update existing robot position and heading
        position_plt.update({'center': [x, y]})
        heading_plt.update({'xdata': xdata, 'ydata': ydata})

    return position_plt, heading_plt
