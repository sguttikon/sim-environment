#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def draw_text(text, bgcolor, plt_ax, text_plt):
    """
    Render the text
    :param str text: text to render
    :param str bgcolor: backgroundcolor used to render text
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :param matplotlib.text.Text text_plt: plot of text
    :return matplotlib.text.Text: updated plot of text
    """

    if text_plt is None:
        # render text with color
        text_plt = plt_ax.text(0.95, 0.95, text, backgroundcolor=bgcolor,
                        horizontalalignment='right', verticalalignment='top',
                        transform=plt_ax.transAxes, fontsize=10)
    else:
        # update existing text
        text_plt.set_text(text)

    return text_plt

def draw_floor_map(floor_map, plt_ax, map_plt):
    """
    Render the scene floor map
    :param ndarray floor_map: environment scene floor map
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :return matplotlib.image.AxesImage: updated plot of scene floor map
    """

    origin_x, origin_y = floor_map.shape[1]/2, floor_map.shape[0]/2
    if map_plt is None:
        # draw floor map
        floor_map = cv2.flip(floor_map, 0)  # flip image
        map_plt = plt_ax.imshow(floor_map, cmap='gray')
        plt.scatter(origin_x, origin_y, s=10, c='black', marker='x', alpha=1)
    else:
        # do nothing
        pass
    return map_plt

def draw_particles_pose(particles, weights, map_shape, particles_plt, scale=1):
    """
    Render the particle poses on the scene floor map
    :param ndarray particles: estimates of particle pose
    :param ndarray weights: corresponding weights of particle pose estimates
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param matplotlib.collections.PathCollection: plot of particle position color coded according to weights
    :param scale: integer rescaling value
    :return matplotlib.collections.PathCollection: updated plot of particles
    """

    part_x, part_y, part_th = tf.unstack(particles, axis=-1, num=3)   # (k, 3)
    height, width, channel = map_shape

    # rescale
    part_x = part_x / scale
    part_y = part_y / scale

    # flip image changes
    part_y = height - part_y

    color = cm.rainbow(weights)

    if particles_plt is None:
        # render particles positions with color
        particles_plt = plt.scatter(part_x, part_y, s=10, c=color, alpha=.4)
    else:
        # update existing particles positions and color
        particles_plt.set_offsets(tf.stack([part_x, part_y], axis=-1))
        particles_plt.set_color(color)

    return particles_plt

def draw_robot_pose(robot_pose, color, map_shape, plt_ax, position_plt, heading_plt, scale=1):
    """
    Render the robot pose on the scene floor map
    :param ndarray robot_pose: ndarray representing robot position (x, y) and heading (theta)
    :param str color: color used to render robot position and heading
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :param matplotlib.patches.Wedge position_plt: plot of robot position
    :param matplotlib.lines.Line2D heading_plt: plot of robot heading
    :param scale: integer rescaling value
    :return tuple(matplotlib.patches.Wedge, matplotlib.lines.Line2D): updated position and heading plot of robot
    """

    x, y, heading = robot_pose
    height, width, channel = map_shape

    # rescale
    x = x / scale
    y = y / scale

    # flip image changes
    y = height - y
    heading = -heading

    heading_len  = robot_radius = 10.0
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
