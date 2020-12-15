#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

class Render(object):

    def __init__(self, fig_shape=(7, 7)):
        self.fig = plt.figure(figsize=fig_shape)
        self.plt_ax = self.fig.add_subplot(111)
        plt.ion()
        plt.show()

        self.plots = {
            'map': None,
            'gt_robot': {
                'pose': None,
                'heading': None,
            },
            'est_robot': {
                'particles': None,
            },
        }

    def update_figures(self):
        plt.draw()
        plt.pause(0.00000000001)

    def plot_map(self, map):

        rows, cols = map.shape
        extent = [-cols/2, cols/2, -rows/2, rows/2]

        map_plt = self.plots['map']
        if map_plt is None:
            map = cv2.flip(map, 0)
            map_plt = self.plt_ax.imshow(map, origin='upper', extent=extent)

            self.plt_ax.grid()
            self.plt_ax.set_xlabel('x coords')
            self.plt_ax.set_ylabel('x coords')
        else:
            pass

        self.plots['map'] = map_plt

    def plot_robot(self, robot_pose, clr):
        x, y, theta = robot_pose

        #hardcoded
        x = x * 100
        y = y * 100
        radius = 10
        length = 10

        xdata = [x, x + (radius + length) * np.cos(theta)]
        ydata = [y, y + (radius + length) * np.sin(theta)]

        pose_plt = self.plots['gt_robot']['pose']
        heading_plt = self.plots['gt_robot']['heading']
        if pose_plt is None:
            pose_plt = Wedge((x, y), radius, 0, 360, color=clr, alpha=.75)
            self.plt_ax.add_artist(pose_plt)
            heading_plt, = self.plt_ax.plot(xdata, ydata, color=clr, alpha=.75)
        else:
            pose_plt.update({
                'center': [x, y],
            })
            heading_plt.update({
                'xdata': xdata,
                'ydata': ydata,
            })

        self.plots['gt_robot']['pose'] = pose_plt
        self.plots['gt_robot']['heading'] = heading_plt

    def plot_particles(self, particles, clr):

        #hardcoded
        positions = particles[:, 0:2] * 100
        particles_plt = self.plots['est_robot']['particles']
        if particles_plt is None:
            particles_plt = plt.scatter(positions[:, 0], positions[:, 1], s=10, c=clr, alpha=.5)
        else:
            particles_plt.set_offsets(positions[:, 0:2])

        self.plots['est_robot']['particles'] = particles_plt

    def __del__(self):
        # to prevent plot from automatic closing
        plt.ioff()
        plt.show()
