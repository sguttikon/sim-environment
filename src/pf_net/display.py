#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.cm as cm
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
                'pose': None,
                'heading': None,
                'particles': None,
            },
        }

    def update_figures(self, data):

        gt_pose = data['gt_pose']
        est_pose = data['est_pose']
        particle_states = data['particle_states']
        particle_weights = data['particle_weights']

        self.plot_gt_robot(gt_pose, 'green')
        self.plot_est_robot(est_pose, 'blue')
        self.plot_particles(particle_states, particle_weights, 'coral')

        self.plt_ax.legend([self.plots['gt_robot']['pose'], self.plots['est_robot']['pose']], \
            ["gt_robot", "est_robot"])

        plt.draw()
        plt.pause(0.00000000001)

    def plot_map(self, floor_map, floor_map_res):

        rows, cols = floor_map.shape
        self.map_res = floor_map_res
        self.map_rows = rows

        extent = [-cols/2, cols/2, -rows/2, rows/2]

        map_plt = self.plots['map']
        if map_plt is None:
            # floor_map = cv2.flip(floor_map, 0)
            map_plt = self.plt_ax.imshow(floor_map, origin='upper', extent=extent)

            self.plt_ax.grid()
            self.plt_ax.set_xlabel('x coords')
            self.plt_ax.set_ylabel('x coords')
        else:
            pass

        self.plots['map'] = map_plt

    def plot_gt_robot(self, gt_pose, clr):
        pose_plt = self.plots['gt_robot']['pose']
        heading_plt = self.plots['gt_robot']['heading']

        pose_plt, heading_plt = self.plot_robot(gt_pose, pose_plt, heading_plt, clr)

        self.plots['gt_robot']['pose'] = pose_plt
        self.plots['gt_robot']['heading'] = heading_plt

    def plot_est_robot(self, est_pose, clr):
        pose_plt = self.plots['est_robot']['pose']
        heading_plt = self.plots['est_robot']['heading']

        pose_plt, heading_plt = self.plot_robot(est_pose, pose_plt, heading_plt, clr)

        self.plots['est_robot']['pose'] = pose_plt
        self.plots['est_robot']['heading'] = heading_plt

    def plot_robot(self, robot_pose, pose_plt, heading_plt, clr):
        x, y, theta = robot_pose

        x = x * self.map_rows * self.map_res
        y = y * self.map_rows * self.map_res
        radius = 0.1 * self.map_rows * self.map_res
        length = 0.1 * self.map_rows * self.map_res

        xdata = [x, x + (radius + length) * np.cos(theta)]
        ydata = [y, y + (radius + length) * np.sin(theta)]


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

        return pose_plt, heading_plt

    def plot_particles(self, particles, particle_weights, clr):

        if particle_weights is None:
            colors = clr
        else:
            colors = cm.rainbow(particle_weights)

        positions = particles[:, 0:2] * self.map_rows * self.map_res
        particles_plt = self.plots['est_robot']['particles']
        if particles_plt is None:
            particles_plt = plt.scatter(positions[:, 0], positions[:, 1], s=10, c=colors, alpha=.5)
        else:
            particles_plt.set_offsets(positions[:, 0:2])
            particles_plt.set_color(colors)
        self.plots['est_robot']['particles'] = particles_plt

    def __del__(self):
        # to prevent plot from automatic closing
        plt.ioff()
        plt.show()
