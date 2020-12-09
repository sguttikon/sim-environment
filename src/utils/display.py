#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Wedge
import cv2
import numpy as np
from utils import helpers

class Render(object):
    """
    """

    def __init__(self, fig_size=(7, 7)):
        fig = plt.figure(figsize=fig_size)
        self.plt_ax = fig.add_subplot(111)
        plt.ion()
        plt.show()

        self.plots = {
            'occ_map': None,
            'robot_gt': {
                'pose': None,
                'heading': None,
                'particles': None,
            },
        }

    def process_data(self, data):
        self.occ_map = helpers.to_numpy(data['occ_map'][0])
        self.occ_map_res = helpers.to_numpy(data['occ_map_res'][0])
        self.robot_gt_pose = helpers.to_numpy(data['robot_gt_pose'][0][0])
        self.robot_gt_particles = helpers.to_numpy(data['robot_gt_particles'][0])
        self.robot_gt_labels = helpers.to_numpy(data['robot_gt_labels'][0])

    def update_figures(self, data):
        self.process_data(data)

        self.plot_map()
        #self.plot_robot(self.robot_gt_pose, 'green')
        self.plot_particles(self.robot_gt_particles, 'coral', self.robot_gt_labels)

        plt.draw()
        plt.pause(0.00000000001)

    def plot_map(self):

        occ_map = self.occ_map
        origin_x, origin_y = 0., 0.

        rows, cols = occ_map.shape
        x_max = (cols )/2 + origin_x
        x_min = (-cols )/2 + origin_x
        y_max = (rows )/2 + origin_y
        y_min = (-rows )/2 + origin_y
        extent = [x_min, x_max, y_min, y_max]

        map_plt = self.plots['occ_map']
        if map_plt is None:
            occ_map = cv2.flip(occ_map, 0)
            map_plt = self.plt_ax.imshow(occ_map, origin='upper', extent=extent)

            self.plt_ax.grid()
            self.plt_ax.plot(origin_x, origin_y, 'm+', markersize=12)
            self.plt_ax.set_xlim([x_min, x_max])
            self.plt_ax.set_ylim([y_min, y_max])

            # ticks_x = np.linspace(x_min, x_max)
            # ticks_y = np.linspace(y_min, y_max)
            # self.plt_ax.set_xticks(ticks_x)
            # self.plt_ax.set_xticklabels(' ')
            # self.plt_ax.set_yticks(ticks_y)
            # self.plt_ax.set_yticklabels(' ')
            self.plt_ax.set_xlabel('x coords')
            self.plt_ax.set_ylabel('y coords')
        else:
            pass

        self.plots['occ_map'] = map_plt

    def plot_robot(self, robot_pose, color):

        pos_x, pos_y, heading = robot_pose
        # rescale factor for position is 10/self.occ_map_res
        pos_x = pos_x * 10/self.occ_map_res
        pos_y = pos_y * 10/self.occ_map_res

        radius = 0.1 * 10/self.occ_map_res
        len = 0.1 * 10/self.occ_map_res

        xdata = [pos_x, pos_x + (radius + len) * np.cos(heading)]
        ydata = [pos_y, pos_y + (radius + len) * np.sin(heading)]

        pose_plt = self.plots['robot_gt']['pose']
        heading_plt = self.plots['robot_gt']['heading']
        if pose_plt is None:
            pose_plt = Wedge( (pos_x, pos_y), radius, 0, 360, color=color, alpha=0.75)
            self.plt_ax.add_artist(pose_plt)
            heading_plt, = self.plt_ax.plot(xdata, ydata, color=color, alpha=0.75)
        else:
            pose_plt.update({'center': [pos_x, pos_y],})
            heading_plt.update({'xdata': xdata, 'ydata': ydata,})

        self.plots['robot_gt']['pose'] = pose_plt
        self.plots['robot_gt']['heading'] = heading_plt

    def plot_particles(self, particles, color, labels=None):
        # rescale factor for position is 10/self.occ_map_res
        particles = particles * 10/self.occ_map_res

        if labels is None:
            particles_plt = self.plots['robot_gt']['particles']
            if particles_plt is None:
                particles_plt = plt.scatter(particles[:, 0], particles[:, 1], s= 0.5 * 10/self.occ_map_res, c=color, alpha=0.5)
            else:
                particles_plt.set_offsets(particles[:, 0:2])
            self.plots['robot_gt']['particles'] = particles_plt
        else:
            colors = cm.rainbow(labels)
            particles_plt = self.plots['robot_gt']['particles']
            plts = []
            idx = 0
            for pose, c in zip(particles[:, 0:2], colors):
                if particles_plt is None:
                    s_plt = plt.scatter(pose[0], pose[1], color=c)
                else:
                    s_plt = particles_plt[idx]
                    s_plt.set_offsets(pose)
                    s_plt.set_color(c)
                    idx += 1
                plts.append(s_plt)
            self.plots['robot_gt']['particles'] = plts

    def __del__(self):
        # to prevent plot from closing
        plt.ioff()
        plt.show()
