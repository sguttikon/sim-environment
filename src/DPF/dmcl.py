#!/usr/bin/env python3

import numpy as np
import torch
import utils.constants as constants
import networks.networks as nets
import utils.helpers as helpers
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
import os
import cv2

np.random.seed(42)
random.seed(42)
if constants.IS_CUDA:
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

class DMCL():
    """
    """

    def __init__(self, config_filename, render=False):
        super(DMCL, self).__init__()

        self.odom_net = nets.OdomNetwork().to(constants.DEVICE)
        self.vision_net = nets.VisionNetwork().to(constants.DEVICE)
        self.likelihood_net = nets.LikelihoodNetwork().to(constants.DEVICE)
        self.particles_net = nets.ParticlesNetwork().to(constants.DEVICE)

        params = list(self.vision_net.parameters()) + \
                 list(self.likelihood_net.parameters())
                 # TODO add odom net parameters for optimization
        self.optimizer = torch.optim.Adam(params, lr=2e-4)
        self.env = NavigateRandomEnv(config_file = config_filename,
                    mode = 'headless',  # ['headless', 'gui']
        )
        self.config_data = parse_config(config_filename)
        self.robot = self.env.robots[0]

        self.render = render
        if self.render:
            fig = plt.figure(figsize=(7, 7))
            self.plt_ax = fig.add_subplot(111)
            plt.ion()
            plt.show()

            self.plots = {
                'map': None,
                'robot_gt': {
                    'pose': None,
                    'heading': None,
                },
                'robot_est':{
                    'pose': None,
                    'heading': None,
                    'particles': None,
                },
            }

    def __del__(self):
        # to prevent plot from closing
        plt.ioff()
        plt.show()

    def train_mode(self):
        self.odom_net.train()
        self.vision_net.train()
        self.likelihood_net.train()
        self.particles_net.train()

    def eval_mode(self):
        self.odom_net.eval()
        self.vision_net.eval()
        self.likelihood_net.eval()
        self.particles_net.eval()

    def to_tensor(self, array):
        return torch.from_numpy(array.copy()).float().to(constants.DEVICE)

    def to_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def init_particles(self, init_pose):
        """
        """

        limits = 100
        bounds = np.array([
            [init_pose[0] - limits, init_pose[0] + limits],
            [init_pose[1] - limits, init_pose[1] + limits],
            [-np.pi/6, np.pi/6],
        ])

        rnd_particles = np.array([
            np.random.uniform(bounds[d][0], bounds[d][1], constants.NUM_PARTICLES)
                for d in range(constants.STATE_DIMS)
        ]).T

        rnd_probs = np.ones(constants.NUM_PARTICLES) / constants.NUM_PARTICLES

        self.particles = self.to_tensor(rnd_particles)
        self.particles_probs = self.to_tensor(rnd_probs)
        self.update_figures()
        return self.particles, self.particles_probs

    def transform_particles(self, particles):
        return torch.cat([
            particles[:, 0:2], torch.cos(particles[:, 2:3]), torch.sin(particles[:, 2:3])
        ], axis=-1)

    def transform_pose(self, pose):
        return torch.cat([
            pose[0:2], torch.cos(pose[2:3]), torch.sin(pose[2:3])
        ], axis=-1)

    def step(self, particles, std = 0.75):
        """
        """
        self.train_mode()

        # --------- RL Agent Network --------- #
        # TODO: get action from trained agent
        acts = self.env.action_space.sample() * 1.5
        obs, reward, done, info = self.env.step(acts)
        imgs = obs['rgb']

        # --------- Odometry Network --------- #
        #acts  = self.to_tensor(acts)
        particles = self.odom_net(particles, acts)

        # --------- Vision Network --------------- #
        imgs = self.to_tensor(imgs).unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
        encoded_imgs = self.vision_net(imgs)

        # --------- Observation Likelihood ------- #
        trans_particles = self.transform_particles(particles)
        input_features = torch.cat([trans_particles, \
                        encoded_imgs.repeat(particles.shape[0], 1)], axis=-1)
        obs_likelihoods = self.likelihood_net(input_features).squeeze(1)
        #particles_probs = particles_probs * obs_likelihoods
        #particles_probs = torch.div(particles_probs, torch.sum(particles_probs))
        particles_probs = torch.div(obs_likelihoods, torch.sum(obs_likelihoods))

        # --------- Loss ----------- #

        # to optimize
        gt_pose = self.transform_pose(self.to_tensor(helpers.get_gt_pose(self.robot)))
        sqrt_dist = helpers.eucld_dist(gt_pose, trans_particles)

        gaussian_pdf = particles_probs * (1/np.sqrt(2 * np.pi * std**2)) * \
                        torch.exp(-.5 * (sqrt_dist/std)**2 )

        loss = torch.mean(-torch.log(1e-8 + gaussian_pdf))

        # to monitor: calculate mix of gaussians
        mean_particles = torch.sum(trans_particles*particles_probs.unsqueeze(1), axis=0)
        mse = helpers.eucld_dist(gt_pose, mean_particles)

        total_loss = loss

        # --------- Backward Pass --------- #
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        # --------- Particle Network -------- #
        particles = self.particles_net(particles, particles_probs)

        particles = particles.detach() # stop gradient flow here

        return particles, total_loss, mse

    def predict(self, particles):
        """
        """
        self.eval_mode()

        with torch.no_grad():

            # --------- RL Agent Network --------- #
            # TODO: get action from trained agent
            acts = self.env.action_space.sample() * 1.5
            obs, reward, done, info = self.env.step(acts)
            imgs = obs['rgb']

            # --------- Odometry Network --------- #
            #acts  = self.to_tensor(acts)
            particles = self.odom_net(particles, acts)

            # --------- Vision Network --------------- #
            imgs = self.to_tensor(imgs).unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
            encoded_imgs = self.vision_net(imgs)

            # --------- Observation Likelihood ------- #
            trans_particles = self.transform_particles(particles)
            input_features = torch.cat([trans_particles, \
                            encoded_imgs.repeat(particles.shape[0], 1)], axis=-1)
            obs_likelihoods = self.likelihood_net(input_features).squeeze(1)
            particles_probs = torch.div(obs_likelihoods, torch.sum(obs_likelihoods))

            # to monitor: calculate mix of gaussians
            gt_pose = self.transform_pose(self.to_tensor(helpers.get_gt_pose(self.robot)))
            mean_particles = torch.sum(trans_particles*particles_probs.unsqueeze(1), axis=0)
            mse = helpers.eucld_dist(gt_pose, mean_particles)

            # --------- Particle Network -------- #
            particles = self.particles_net(particles, particles_probs)

        return particles, mse

    def update_figures(self):
        if self.render:
            self.plots['map'] = self.plot_map(self.plots['map'])
            self.plots['robot_gt']['pose'], self.plots['robot_gt']['heading'] = \
                self.plot_robot_gt(self.plots['robot_gt']['pose'],
                                   self.plots['robot_gt']['heading'], 'navy')

            self.plots['robot_est']['particles'] = \
                self.plot_particles(self.plots['robot_est']['particles'], 'coral')

            plt.draw()
            plt.pause(0.00000000001)

    def plot_map(self, map_plt):
        model_id = self.config_data['model_id']
        model_path = get_model_path(model_id)
        with open(os.path.join(model_path, 'floors.txt'), 'r') as f:
            floors = sorted(list(map(float, f.readlines())))

        floor_idx = self.env.floor_num
        trav_map = cv2.imread(os.path.join(model_path, 'floor_trav_{0}.png'.format(floor_idx)))

        origin_x, origin_y = 0., 0.

        rows, cols, _ = trav_map.shape
        x_max = (cols)/2 + origin_x
        x_min = (-cols)/2 + origin_x
        y_max = (rows/2) + origin_y
        y_min = (-rows/2) + origin_y
        extent = [x_min, x_max, y_min, y_max]

        if map_plt is None:
            map_plt = self.plt_ax.imshow(trav_map, cmap=plt.cm.binary, origin='upper', extent=extent)

            self.plt_ax.grid()
            self.plt_ax.plot(origin_x, origin_y, 'm+', markersize=12)
            self.plt_ax.set_xlim([x_min, x_max])
            self.plt_ax.set_ylim([y_min, y_max])

            ticks_x = np.linspace(x_min, x_max)
            ticks_y = np.linspace(y_min, y_max)
            self.plt_ax.set_xticks(ticks_x, ' ')
            self.plt_ax.set_yticks(ticks_y, ' ')
            self.plt_ax.set_xlabel('x coords')
            self.plt_ax.set_ylabel('y coords')
        else:
            pass
        return map_plt

    def plot_robot_gt(self, pose_plt, heading_plt, color):
        gt_pose = helpers.get_gt_pose(self.robot)
        return self.plot_robot(gt_pose, pose_plt, heading_plt, color)

    def plot_robot(self, robot_pose, pose_plt, heading_plt, color):
        pos_x, pos_y, heading = robot_pose

        radius = 10.
        len = 10.

        xdata = [pos_x, pos_x + (radius + len) * np.cos(heading)]
        ydata = [pos_y, pos_y + (radius + len) * np.sin(heading)]

        if pose_plt is None:
            pose_plt = Wedge( (pos_x, pos_y), radius, 0, 360, color=color, alpha=0.75)
            self.plt_ax.add_artist(pose_plt)
            heading_plt, = self.plt_ax.plot(xdata, ydata, color=color, alpha=0.75)
        else:
            pose_plt.update({'center': [pos_x, pos_y],})
            heading_plt.update({'xdata': xdata, 'ydata': ydata,})
        return pose_plt, heading_plt

    def plot_particles(self, particles_plt, color):
        particles = self.to_numpy(self.particles)

        if particles_plt is None:
            particles_plt = plt.scatter(particles[:, 0], particles[:, 1], s=12, c=color, alpha=0.5)
        else:
            particles_plt.set_offsets(particles[:, 0:2])
        return particles_plt

    def save(self, file_name):
        torch.save({
            #'odom_net': self.odom_net.state_dict(),
            'vision_net': self.vision_net.state_dict(),
            'likelihood_net': self.likelihood_net.state_dict(),
            #'particles_net': self.particles_net.state_dict(),
        }, file_name)
        #print('=> created checkpoint')

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        #self.odom_net.load_state_dict(checkpoint['odom_net'])
        self.vision_net.load_state_dict(checkpoint['vision_net'])
        self.likelihood_net.load_state_dict(checkpoint['likelihood_net'])
        #self.particles_net.load_state_dict(checkpoint['particles_net'])
        print('=> loaded checkpoint')
