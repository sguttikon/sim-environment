#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import utils.constants as constants
import networks.networks as nets
import utils.helpers as helpers
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
from pytorch_metric_learning import losses
import os
import cv2
import time

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)

class DMCL():
    """
    """

    def __init__(self, config_filename, render=False, agent='RANDOM'):
        super(DMCL, self).__init__()

        self.motion_net = nets.MotionNetwork().to(constants.DEVICE)
        self.vision_net = nets.VisionNetwork(constants.WIDTH, constants.HEIGHT).to(constants.DEVICE)
        self.likelihood_net = nets.LikelihoodNetwork().to(constants.DEVICE)
        self.particles_net = nets.ParticlesNetwork().to(constants.DEVICE)
        self.action_net = nets.ActionNetwork().to(constants.DEVICE)

        self.agent_type = agent
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

            mode = 'headless'
        else:
            mode = 'headless'

        params = list(self.vision_net.parameters()) \
                + list(self.likelihood_net.parameters())
                 # TODO add odom net parameters for optimization
        self.optimizer = torch.optim.Adam(params, lr=2e-4)
        self.triplet_loss_fn = losses.TripletMarginLoss()
        self.env = NavigateRandomEnv(config_file = config_filename,
                    mode = mode,  # ['headless', 'gui']
        )
        self.env.seed(constants.RANDOM_SEED)
        self.config_data = parse_config(config_filename)
        self.robot = self.env.robots[0]
        self.map_res = self.config_data['trav_map_resolution']

        #HACK using pretrained motion model
        checkpoint = torch.load('motion_model_complex.pt')
        self.motion_net.load_state_dict(checkpoint['motion_net'])
        self.motion_net.eval()

    def __del__(self):
        # to prevent plot from closing
        plt.ioff()
        plt.show()

    def train_mode(self):
        #self.motion_net.train()
        self.vision_net.train()
        self.likelihood_net.train()
        self.particles_net.train()
        self.action_net.train()

    def eval_mode(self):
        #self.motion_net.eval()
        self.vision_net.eval()
        self.likelihood_net.eval()
        self.particles_net.eval()
        self.action_net.eval()

    def init_particles(self, is_uniform=False):
        """
        """
        self.curr_obs = self.env.reset()

        # get environment map
        model_id = self.config_data['model_id']
        model_path = get_model_path(model_id)
        with open(os.path.join(model_path, 'floors.txt'), 'r') as f:
            floors = sorted(list(map(float, f.readlines())))

        floor_idx = self.env.floor_num
        trav_map = cv2.imread(os.path.join(model_path, 'floor_trav_{0}.png'.format(floor_idx)))
        self.o_map_size = trav_map.shape[0]
        self.env_map = cv2.resize(trav_map, (constants.WIDTH, constants.HEIGHT))

        gt_pose = helpers.get_gt_pose(self.robot)
        if is_uniform:
            # initialize particles with uniform dist
            bounds = np.array([
                [gt_pose[0] - 2, gt_pose[0] + 2],
                [gt_pose[1] - 2, gt_pose[1] + 2],
                [gt_pose[2] - np.pi/6, gt_pose[2] + np.pi/6],
            ])

            rnd_particles = []
            while len(rnd_particles) < constants.NUM_PARTICLES:
                _, self.initial_pos = self.env.scene.get_random_point_floor(self.env.floor_num, self.env.random_height)
                self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                rnd_pose = [self.initial_pos[0], self.initial_pos[1], self.initial_orn[2]]

                if bounds[0][0] <= rnd_pose[0] <= bounds[0][1] and \
                   bounds[1][0] <= rnd_pose[1] <= bounds[1][1] and \
                   bounds[2][0] <= rnd_pose[2] <= bounds[2][1] :
                   rnd_particles.append(rnd_pose) #HACK bound the particles postion
                else:
                    continue
            rnd_particles = np.array(rnd_particles)

        else:
            # initialize particles with gaussian dist
            rnd_particles = gt_pose + np.random.normal(0., 0.2, size=(constants.NUM_PARTICLES, 3))

        rnd_probs = np.ones(constants.NUM_PARTICLES) / constants.NUM_PARTICLES

        self.particles = rnd_particles
        self.update_figures()

        return gt_pose, rnd_particles

    def get_entropy(self, diff_particles, particles_probs):
        # reference https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
        cov = torch.zeros((4, 4)).to(constants.DEVICE)
        cov[0, 0] = 0.5 * 0.5
        cov[1, 1] = 0.5 * 0.5
        cov[2, 2] = 0.5 * 0.5
        cov[3, 3] = 0.5 * 0.5
        cov = cov.unsqueeze(0).repeat(diff_particles.shape[0], 1, 1)

        cov_particles = torch.sum(cov * particles_probs[:, None, None], axis=0) + \
                torch.sum(torch.var(diff_particles, dim=0, keepdim=True) * particles_probs[:, None], axis=0)
        # reference https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        return 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * helpers.to_numpy(cov_particles)))

    def step(self, old_particles, std = 0.75):
        """
        """
        self.train_mode()

        # --------- Agent Network --------- #
        if self.agent_type == 'RANDOM':
            acts = self.env.action_space.sample()
            acts = 0 if acts == 1 else acts #HACK avoid back movement
            vel_cmd = np.array(self.robot.action_list[acts])
        elif self.agent_type == 'TRAIN':
            #acts = helpers.to_numpy(self.action_net(encoded_imgs))[0]
            pass

        # take action in environment
        start_time = time.time()
        obs, reward, done, info = self.env.step(acts)
        end_time = time.time()
        delta_t = np.array([end_time-start_time])
        self.curr_obs = obs

        # --------- Odometry Network --------- #
        vel_cmds = np.repeat(np.expand_dims(vel_cmd, axis=0), old_particles.shape[0], axis=0)
        delta_ts = np.repeat(delta_t, old_particles.shape[0], axis=0)
        moved_particles = self.motion_net(old_particles, vel_cmds, delta_ts, learn_noise=True, simple_model=False)

        # --------- Vision Network --------------- #
        rgb = helpers.to_tensor(self.curr_obs['rgb'])
        env_map = helpers.to_tensor(self.env_map)
        imgs = torch.cat([rgb, env_map], axis=-1).unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
        encoded_imgs = self.vision_net(imgs)

        # --------- Observation Likelihood ------- #
        gt_pose = helpers.transform_poses(helpers.to_tensor(helpers.get_gt_pose(self.robot)))
        trans_particles = helpers.transform_poses(moved_particles)
        encoded_imgs = encoded_imgs.repeat(trans_particles.shape[0], 1)

        input_features = torch.cat([encoded_imgs, trans_particles], axis=-1)
        embeddings, obs_likelihoods = self.likelihood_net(input_features)
        obs_likelihoods = F.softmax(obs_likelihoods, dim=0)
        labels = helpers.get_triplet_labels(gt_pose, trans_particles)

        arg_gt_poses = gt_pose + torch.normal(0., 0.1, size=trans_particles.shape).to(constants.DEVICE)
        input_features = torch.cat([encoded_imgs, arg_gt_poses], axis=-1)
        augmented, _ = self.likelihood_net(input_features)
        arg_labels = helpers.get_triplet_labels(gt_pose, arg_gt_poses)

        embeddings = torch.cat([embeddings, augmented], dim=0)
        labels = torch.cat([labels, arg_labels], dim=0)

        # --------- Losses ----------- #

        triplet_loss = self.triplet_loss_fn(embeddings, labels)
        mse_loss = helpers.get_mse_loss(gt_pose, trans_particles)

        total_loss = triplet_loss + mse_loss

        # --------- Backward Pass --------- #
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # --------- Particle Network -------- #
        resampled_particles = self.particles_net(moved_particles, obs_likelihoods)

        # stop gradient flow here
        resampled_particles = helpers.to_numpy(resampled_particles)
        gt_pose = helpers.get_gt_pose(self.robot)
        self.particles = resampled_particles
        self.update_figures()

        return gt_pose, resampled_particles, { 'triplet_loss': triplet_loss, 'mse_loss': mse_loss, }

    def predict(self, old_particles):
        """
        """
        self.eval_mode()

        with torch.no_grad():

            # --------- Agent Network --------- #
            if self.agent_type == 'RANDOM':
                acts = self.env.action_space.sample()
                acts = 0 if acts == 1 else acts #HACK avoid back movement
                vel_cmd = np.array(self.robot.action_list[acts])
            elif self.agent_type == 'TRAIN':
                #acts = helpers.to_numpy(self.action_net(encoded_imgs))[0]
                pass

            # take action in environment
            start_time = time.time()
            obs, reward, done, info = self.env.step(acts)
            end_time = time.time()
            delta_t = np.array([end_time-start_time])
            self.curr_obs = obs

            # --------- Odometry Network --------- #
            vel_cmds = np.repeat(np.expand_dims(vel_cmd, axis=0), old_particles.shape[0], axis=0)
            delta_ts = np.repeat(delta_t, old_particles.shape[0], axis=0)
            moved_particles = self.motion_net(old_particles, vel_cmds, delta_ts, learn_noise=True, simple_model=False)

            # --------- Vision Network --------------- #
            rgb = helpers.to_tensor(self.curr_obs['rgb'])
            env_map = helpers.to_tensor(self.env_map)
            imgs = torch.cat([rgb, env_map], axis=-1).unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
            encoded_imgs = self.vision_net(imgs)

            # --------- Observation Likelihood ------- #
            gt_pose = helpers.transform_poses(helpers.to_tensor(helpers.get_gt_pose(self.robot)))
            trans_particles = helpers.transform_poses(moved_particles)
            encoded_imgs = encoded_imgs.repeat(trans_particles.shape[0], 1)

            input_features = torch.cat([encoded_imgs, trans_particles], axis=-1)
            embeddings, obs_likelihoods = self.likelihood_net(input_features)
            obs_likelihoods = F.softmax(obs_likelihoods, dim=0)
            labels = helpers.get_triplet_labels(gt_pose, trans_particles)

            arg_gt_poses = gt_pose + torch.normal(0., 0.1, size=trans_particles.shape).to(constants.DEVICE)
            input_features = torch.cat([encoded_imgs, arg_gt_poses], axis=-1)
            augmented, _ = self.likelihood_net(input_features)
            arg_labels = helpers.get_triplet_labels(gt_pose, arg_gt_poses)

            embeddings = torch.cat([embeddings, augmented], dim=0)
            labels = torch.cat([labels, arg_labels], dim=0)

            # --------- Losses ----------- #

            triplet_loss = self.triplet_loss_fn(embeddings, labels)
            mse_loss = helpers.get_mse_loss(gt_pose, trans_particles)

            # --------- Particle Network -------- #
            resampled_particles = self.particles_net(moved_particles, obs_likelihoods)

            # stop gradient flow here
            resampled_particles = helpers.to_numpy(resampled_particles)
            gt_pose = helpers.get_gt_pose(self.robot)
            self.particles = resampled_particles
            self.update_figures()

        return gt_pose, resampled_particles, { 'triplet_loss': triplet_loss, 'mse_loss': mse_loss, }

    def update_figures(self):
        if self.render:
            self.plots['map'] = self.plot_map(self.plots['map'])
            self.plots['robot_gt']['pose'], self.plots['robot_gt']['heading'] = \
                self.plot_robot_gt(self.plots['robot_gt']['pose'],
                                   self.plots['robot_gt']['heading'], 'navy')
            self.plots['robot_est']['pose'], self.plots['robot_est']['heading'] = \
                self.plot_robot_est(self.plots['robot_est']['pose'],
                                    self.plots['robot_est']['heading'], 'maroon')
            self.plots['robot_est']['particles'] = \
                self.plot_particles(self.plots['robot_est']['particles'], 'coral')

            plt.draw()
            plt.pause(0.00000000001)

    def plot_map(self, map_plt):

        trav_map = self.env_map
        origin_x, origin_y = 0.*self.map_res, 0*self.map_res

        rows, cols, _ = trav_map.shape
        x_max = (cols * self.map_res)/2 + origin_x
        x_min = (-cols * self.map_res)/2 + origin_x
        y_max = (rows * self.map_res/2) + origin_y
        y_min = (-rows * self.map_res/2) + origin_y
        extent = [x_min, x_max, y_min, y_max]

        if map_plt is None:
            trav_map = cv2.flip(trav_map, 0)
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

    def plot_robot_est(self, pose_plt, heading_plt, color):
        est_pose = np.mean(self.particles, axis=0)
        est_pose[2] = helpers.wrap_angle(est_pose[2], use_numpy=True)
        return self.plot_robot(est_pose, pose_plt, heading_plt, color)

    def plot_robot(self, robot_pose, pose_plt, heading_plt, color):
        pos_x, pos_y, heading = robot_pose
        res = self.map_res * (self.o_map_size/constants.WIDTH) # rescale again
        pos_x = pos_x/res
        pos_y = pos_y/res

        radius = .75*res
        len = .75*res

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
        res = self.map_res * (self.o_map_size/constants.WIDTH) # rescale again
        particles = self.particles/res

        if particles_plt is None:
            particles_plt = plt.scatter(particles[:, 0], particles[:, 1], s=12, c=color, alpha=0.5)
        else:
            particles_plt.set_offsets(particles[:, 0:2])
        return particles_plt

    def save(self, file_name):
        torch.save({
            'motion_net': self.motion_net.state_dict(),
            'vision_net': self.vision_net.state_dict(),
            'likelihood_net': self.likelihood_net.state_dict(),
            'action_net': self.action_net.state_dict(),
            #'particles_net': self.particles_net.state_dict(),
        }, file_name)
        #print('=> created checkpoint')

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        #self.motion_net.load_state_dict(checkpoint['motion_net'])
        self.vision_net.load_state_dict(checkpoint['vision_net'])
        self.likelihood_net.load_state_dict(checkpoint['likelihood_net'])
        if self.agent_type == 'TRAIN':
            self.action_net.load_state_dict(checkpoint['action_net'])
        #self.particles_net.load_state_dict(checkpoint['particles_net'])
        print('=> loaded checkpoint')
