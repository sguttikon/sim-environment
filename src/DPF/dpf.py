#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from transforms3d.euler import quat2euler
import os
import cv2
from pathlib import Path
import random

np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(42)
else:
    device = torch.device('cpu')
    torch.manual_seed(42)

class DMCL():
    """
    Differentiable Monte Carlo Localization class implementation
    """

    def __init__(self, env_config_file: str):
        """
        """

        self.__state_dim = 3 # robot's pose [x, y, theta]
        self.__action_dim = 2 # [linear_vel, angular_vel]
        self.__num_particles = 5000
        self.__particles = None    # shape (num, state_dim)
        self.__particles_probs = None

        # ui display
        fig = plt.figure(figsize=(7, 7))
        self.__plt_ax = fig.add_subplot(111)
        plt.ion()
        plt.show()

        self.__plots = {
            'map': None,
            'gt_pose': None,
            'gt_heading': None,
            'est_pose': None,
            'est_heading': None,
            'particles_cloud': None,
        }
        self.__map_scale = 1.

        self.__configure_env(env_config_file)
        self.__build_modules()

        self.__writer = SummaryWriter()

        Path("saved_models").mkdir(parents=True, exist_ok=True)
        Path("best_models").mkdir(parents=True, exist_ok=True)

    def __configure_env(self, env_config_file: str):
        """
        """

        self.__config_data = parse_config(env_config_file)
        self.__map_scale = self.__config_data['trav_map_resolution']

        self.__env = NavigateRandomEnv(config_file = env_config_file,
                                mode = 'headless', # ['headless', 'gui']
                                render_to_tensor = True)

        self.__robot = self.__env.robots[0] # hard coded

    def __build_modules(self):
        """
        """

        #
        self.__noise_generator = nn.Sequential(
                nn.Linear(2 * self.__action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self.__state_dim),
                nn.ReLU(),
        ).to(device)

        self.__transition_model = nn.Sequential(
            nn.Linear(self.__state_dim + self.__action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.__state_dim),
        ).to(device)

        #
        N, C, H, W = 1, 3, 64, 64 # refer turtlebot.yaml of rgb specs
        conv_config = np.array([
            [3, 48, 5, 1, 2], # [in, out, kernel, padding, stride]
            [48, 64, 3, 1, 1],
        ])
        for idx in range(len(conv_config)):
            W = (W - conv_config[idx][2] + 2*conv_config[idx][3]) \
                    / conv_config[idx][4] + 1
            H = (H - conv_config[idx][2] + 2*conv_config[idx][3]) \
                    / conv_config[idx][4] + 1
            C = conv_config[idx][1]
        conv_output = N*C*int(H)*int(W)

        #
        self.__encoder = nn.Sequential(
                nn.Conv2d(in_channels=conv_config[0][0],
                          out_channels=conv_config[0][1],
                          kernel_size=conv_config[0][2],
                          padding=conv_config[0][3],
                          stride=conv_config[0][4]),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv2d(in_channels=conv_config[1][0],
                          out_channels=conv_config[1][1],
                          kernel_size=conv_config[1][2],
                          padding=conv_config[1][3],
                          stride=conv_config[1][4]),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Flatten(),
                nn.Linear(conv_output, 2048),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 64),
                nn.ReLU(),
        ).to(device)

        #
        self.__obs_like_estimator = nn.Sequential(
                nn.Linear(64 + self.__state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softmax(dim=0)
        ).to(device)

        self.__motion_optim = torch.optim.Adam(self.__noise_generator.parameters())
        self.__measurement_optim = torch.optim.Adam(self.__obs_like_estimator.parameters())

    def __init_particles(self, init_pose: np.ndarray) -> (torch.Tensor, torch.Tensor):
        """
        """
        radius = 5 / self.__map_scale
        state_range = np.array([
            [init_pose[0] - radius, init_pose[0] + radius],
            [init_pose[1] - radius, init_pose[1] + radius],
            [-np.pi, np.pi]
        ])

        # random uniform particles between [low, high]
        # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        self.__particles = torch.cat([
            (state_range[d][0] - state_range[d][1]) *
                torch.rand(self.__num_particles, 1) + state_range[d][1]
                    for d in range(self.__state_dim)
        ], axis = -1).to(device)

        self.__particles_probs = (torch.ones(self.__num_particles) \
                                    / self.__num_particles).to(device)

        return (self.__particles, self.__particles_probs)

    def __motion_update(self, actions: np.ndarray, particles: torch.Tensor) -> torch.Tensor:
        """
        motion update based on velocity model

        :param np.ndarray actions: linear and angular velocity commands
        :param torch.Tensor particles: belief represented by particles
        :return torch.Tensor: motion updated particles
        """

        #action_input = np.tile(actions, (particles.shape[0], 1))
        #action_input = torch.from_numpy(action_input).float().to(device)

        actions = torch.from_numpy(actions).float().to(device)
        actions = actions.unsqueeze(0)
        action_input = actions.repeat(self.__num_particles, 1)
        random_input = torch.normal(mean=0.0, std=1.0, size=action_input.shape).to(device)
        input = torch.cat([action_input, random_input], axis=-1)

        # estimate action noise
        delta = self.__noise_generator(input)

        # add zero-mean action noise to original actions
        delta = delta - torch.mean(delta, 1, True)

        use_odom_model = True
        if use_odom_model:
            delta = delta.detach()  # this is necessary for gradient flow
            noisy_actions = torch.cat([
                delta[:, 0:1] + action_input[:, 0:1],
                delta[:, 1:2] + action_input[:, 1:2],
                #delta[:, 2:3]
            ], axis=-1)

            state_input = torch.cat([particles, noisy_actions], axis=-1)
            state_delta = self.__transition_model(state_input)
            new_states = [particles[:, i:i+1] + state_delta[:, i:i+1] for i in range(3)]

            moved_particles = torch.cat(new_states, axis=-1)
            moved_particles[:, 2:3] = utils.wrap_angle(moved_particles[:, 2:3])
        else:
            # reference: probabilistic robotics: 'algorithm sample_motion_model_velocity()'
            # move the particles using noisy actions
            x = particles[:, 0:1]
            y = particles[:, 1:2]
            theta = particles[:, 2:3]

            noisy_v = delta[:, 0:1] + action_input[:, 0:1]
            noisy_w = delta[:, 1:2] + action_input[:, 1:2]
            noisy_r = delta[:, 2:3]
            radius = (noisy_v/noisy_w)
            delta_t = 1 # 1sec

            new_x = x - radius*torch.sin(theta) + radius*torch.sin(theta + noisy_w*delta_t)
            new_y = y + radius*torch.cos(theta) - radius*torch.cos(theta + noisy_w*delta_t)
            new_theta = utils.wrap_angle(theta + noisy_w*delta_t + noisy_r*delta_t)

            moved_particles = torch.cat([new_x, new_y, new_theta], axis=-1)

        return moved_particles

    def __measurement_update(self, obs: dict, particles: torch.Tensor) -> torch.Tensor:
        """

        :param collections.OrderedDict obs: observation from environment
        :param torch.Tensor particles: belief represented by particles
        :return torch.Tensor: likelihood of particles
        """

        rgb = obs['rgb'].float().to(device)
        rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
        #depth = obs['depth'].to(device)

        encoding = self.__encoder(rgb)
        encoding_input = encoding.repeat(self.__num_particles, 1)
        input = torch.cat([encoding_input, particles], axis=-1)

        obs_likelihood = self.__obs_like_estimator(input)
        # obs_likelihood = obs_likelihood * (1 - self._min_obs_likelihood) + \
        #                     self._min_obs_likelihood

        return obs_likelihood.squeeze(1)

    def __resample_particles(self, particles:torch.Tensor, particle_probs: torch.Tensor) -> torch.Tensor:
        """
        stochastic universal resampling according to particle weight (probs)

        :param torch.Tensor particles: motion updated particles
        :param torch.Tensor particle_probs: likelihood of particles
        :return torch.Tensor: resampled particles
        """

        low = 0.0
        step = 1./self.__num_particles
        rnd_offset = ((low - step) * torch.rand(1) + step).to(device)    # uniform random [0, step]
        markers = torch.linspace(rnd_offset.item(), 1.0, self.__num_particles).to(device)
        cum_probs = torch.cumsum(particle_probs, axis=0)

        i = j = 0
        new_particles = []
        for idx in range(self.__num_particles):
            while j<self.__num_particles and markers[i] > cum_probs[j]:
                j = j+1

            if j == self.__num_particles: # border case
                j = j-1

            new_particles.append(particles[j]) # add the particle
            i = i+1
        new_particles = torch.stack(new_particles, axis=0)

        return new_particles

    def __particles_to_state(self, particles:torch.Tensor, particle_probs:torch.Tensor) -> torch.Tensor:
        """
        gaussian mixture model, we treat each particle as a gaussian in a mixture with weights

        :param torch.Tensor particles: particles
        :param torch.Tensor particle_probs: likelihood of particles
        :return torch.tensor robot pose belief
        """
        mean_position = torch.sum(particle_probs.unsqueeze(1) * particles[:, :2], axis=0)
        mean_orientation = torch.atan2(
            torch.sum(particle_probs.unsqueeze(1) * torch.sin(particles[:, 2:3]), axis=0),
            torch.sum(particle_probs.unsqueeze(1) * torch.cos(particles[:, 2:3]), axis=0)
        )
        return torch.cat([mean_position, mean_orientation], axis=0)

    def __state_to_particles(self, state: torch.Tensor):
        """
        """

        mean = state
        cov = torch.eye(3).to(device) * 0.1
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

        particles = []
        for idx in range(self.__num_particles):
            particles.append(mvn.sample())

        particles = torch.stack(particles, axis=0)
        return particles

    def __update_figures(self):
        """
        """
        self.__plots['map'] = self.__plot_map(self.__plots['map'])
        self.__plots['gt_pose'], self.__plots['gt_heading'] = self.__plot_robot_gt(
                                    self.__plots['gt_pose'],
                                    self.__plots['gt_heading']
                                )
        self.__plots['est_pose'], self.__plots['est_heading'] = self.__plot_robot_est(
                                    self.__plots['est_pose'],
                                    self.__plots['est_heading']
                                )
        self.__plots['particles_cloud'] = self.__plot_particle_cloud(
                                    self.__plots['particles_cloud']
                                )

        plt.draw()
        plt.pause(0.00000000001)

    def __plot_robot_gt(self, pose_plt, heading_plt):
        """
        """

        pose = self.get_gt_pose(to_tensor=False)
        return self.__plot_robot_pose(pose, pose_plt, heading_plt, 'navy')

    def __plot_robot_est(self, pose_plt, heading_plt):
        """
        """

        pose = self.get_est_pose(to_tensor=False)
        return self.__plot_robot_pose(pose, pose_plt, heading_plt, 'maroon')

    def __plot_robot_pose(self, robot_pose, pose_plt, heading_plt, color: str):
        """
        """

        pose_x, pose_y, heading = robot_pose

        # rescale position
        pose_x = pose_x * self.__map_scale
        pose_y = pose_y * self.__map_scale

        robot_radius = 10. * self.__map_scale
        arrow_len = 10.0 * self.__map_scale

        xdata = [pose_x, pose_x + (robot_radius + arrow_len) * np.cos(heading)]
        ydata = [pose_y, pose_y + (robot_radius + arrow_len) * np.sin(heading)]

        if pose_plt == None:
            pose_plt = Wedge((pose_x, pose_y),
                             robot_radius, 0, 360,
                             color=color, alpha=0.5)
            self.__plt_ax.add_artist(pose_plt)
            heading_plt, = self.__plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
        else:
            pose_plt.update({
                        'center' : [pose_x, pose_y]
            })
            heading_plt.update({
                        'xdata' : xdata,
                        'ydata' : ydata,
            })

        return pose_plt, heading_plt

    def __plot_map(self, map_plt):
        """
        """

        model_id = self.__config_data['model_id']

        model_path = get_model_path(model_id)
        with open(os.path.join(model_path, 'floors.txt'), 'r') as f:
            floors = sorted(list(map(float, f.readlines())))

        # default considering floor env is pointing to
        floor_idx = self.__env.floor_num
        trav_map = cv2.imread(os.path.join(model_path,
                                'floor_trav_{0}.png'.format(floor_idx)))
        obs_map = cv2.imread(os.path.join(model_path,
                                'floor_{0}.png'.format(floor_idx)))

        origin_x, origin_y = 0., 0. # hard coded

        rows, cols, _ = trav_map.shape
        x_max = (cols/2 + origin_x) * self.__map_scale
        x_min = (-cols/2 + origin_x) * self.__map_scale
        y_max = (rows/2 + origin_y) * self.__map_scale
        y_min = (-rows/2 + origin_y) * self.__map_scale
        extent = [x_min, x_max, y_min, y_max]

        if map_plt == None:
            map_plt = self.__plt_ax.imshow(trav_map, cmap=plt.cm.binary, origin='upper', extent=extent)

            self.__plt_ax.plot(origin_x, origin_y, 'm+', markersize=12)
            self.__plt_ax.grid()
            self.__plt_ax.set_xlim([x_min, x_max])
            self.__plt_ax.set_ylim([y_min, y_max])

            ticks_x = np.linspace(x_min, x_max)
            ticks_y = np.linspace(y_min, y_max)
            self.__plt_ax.set_xticks(ticks_x, ' ')
            self.__plt_ax.set_yticks(ticks_y, ' ')
            self.__plt_ax.set_xlabel('x coords')
            self.__plt_ax.set_ylabel('y coords')
        else:
            pass

        return map_plt

    def __plot_particle_cloud(self, particles_plt):
        """
        """

        particles = self.__particles.cpu().detach().numpy()
        particles[:, 0:2] = particles[:, 0:2] * self.__map_scale

        if particles_plt == None:
            particles_plt = plt.scatter(particles[:, 0], particles[:, 1],
                                            s=12, c='coral', alpha=0.5)
        else:
            particles_plt.set_offsets(particles[:, 0:2])

        return particles_plt

    def __compute_motion_loss(self):
        """
        """
        gt_pose = self.get_gt_pose(to_tensor = True).repeat(self.__num_particles, 1)
        sq_dist = torch.log(utils.compute_sq_distance(self.__particles, gt_pose))
        std = 0.5
        mvn_pdf = (1/self.__num_particles) * (1/np.sqrt(2 * np.pi * std**2)) \
                        * torch.exp(-sq_dist / (2. * std**2))
        # loss
        loss = torch.mean(-torch.log(1e-16 + mvn_pdf), axis=0)

        return loss

    def __compute_measurement_loss(self, temperature=0.07, base_temperature=0.07):
        """
        """
        # reference https://github.com/wangz10/contrastive_loss/blob/master/losses.py#L104
        # and https://github.com/HobbitLong/SupContrast/blob/master/losses.py

        gt_pose = self.get_gt_pose(to_tensor = True).repeat(self.__num_particles, 1)
        features = torch.log(utils.compute_sq_distance(self.__particles, gt_pose)).unsqueeze(1)
        labels = torch.zeros([self.__num_particles, 1]).to(device)

        use_labels = True
        if use_labels:
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(self.__num_particles).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max

        # tile mask
        logits_mask = torch.ones_like(mask).to(device) - \
                            torch.eye(self.__num_particles).to(device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # loss
        loss = torch.mean(-(temperature/base_temperature) * mean_log_prob_pos, axis=0)

        return loss

    def ___plot_grad_flow(self, named_parameters):
        """
        plots the gradients flowing through different layers in the network during training.

        usage: plug this function in training class after loss.backwards as
               '___plot_grad_flow(self.model.named_parameters())' to visualize
        """
        # reference https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def __save_model(self, file_path):
        """
        """
        torch.save({
            '__noise_generator': self.__noise_generator.state_dict(),
            '__transition_model': self.__transition_model.state_dict(),
            '__encoder': self.__encoder.state_dict(),
            '__obs_like_estimator': self.__obs_like_estimator.state_dict(),
            '__motion_optim': self.__motion_optim.state_dict(),
            '__measurement_optim': self.__measurement_optim.state_dict(),
        }, file_path)

    def __load_model(self, file_path):
        """
        """
        checkpoint = torch.load(file_path)
        self.__noise_generator.load_state_dict([checkpoint['__noise_generator']])
        self.__transition_model.load_state_dict([checkpoint['__transition_model']])
        self.__encoder.load_state_dict([checkpoint['__encoder']])
        self.__obs_like_estimator.load_state_dict([checkpoint['__obs_like_estimator']])
        self.__motion_optim.load_state_dict([checkpoint['__motion_optim']])
        self.__measurement_optim.load_state_dict([checkpoint['__measurement_optim']])

    def __set_train_mode(self):
        """
        """
        self.__noise_generator.train()
        self.__transition_model.train()
        self.__encoder.train()
        self.__obs_like_estimator.train()

    def __set_eval_mode(self):
        """
        """
        self.__noise_generator.eval()
        self.__transition_model.eval()
        self.__encoder.eval()
        self.__obs_like_estimator.eval()

    #############################
    ##### PUBLIC METHODS
    #############################

    def get_gt_pose(self, to_tensor: bool = False):
        """
        """
        position = self.__robot.get_position()
        euler = quat2euler(self.__robot.get_orientation())
        gt_pose = np.array([
            position[0],
            position[1],
            utils.wrap_angle(euler[0])
        ])

        if to_tensor:
            gt_pose = torch.from_numpy(gt_pose).to(device)
        return gt_pose

    def get_est_pose(self, to_tensor: bool = False):
        """
        """
        pose = self.__particles_to_state(self.__particles, self.__particles_probs)
        if not to_tensor:
            pose = pose.cpu().detach().numpy()
        return pose

    def train(self):
        num_epochs = 100
        epoch_len = 50
        eval_epoch = 10

        curr_epoch = 0
        train_idx = 0
        eval_idx = 0

        eval_acc = np.inf
        while curr_epoch < num_epochs:
            curr_epoch += 1
            obs = self.__env.reset()

            gt_pose = self.get_gt_pose()
            self.__init_particles(gt_pose)
            self.__update_figures()

            if curr_epoch%eval_epoch == 0:
                self.__set_eval_mode()
                with torch.no_grad():
                    for curr_step in range(epoch_len):
                        eval_idx = eval_idx + 1
                        # take the action in environment
                        action = self.__env.action_space.sample() # will be changed to rl action
                        obs, reward, done, info = self.__env.step(action)

                        ##### motion update #####
                        self.__particles = self.__motion_update(action, self.__particles)

                        ##### measurement update #####
                        self.__particles_probs *= self.__measurement_update(obs, self.__particles)
                        self.__particles_probs /= torch.sum(self.__particles_probs, axis=0) # normalize probabilities

                        ##### resample particles #####
                        self.__particles = self.__resample_particles(self.__particles, self.__particles_probs)

                sqr_dist_error = np.sqrt(utils.compute_sq_distance(
                                                self.get_est_pose(),
                                                self.get_gt_pose()
                                        ) )
                self.__writer.add_scalars('eval', {
                                            'dist_error': sqr_dist_error
                                        }, eval_idx)
                if sqr_dist_error < eval_acc:
                    eval_acc = sqr_dist_error
                    file_path = 'best_models/model.pt'.format(train_idx)
                    self.__save_model(file_path)
            else:
                self.__set_train_mode()
                for curr_step in range(epoch_len):
                    train_idx = train_idx + 1
                    # take the action in environment
                    action = self.__env.action_space.sample() # will be changed to rl action
                    obs, reward, done, info = self.__env.step(action)

                    ##### motion update #####
                    self.__particles = self.__motion_update(action, self.__particles)
                    motion_loss = self.__compute_motion_loss()
                    self.__motion_optim.zero_grad()

                    ##### measurement update #####
                    self.__particles_probs *= self.__measurement_update(obs, self.__particles)
                    self.__particles_probs /= torch.sum(self.__particles_probs, axis=0) # normalize probabilities
                    measurement_loss = self.__compute_measurement_loss()
                    self.__measurement_optim.zero_grad()

                    ##### resample particles #####
                    self.__particles = self.__resample_particles(self.__particles, self.__particles_probs)

                    motion_loss.backward(retain_graph=True)
                    #self.___plot_grad_flow(self.__transition_model.named_parameters())
                    measurement_loss.backward(retain_graph=True)
                    #self.___plot_grad_flow(self.__obs_like_estimator.named_parameters())

                    self.__motion_optim.step()
                    self.__measurement_optim.step()

                    self.__update_figures()

                    # log stats
                    if train_idx%25 == 0:
                        print('Motion Loss: {0}, Measurement Loss: {1}'\
                            .format(motion_loss.item(), measurement_loss.item()))

                    self.__writer.add_scalars('train', {
                                'motion_loss': motion_loss.item()
                            }, train_idx)
                    self.__writer.add_scalars('train', {
                                'measurement_loss': measurement_loss.item()
                            }, train_idx)

                    if curr_epoch%10 == 0:
                        file_path = 'saved_models/model_train_idx{}.pt'.format(train_idx)
                        self.__save_model(file_path)

                sqr_dist_error = np.sqrt(utils.compute_sq_distance(
                                                self.get_est_pose(),
                                                self.get_gt_pose()
                                        ) )
                self.__writer.add_scalars('train', {
                            'dist_error': sqr_dist_error
                        }, train_idx)
        self.__writer.close()
        print('training done')
