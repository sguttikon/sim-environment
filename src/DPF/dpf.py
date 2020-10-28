#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DMCL():
    """
    Differentiable Monte Carlo Localization class implementation
    """

    def __init__(self):
        """
        """

        self._state_dim = 3 # robot's pose [x, y, theta]
        self._action_dim = 2 # [linear_vel, angular_vel]
        self._num_particles = 1000
        self._init_particles = None
        self._init_particles_probs = None   # shape (num, state_dim)
        self._state_range = None

        self._min_obs_likelihood = 0.004

        self._build_modules()

    def get_device(self) -> torch.device:
        """
        return the currenly running device name
        """
        return device

    def get_num_particles(self) -> int:
        """
        return the number of particles
        """
        return self._num_particles

    def _build_modules(self):
        """
        """

        #
        self._noise_generator = nn.Sequential(
                nn.Linear(2 * self._action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self._state_dim),
                nn.ReLU(),
        ).to(device)

        #
        N, C, H, W = 1, 3, 240, 320 # refer turtlebot.yaml of rgb specs
        conv_config = np.array([
            [3, 48, 7, 3, 5],
            [48, 128, 7, 3, 5],
        ])
        for idx in range(len(conv_config)):
            W = (W - conv_config[idx][2] + 2*conv_config[idx][3]) \
                    / conv_config[idx][4] + 1
            H = (H - conv_config[idx][2] + 2*conv_config[idx][3]) \
                    / conv_config[idx][4] + 1
            C = conv_config[idx][1]
        conv_output = N*C*int(H)*int(W)

        #
        self._encoder = nn.Sequential(
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
                nn.Linear(conv_output, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 128),
                nn.ReLU(),
        ).to(device)

        #
        self._obs_like_estimator = nn.Sequential(
                nn.Linear(128 + 3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Softmax(dim=0)
        ).to(device)

    def initialize_particles(self, init_pose: np.ndarray) -> (torch.Tensor, torch.Tensor):
        """
        """
        radius = 3
        self._state_range = np.array([
            [init_pose[0] - radius, init_pose[0] + radius],
            [init_pose[1] - radius, init_pose[1] + radius],
            [-np.pi, np.pi]
        ])

        # random uniform particles between [low, high]
        # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        self._init_particles = torch.cat([
            (self._state_range[d][0] - self._state_range[d][1]) *
                torch.rand(self._num_particles, 1) + self._state_range[d][1]
                    for d in range(self._state_dim)
        ], axis = -1).to(device)

        self._init_particles_probs = (torch.ones(self._num_particles) / self._num_particles).to(device)

        return (self._init_particles, self._init_particles_probs)

    def motion_update(self, actions: np.ndarray, particles: torch.Tensor) -> torch.Tensor:
        """
        motion update based on velocity model

        :param np.ndarray actions: linear and angular velocity commands
        :param torch.Tensor particles: belief represented by particles
        :return torch.Tensor: motion updated particles
        """

        #action_input = np.tile(actions, (particles.shape[0], 1))
        #action_input = torch.from_numpy(action_input).float().to(device)

        actions = torch.from_numpy(actions).float().to(device)
        action_input = actions.repeat(particles.shape[0], 1)
        random_input = torch.normal(mean=0.0, std=1.0, size=action_input.shape).to(device)
        input = torch.cat([action_input, random_input], axis=-1)

        # estimate action noise
        delta = self._noise_generator(input)

        # add zero-mean action noise to original actions
        delta -= torch.mean(delta, 1, True)

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

    def measurement_update(self, obs: dict, particles: torch.Tensor) -> torch.Tensor:
        """

        :param collections.OrderedDict obs: observation from environment
        :param torch.Tensor particles: belief represented by particles
        :return torch.Tensor: likelihood of particles
        """

        rgb = obs['rgb'].float().to(device)
        rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2) # from NHWC to NCHW
        #depth = obs['depth'].to(device)

        encoding = self._encoder(rgb)
        encoding_input = encoding.repeat(particles.shape[0], 1)
        input = torch.cat([encoding_input, particles], axis=-1)

        obs_likelihood = self._obs_like_estimator(input)
        # obs_likelihood = obs_likelihood * (1 - self._min_obs_likelihood) + \
        #                     self._min_obs_likelihood

        return obs_likelihood.squeeze(1)

    def resample_particles(self, particles:torch.Tensor, particle_probs: torch.Tensor) -> torch.Tensor:
        """
        stochastic universal resampling according to particle weight (probs)

        :param torch.Tensor particles: motion updated particles
        :param torch.Tensor particle_probs: likelihood of particles
        :return torch.Tensor: resampled particles
        """

        low = 0.0
        step = 1 / self._num_particles
        rnd_offset = ((low - step) * torch.rand(1) + step).to(device)    # uniform random [0, step]
        cum_prob = particle_probs[0]
        i = 0

        new_particles = []
        for idx in range(self._num_particles):
            while rnd_offset > cum_prob:
                i += 1
                cum_prob += particle_probs[i]

            new_particles.append(particles[i]) # add the particle
            rnd_offset += step

        new_particles = torch.stack(new_particles, axis=0)
        return new_particles

    def particles_to_state(self, particles:torch.Tensor, particle_probs:torch.Tensor) -> torch.Tensor:
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
        return torch.cat([mean_position, mean_orientation])
