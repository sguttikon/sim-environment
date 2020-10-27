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
        self._action_dim = 2
        self._num_particles = 10
        self._init_particles = None
        self._init_particles_probs = None   # shape (num, state_dim)
        self._state_range = None

        self._build_modules()

    def _build_modules(self):
        """
        """

        self._noise_generator = nn.Sequential(
                nn.Linear(2 * self._action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self._state_dim),
                nn.ReLU(),
        ).to(device)

        self._encoder = None

    def initialize_particles(self, init_pose: np.ndarray):
        """
        """
        self._state_range = np.array([
            [init_pose[0] - 1, init_pose[0] + 1],
            [init_pose[1] - 1, init_pose[1] + 1],
            [-np.pi, np.pi]
        ])

        # random uniform particles between [low, high]
        # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        self._init_particles = torch.cat([
            (self._state_range[d][0] - self._state_range[d][1]) *
                torch.rand(self._num_particles, 1) + self._state_range[d][1]
                    for d in range(self._state_dim)
        ], axis = -1).to(device)

        self._init_particles_probs = torch.ones(self._num_particles).to(device) / self._num_particles

    def motion_update(self, actions, particles) -> torch.Tensor:
        """
        Motion Update based on Velocity Model

        :param np.ndarray actions: linear and angular velocity commands
        :param torch.Tensor particles: belief represented by particles
        """

        action_input = np.tile(actions, (particles.shape[0], 1))
        action_input = torch.from_numpy(action_input).float().to(device)
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
