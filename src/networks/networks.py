#!/usr/bin/env python3
import torch
import torch.nn as nn
import utils.constants as constants
import utils.helpers as helpers
import numpy as np

class VisionNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(VisionNetwork, self).__init__()

        # w, h, kernel, padding, stride
        w, h = 128, 128
        w, h = self.cal_out_dim(w, h, 8, 0, 4)
        w, h = self.cal_out_dim(w, h, 4, 0, 2)
        w, h = self.cal_out_dim(w, h, 3, 0, 1)
        in_features = 1 * 64 * w * h

        # TODO: also include map of environment ??

        # model
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), # shape: [N, 3, 128, 128]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), # shape: [N, 32, 31, 31]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), # shape: [N, 32, 14, 14]
            nn.ReLU(),
            nn.Flatten(), # shape: [N, 64, 12, 12]
            nn.Linear(in_features=in_features, out_features=512), # shape: [N, ...]
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64), # shape: [N, 512]
        )

    def forward(self, x):
        x = self.conv_model(x)
        return x # shape: [N, 64]

    def cal_out_dim(self, w, h, kernel, padding, stride):
        """
        """
        w = (w - kernel + 2*padding)//stride + 1
        h = (h - kernel + 2*padding)//stride + 1
        return w, h

class LikelihoodNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(LikelihoodNetwork, self).__init__()
        self.in_features = constants.VISUAL_FEATURES + 4

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=128), # shape: [N, 67]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128), # shape: [N, 128]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1), # shape: [N, 128]
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        x = self.fc_model(x)
        min_obs_likelihood =  0.004
        x = x*(1 - min_obs_likelihood) + min_obs_likelihood # is this step required ?
        return x # shape: [N, 1]

class OdomNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(OdomNetwork, self).__init__()
        # TODO

    def forward(self, particles, acts):
        # reference: probabilistic robotics: 'algorithm sample_motion_model_velocity()'

        # -------- NON-LEARNING --------- #
        alpha1 = alpha2 = alpha3 = alpha4 = alpha5 = alpha6 = 0.2
        delta_t = 1.

        lin_vel = acts[0]
        ang_vel = acts[1]

        # noisy linear velocity
        sigma = np.sqrt(alpha1*lin_vel*lin_vel + alpha2*ang_vel*ang_vel)
        lin_vel_hat = (lin_vel + np.random.normal(0., sigma)) * delta_t
        # noisy angular velocity
        sigma = np.sqrt(alpha3*lin_vel*lin_vel + alpha4*ang_vel*ang_vel)
        ang_vel_hat = (ang_vel + np.random.normal(0., sigma)) * delta_t
        # bearing correction
        sigma = np.sqrt(alpha5*lin_vel*lin_vel + alpha6*ang_vel*ang_vel)
        bearing_hat = np.random.normal(0., sigma) * delta_t

        radius = torch.div(lin_vel_hat, ang_vel_hat)
        theta = particles[:, 2:3]

        particles[:, 0:1] = particles[:, 0:1] - radius*torch.sin(theta) + \
                            radius*torch.sin(theta + ang_vel_hat)
        particles[:, 1:2] = particles[:, 1:2] + radius*torch.cos(theta) - \
                            radius*torch.cos(theta + ang_vel_hat)
        particles[:, 2:3] = helpers.wrap_angle(theta + ang_vel_hat + bearing_hat)

        return particles

class ParticlesNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(ParticlesNetwork, self).__init__()
        # TODO

    def forward(self, particles, particles_probs):
        # reference: stochastic resampling according to particle probabilities

        # TODO: improve implementation
        step = 1./particles.shape[0]
        rnd_offset = np.random.uniform(0, step)
        cum_sum = particles_probs[0]

        i = 0
        new_particles = []
        for particle in particles:
            while i<particles.shape[0] and rnd_offset > cum_sum:
                i = i + 1
                cum_sum = cum_sum + particles_probs[i]
            new_particles.append(particles[i])
            rnd_offset = rnd_offset + step
        particles = torch.stack(new_particles, axis=0)
        return particles
