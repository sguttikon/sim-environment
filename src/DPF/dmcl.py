#!/usr/bin/env python3

import numpy as np
import torch
import utils.constants as constants
import networks.networks as nets
import utils.helpers as helpers
import random

np.random.seed(42)
random.seed(42)
if constants.IS_CUDA:
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

class DMCL():
    """
    """

    def __init__(self, robot):
        super(DMCL, self).__init__()

        self.odom_net = nets.OdomNetwork().to(constants.DEVICE)
        self.vision_net = nets.VisionNetwork().to(constants.DEVICE)
        self.likelihood_net = nets.LikelihoodNetwork().to(constants.DEVICE)
        self.particles_net = nets.ParticlesNetwork().to(constants.DEVICE)

        params = list(self.vision_net.parameters()) + \
                 list(self.likelihood_net.parameters())
                 # TODO add odom net parameters for optimization
        self.optimizer = torch.optim.Adam(params, lr=2e-4)
        self.robot = robot

    def train_mode(self):
        self.odom_net.train()
        self.vision_net.train()
        self.likelihood_net.train()
        self.particles_net.train()

    def eval_model(self):
        self.odom_net.eval()
        self.vision_net.eval()
        self.likelihood_net.eval()
        self.particles_net.eval()

    def to_tensor(self, array):
        return torch.from_numpy(array.copy()).float().to(constants.DEVICE)

    def init_particles(self, init_pose):
        """
        """

        limits = 5
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

        return self.to_tensor(rnd_particles), self.to_tensor(rnd_probs)

    def transform_particles(self, particles):
        return torch.cat([
            particles[:, 0:2], torch.cos(particles[:, 2:3]), torch.sin(particles[:, 2:3])
        ], axis=-1)

    def transform_pose(self, pose):
        return torch.cat([
            pose[0:2], torch.cos(pose[2:3]), torch.sin(pose[2:3])
        ], axis=-1)

    def step(self, imgs, acts, particles):
        """
        """
        self.train_mode()

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

        # --------- Loss ----------- #
        #particles_probs = particles_probs * obs_likelihoods
        #particles_probs = torch.div(particles_probs, torch.sum(particles_probs))
        particles_probs = torch.div(obs_likelihoods, torch.sum(obs_likelihoods))

        mean_particles = torch.sum(trans_particles*particles_probs.unsqueeze(1), axis=0)
        gt_pose = self.transform_pose(self.to_tensor(helpers.get_gt_pose(self.robot)))
        sq_dist = helpers.eucld_dist(gt_pose, mean_particles)

        total_loss = sq_dist

        # --------- Backward Pass --------- #
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        # --------- Particle Network -------- #
        particles = self.particles_net(particles, particles_probs)

        particles = particles.detach() # stop gradient flow here

        return particles, total_loss, 0.
