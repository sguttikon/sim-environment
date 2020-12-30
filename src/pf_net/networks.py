#!/usr/bin/env python3

from typing import Iterable, Callable
from torchvision import models
from torch import nn, Tensor
import numpy as np
import datautils
import argparse
import helpers
import torch

class TransitionModel(nn.Module):

    def __init__(self, params):
        super(TransitionModel, self).__init__()

        self.params = params

        self.params.alpha = 0.1

    def forward(self, particle_states: Tensor, odometry: np.ndarray, old_pose: np.ndarray) -> Tensor:

        # translation_std = self.params.transition_std[0]
        # rotation_std = self.params.transition_std[1]
        #
        # parts_x, parts_y, parts_th = particle_states.unbind(dim=-1)
        #
        # odom_x, odom_y, odom_th = odometry
        #
        # # add orientation noise
        # noise_th = torch.normal(mean=0, std=1.0, size=parts_th.shape).to(self.params.device) * rotation_std
        # parts_th = parts_th + noise_th
        #
        # sin_th = torch.sin(parts_th)
        # cos_th = torch.cos(parts_th)
        # delta_x = odom_x * cos_th + odom_y * sin_th
        # delta_y = odom_x * sin_th - odom_y * cos_th
        # delta_th = odom_th
        #
        # noise_x = torch.normal(mean=0, std=1.0, size=parts_th.shape).to(self.params.device) * translation_std
        # noise_y = torch.normal(mean=0, std=1.0, size=parts_th.shape).to(self.params.device) * translation_std
        # delta_x = delta_x + noise_x
        # delta_y = delta_y + noise_y
        #
        # x = parts_x + delta_x
        # y = parts_y + delta_y
        # th = datautils.wrap_angle(parts_th + delta_th, isTensor=True)
        #
        # return torch.stack([x, y, th], axis=-1)

        return self.sample_motion_odometry(particle_states, odometry, old_pose)

    def sample_motion_odometry(self, particle_states: Tensor, odometry: np.ndarray, old_pose: np.ndarray) -> Tensor:
        x1, y1, th1 = old_pose
        abs_x, abs_y, abs_th = odometry

        delta_trans = np.sqrt(abs_y**2 + abs_x**2)
        if delta_trans < 0.01:
            # avoid computing bearing if poses are extremely near => say in-place rotation
            delta_rot1 = 0.0
        else:
            delta_rot1 = helpers.angle_diff(np.arctan2(abs_y, abs_x), th1)
        delta_rot2 = helpers.angle_diff(abs_th, delta_rot1)

        delta_rot1_noise = np.minimum(
                                        np.fabs(helpers.angle_diff(delta_rot1, 0.0)),
                                        np.fabs(helpers.angle_diff(delta_rot1, np.pi))
                                    )

        delta_rot2_noise = np.minimum(
                                        np.fabs(helpers.angle_diff(delta_rot2, 0.0)),
                                        np.fabs(helpers.angle_diff(delta_rot2, np.pi))
                                    )

        alpha1 = self.params.alpha
        alpha2 = self.params.alpha
        alpha3 = self.params.alpha
        alpha4 = self.params.alpha

        delta_rot1_scale = alpha1*delta_rot1**2 + alpha2*delta_trans**2
        delta_trans_scale = alpha3*delta_trans**2 + alpha4*delta_rot1**2 + \
                            alpha4*delta_rot2**2
        delta_rot2_scale = alpha1*delta_rot2**2 + alpha2*delta_trans**2

        # shape [batch_size, particles, 3]
        # iterate per batch
        for b_idx in range(particle_states.shape[0]):
            # iterate per particle
            for p_idx in range(particle_states.shape[1]):
                # sample pose differences
                delta_rot1_hat = helpers.angle_diff(delta_rot1, np.random.normal(0.0, delta_rot1_scale))

                delta_trans_hat = delta_trans - np.random.normal(0.0, delta_trans_scale)

                delta_rot2_hat = helpers.angle_diff(delta_rot2, np.random.normal(0.0, delta_rot2_scale))

                # apply sampled update to particle pose
                particle_states[b_idx, p_idx, 0] = particle_states[b_idx, p_idx, 0] + \
                    delta_trans_hat * torch.cos(particle_states[b_idx, p_idx, 2] + delta_rot1_hat)
                particle_states[b_idx, p_idx, 1] = particle_states[b_idx, p_idx, 1] + \
                    delta_trans_hat * torch.sin(particle_states[b_idx, p_idx, 2] + delta_rot1_hat)
                particle_states[b_idx, p_idx, 2] = particle_states[b_idx, p_idx, 2] + \
                    delta_rot1_hat + delta_rot2_hat

        return particle_states

class FeatureExtractor(nn.Module):
    """
    reference: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    """
    def __init__(self, model: nn.Module, layers = Iterable[str]):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        # register a hook for each layer
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._features

class LikelihoodNetwork(nn.Module):

    def __init__(self):
        super(LikelihoodNetwork, self).__init__()
        self.in_features = 512 + 4
        self.out_features = 1

        # model
        self.model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024), # shape: [N, 4]
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024), # shape: [N, 32]
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.out_features), # shape: [N, 32]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        out = out.squeeze(2)
        lik = torch.sigmoid(out)
        return lik

class ObservationModel(nn.Module):

    def __init__(self, params):
        super(ObservationModel, self).__init__()

        self.params = params
        self.resnet = models.resnet34(pretrained=self.params.pretrained_model).to(self.params.device)

        if self.params.pretrained_model:
            # no need to train resnet
            for param in self.resnet.parameters():
                param.requires_grad = False
            layers = ['layer4', 'avgpool']
            self.feature_extractor = FeatureExtractor(self.resnet, layers).to(self.params.device)
        else:
            # train resnet
            self.resnet.fc = nn.Identity()

        self.likelihood_net = LikelihoodNetwork().to(self.params.device)

    def forward(self, particle_states: Tensor, obs: Tensor) -> Tensor:
        if self.params.pretrained_model:
            features = self.feature_extractor(obs)['avgpool']
        else:
            features = self.resnet(obs)

        img_features = features.view(obs.shape[0], 1, -1)
        repeat_img_features = img_features.repeat(1, particle_states.shape[1], 1)

        trans_particle_states = datautils.transform_poses(particle_states)
        input_est_features = torch.cat([trans_particle_states, repeat_img_features], axis=-1)

        lik = self.likelihood_net(input_est_features)
        return lik

    def set_train_mode(self):
        self.resnet.train()
        self.likelihood_net.train()
        if self.params.pretrained_model:
            self.feature_extractor.train()

    def set_eval_mode(self):
        self.resnet.eval()
        self.likelihood_net.eval()
        if self.params.pretrained_model:
            self.feature_extractor.eval()

class ResamplingModel(nn.Module):

    def __init__(self, params):
        super(ResamplingModel, self).__init__()

        self.params = params

    def forward(self, particle_states: Tensor, particle_weights: Tensor) -> (Tensor, Tensor):

        batch_size, num_particles, _ = particle_states.shape

        alpha = self.params.alpha_resample_ratio
        assert 0.0 < alpha <= 1.0

        # normalize
        particle_weights = particle_weights - torch.logsumexp(particle_weights, dim=-1, keepdim=True)

        # uniform weights
        uniform_weights = torch.ones((batch_size, num_particles)).to(self.params.device) * -np.log(num_particles)

        # build sampling distribution, q(s) and update particle weights
        if alpha < 1.0:
            # soft resampling
            q_weights = torch.stack([particle_weights + np.log(alpha), uniform_weights + np.log(1.0-alpha)], axis=-1)
            q_weights = torch.logsumexp(q_weights, dim=-1, keepdim=False)
            # normalize
            q_weights = q_weights - torch.logsumexp(q_weights, dim=-1, keepdim=True)

            particle_weights = particle_weights - q_weights # unnormalized
        else:
            # hard resampling -> results zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        m = torch.distributions.categorical.Categorical(logits=q_weights)
        indices = []
        for i in range(batch_size):
            idx = []
            for j in range(num_particles):
                idx.append(m.sample())
            indices.append(torch.stack(idx).squeeze(1))
        indices = torch.stack(indices)

        helper = torch.arange(0, batch_size * num_particles, step=num_particles, dtype=torch.int64).to(self.params.device) # (batch, )
        indices = indices + helper.unsqueeze(1)
        indices = torch.reshape(indices, (batch_size * num_particles, ))

        particle_states = torch.reshape(particle_states, (batch_size * num_particles, 3))
        particle_states = particle_states[indices].view(batch_size, num_particles, -1)

        particle_weights = torch.reshape(particle_weights, (batch_size * num_particles, ))
        particle_weights = particle_weights[indices].view(batch_size, num_particles)

        return particle_states, particle_weights

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--transition_std', nargs='*', default=[0.05, 0.05], help='standard deviations for transition model: translation std and rotation std')
    argparser.add_argument('--pretrained_model', type=bool, default=True, help='use pretrained models')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='trade-off parameter for soft-resampling 0.0 < alpha <= 1.0' )
    argparser.add_argument('--num_particles', type=int, default=5, help='number of particles')
    argparser.add_argument('--seed', type=int, default=42, help='random seed value to set')

    params = argparser.parse_args()
    params.device = torch.device('cuda')

    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)

    trans_model = TransitionModel(params)
    old_pose = np.array([ 0.5757,  0.4682, -1.3175])
    new_pose = np.array([ 0.5892,  0.4159, -1.3187])
    # old_pose = np.array([ 0.5892,  0.4159, -1.3187])
    # new_pose = np.array([ 0.6027,  0.3636, -1.3201])

    odometry = datautils.calc_odometry(old_pose, new_pose)

    state = torch.from_numpy(old_pose).unsqueeze(0).unsqueeze(1)
    particle_states = state.repeat(1, params.num_particles, 1).float().to(params.device)
    particle_states = trans_model(particle_states, odometry)
    particle_weights = torch.ones((1, params.num_particles)).to(params.device) * np.log(1.0/params.num_particles)

    obs_model = ObservationModel(params)
    observation = torch.rand(1, 3, 256, 256).float().to(params.device)
    liklihoods = obs_model(particle_states, observation)
    particle_weights = particle_weights + liklihoods

    resample_model = ResamplingModel(params)
    particle_states, particle_weights = resample_model(particle_states, particle_weights)
    print(particle_weights)
