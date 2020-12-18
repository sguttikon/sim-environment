#!/usr/bin/env python3
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import utils.constants as constants
import utils.helpers as helpers
import numpy as np
from typing import Dict, Iterable, Callable

class VisionNetwork(nn.Module):
    """
    """

    def __init__(self, w, h):
        super(VisionNetwork, self).__init__()

        # w, h, kernel, padding, stride
        w, h = self.cal_out_dim(w, h, 8, 0, 4)
        w, h = self.cal_out_dim(w, h, 4, 0, 2)
        w, h = self.cal_out_dim(w, h, 3, 0, 1)
        in_features = 1 * 64 * w * h

        self.in_features = 3 + 3 # include map image

        # model
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=32, kernel_size=8, stride=4), # shape: [N, 3, 128, 128]
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
    """
    """

    def __init__(self):
        super(LikelihoodNetwork, self).__init__()
        self.in_features = constants.VISUAL_FEATURES + 4
        self.out_features = 1

        # model
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=1024) # shape: [N, self.in_features]
        self.fc2 = nn.Linear(in_features=1024, out_features=1024) # shape: [N, 1024]
        self.fc3 = nn.Linear(in_features=1024, out_features=self.out_features) # shape: [N, 1024]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embedding = x
        x = self.fc3(x)
        # approach [p, img + 4]
        x = torch.sigmoid(x)
        return embedding, x # shape: [N, self.out_features]

class SeqLikeliNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(SeqLikeliNetwork, self).__init__()

        hidden_size = 128
        num_layers = 2
        input_size = constants.SEQ_LEN*hidden_size + (constants.NUM_PARTICLES * 4)
        self.lstm = nn.LSTM(constants.VISUAL_FEATURES, hidden_size, num_layers, batch_first=True) # shape: [N, seq_len, 64]
        self.linear1 = nn.Linear(input_size, 128) # shape: [N, seq_len*128 + particles*4]
        self.linear2 = nn.Linear(128, constants.NUM_PARTICLES) # shape: [N, 128]
        self.soft_max = nn.Softmax(dim=1) # shape: [N, particles]

    def forward(self, imgs, poses):
        batch_size = imgs.shape[0]
        lstm_out, _ = self.lstm(imgs)
        input_features = torch.cat([lstm_out.view(batch_size, -1), poses.view(batch_size, -1)], axis=-1)
        embedding = self.linear1(input_features)
        x = F.relu(embedding)
        x = self.linear2(x)
        probs = self.soft_max(x)
        return embedding, probs

class MotionNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(MotionNetwork, self).__init__()
        self.in_features = 2 * constants.ACTION_DIMS

        # model
        self.noise_gen_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=32), # shape: [N, 4]
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32), # shape: [N, 32]
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=constants.STATE_DIMS), # shape: [N, 32]
        )

    def forward(self, old_poses, actions, delta_t, use_noise=True, simple_model=True):
        # reference: probabilistic robotics: 'algorithm sample_motion_model_velocity()'

        old_poses = helpers.to_tensor(old_poses)
        delta_t = helpers.to_tensor(delta_t)[:, None]
        # ------- learn the noisy actions ------- #
        rnd_input = np.random.normal(loc=0.0, scale=1.0, size=actions.shape)
        input = np.concatenate((actions, rnd_input), axis=-1)
        input = helpers.to_tensor(input)
        action_delta = self.noise_gen_model(input)

        noisy_actions = action_delta
        noisy_actions[:, 0:2] = noisy_actions[:, 0:2] + helpers.to_tensor(actions)

        # noisy actions
        lin_vel_hat = noisy_actions[:, 0:1]*delta_t
        ang_vel_hat = noisy_actions[:, 1:2]*delta_t
        gamma_hat = noisy_actions[:, 2:3]*delta_t

        x = old_poses[:, 0:1]
        y = old_poses[:, 1:2]
        theta = helpers.wrap_angle(old_poses[:, 2:3])
        if simple_model:
            x_prime = x + lin_vel_hat*torch.cos(theta)
            y_prime = y + ang_vel_hat*torch.sin(theta)
            theta_prime = helpers.wrap_angle(theta + ang_vel_hat)
        else:
            radius = torch.div(lin_vel_hat, ang_vel_hat + 1e-8)
            x_prime = x + radius*(torch.sin(theta + ang_vel_hat) - torch.sin(theta))
            y_prime = y + radius*(torch.cos(theta) - torch.cos(theta + ang_vel_hat))
            theta_prime = helpers.wrap_angle(theta + ang_vel_hat + gamma_hat)

        new_poses = torch.cat([x_prime, y_prime, theta_prime], axis=-1)
        return new_poses # shape: [N, 3]

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
            while i<particles.shape[0]-1 and rnd_offset > cum_sum:
                i = i + 1
                cum_sum = cum_sum + particles_probs[i]
            new_particles.append(particles[i])
            rnd_offset = rnd_offset + step
        particles = torch.stack(new_particles, axis=0)
        return particles

class ActionNetwork(nn.Module):
    """
    """

    def __init__(self):
        super(ActionNetwork, self).__init__()

        # model
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=constants.VISUAL_FEATURES, out_features=128), # shape: [N, 64]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128), # shape: [N, 128]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=constants.ACTION_DIMS), # shape: [N, 128]
        )

    def forward(self, x):
        x = self.fc_model(x)
        return x # shape: [N, 2]

class SampleMotionModel(nn.Module):
    """
    """

    def __init__(self):
        super(SampleMotionModel, self).__init__()
        pass

    def forward(self, old_poses, actions, delta_ts, use_noise=True, simple_model=True):
        # reference: probabilistic robotics: 'algorithm sample_motion_model_velocity()'
        if use_noise:
            # ------- don't learn the noisy actions ------- #
            alpha1 = alpha2 = alpha3 = alpha4 = alpha5 = alpha6 = 0.02
            lin_vel = actions[:, 0]
            ang_vel = actions[:, 1]

            std1 = np.sqrt(alpha1*lin_vel*lin_vel + alpha2*ang_vel*ang_vel)
            std2 = np.sqrt(alpha3*lin_vel*lin_vel + alpha4*ang_vel*ang_vel)
            std3 = np.sqrt(alpha5*lin_vel*lin_vel + alpha6*ang_vel*ang_vel)

            # noisy actions
            lin_vel_hat = lin_vel + np.random.normal(loc=.0, scale=std1)*delta_ts
            ang_vel_hat = ang_vel + np.random.normal(loc=.0, scale=std2)*delta_ts
            gamma_hat = np.random.normal(loc=.0, scale=std3)*delta_ts
        else:
            # noisy actions
            lin_vel_hat = actions[:, 0]*delta_ts
            ang_vel_hat = actions[:, 1]*delta_ts
            gamma_hat = 0

        x = old_poses[:, 0:1]
        y = old_poses[:, 1:2]
        theta = helpers.wrap_angle(old_poses[:, 2:3], use_numpy=True)
        if simple_model:
            x_prime = x + lin_vel_hat*np.cos(theta)
            y_prime = y + ang_vel_hat*np.sin(theta)
            theta_prime = helpers.wrap_angle(theta + ang_vel_hat, use_numpy=True)
        else:
            radius = np.divide(lin_vel_hat, ang_vel_hat + 1e-8)
            x_prime = x + radius*(np.sin(theta + ang_vel_hat) - np.sin(theta))
            y_prime = y + radius*(np.cos(theta) - np.cos(theta + ang_vel_hat))
            theta_prime = helpers.wrap_angle(theta + ang_vel_hat + gamma_hat, use_numpy=True)

        new_poses = np.concatenate([x_prime, y_prime, theta_prime], axis=-1)

        return new_poses # shape: [N, 3]
