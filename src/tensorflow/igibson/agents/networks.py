#!/usr/bin/env python3

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
# from matplotlib import pyplot as plt
import torchvision.models as models
from stable_baselines3 import PPO
import torch.nn as nn
import numpy as np
import torch
import gym

class CustomCNN(BaseFeaturesExtractor):
    """
    :reference https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.image_shape = (128, 128, 3) # [H, W, C]
        self.prorio_shape = 20

        block1_layers = [
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2, dilation=1, bias=True),
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=4, dilation=2, bias=True),
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=8, dilation=4, bias=True)
        ]
        self.block1 = nn.ModuleList(block1_layers)

        size = [384, 28, 28]
        block2_layers= [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(size),
            nn.ReLU()
        ]
        self.block2 = nn.ModuleList(block2_layers)

        size = [16, 14, 14]
        block3_layers= [
            nn.Conv2d(384, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(size),
            nn.ReLU()
        ]
        self.block3 = nn.ModuleList(block3_layers)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.image_shape[2], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            observations = torch.as_tensor(observation_space.sample()[None]).float()
            robot_state, rgb_obs = self._process_env_obs(observations)

            # n_flatten = self.prorio_shape + self._get_obs_features(rgb_obs).shape[1]
            n_flatten = self.prorio_shape + self.cnn(rgb_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _get_obs_features(self, observation):
        x = observation

        # block1
        convs = []
        for _, l in enumerate(self.block1):
            convs.append(l(x))
        x = torch.cat(convs, axis=1)

        # block2
        for _, l in enumerate(self.block2):
            x = l(x)

        # block3
        for _, l in enumerate(self.block3):
            x = l(x)

        x = nn.Flatten()(x) # [batch_size, 16, 14, 14]
        return x    # [batch_size, 9216]

    def _process_env_obs(self, observations):
        robot_state, rgb_obs = torch.split(
                            observations,
                            [self.prorio_shape, observations.shape[-1] - self.prorio_shape],
                            dim=-1
        )
        rgb_obs = torch.reshape(rgb_obs, [-1, *self.image_shape])  # [1, H, W, C]
        rgb_obs = rgb_obs.permute(0, 3, 1, 2) # [1, C, H, W]

        return robot_state, rgb_obs

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        robot_state, rgb_obs = self._process_env_obs(observations)

        # # visualize
        # env_obs = rgb_obs.permute(0, 2, 3, 1)[0].cpu().numpy()
        # plt.imshow(env_obs)
        # plt.show()

        #rgb_features = self._get_obs_features(rgb_obs)
        rgb_features = self.cnn(rgb_obs)
        combined_features = torch.cat([robot_state, rgb_features], axis=-1)

        return self.linear(combined_features)

if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    env = gym.make("BreakoutNoFrameskip-v4")
    env = DummyVecEnv([lambda: env])
    print(type(env), env.observation_space)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(1000)
    env = model.get_env()
    print(type(env), env.observation_space)
