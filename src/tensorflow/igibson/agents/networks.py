#!/usr/bin/env python3

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torchvision.models as models
from stable_baselines3 import PPO
import torch.nn as nn
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
        self.image_shape = (3, 210, 160)
        self.prorio_shape = 20

        self.cnn = nn.Sequential(
            nn.Conv2d(self.image_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            observations = torch.as_tensor(observation_space.sample()[None]).float()
            _, rgb_obs = torch.split(
                                observations,
                                [self.prorio_shape, observations.shape[-1] - self.prorio_shape],
                                dim=-1
            )
            rgb_obs = torch.reshape(rgb_obs, [-1, *self.image_shape])   # [1, 3, H, W]
            n_flatten = self.cnn(rgb_obs).shape[1] + self.prorio_shape

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        robot_state, rgb_obs = torch.split(
                            observations,
                            [self.prorio_shape, observations.shape[-1] - self.prorio_shape],
                            dim=-1
        )
        rgb_obs = torch.reshape(rgb_obs, [-1, *self.image_shape])  # [1, 3, H, W]
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
