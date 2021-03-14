#!/usr/bin/env python3

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import render, datautils, arguments, pfnet_loss
from utils.localize_env import LocalizeGibsonEnv
from stable_baselines3 import PPO
from pathlib import Path
import tensorflow as tf
import torch.nn as nn
import numpy as np
import torch
import gym

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, robot_state_size: int, image_shape):
        self.robot_state_size = robot_state_size
        self.image_shape = image_shape

        out_size = self.robot_state_size + np.prod((16, 14, 14))

        super(CustomCNN, self).__init__(observation_space, features_dim=out_size)

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

    def get_obs_features(self, observation):
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

        return x # [batch_size, 16, 14, 14]

    def forward(self, x):
        shape_flat = np.prod(self.image_shape)
        robot_state, obs = torch.split(x, [x.shape[-1] - shape_flat, shape_flat], dim=-1)

        assert list(robot_state.shape)[-1] == self.robot_state_size
        obs = torch.reshape(obs, [-1, *self.image_shape])  # [1, 56, 56, 3]

        obs = obs.permute(0, 3, 1, 2)   # [1, 3, 56, 56]
        obs_featues = nn.Flatten()(self.get_obs_features(obs))

        return torch.cat([robot_state, obs_featues], axis=-1)

def train_localization_agent(params):
    """
    train rl agent for localization in iGibsonEnv with the parsed arguments
    """

    # create gym env
    env = LocalizeGibsonEnv(params)
    env.reset()

    # Train the rl agent
    policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
                                robot_state_size=18,
                                image_shape=(56, 56, 3),
                            ),
    net_arch=[256, 256],
    )
    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, device=params.gpu_num)
    model.learn(total_timesteps=params.timesteps)
    model.save(params.rl_agent)

    del model # remove to demonstrate saving and loading
    env.close()

    print('training finished')

def test_action_sampler(params):
    """
    test trained rl agent for localization in iGibsonEnv with the parsed arguments
    """

    # create gym env
    env = LocalizeGibsonEnv(params)
    env.reset()

    # Test the agent
    policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
                                robot_state_size=18,
                                image_shape=(56, 56, 3),
                            ),
    net_arch=[256, 256],
    )
    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, device=params.gpu_num)
    model = PPO.load(params.rl_agent)

    for _ in range(1):
        # new episode
        custom_state = env.reset()
        env.render()
        for _ in range(params.trajlen-1):
            # get action
            if params.agent == 'manual':
                action = datautils.get_discrete_action()
            elif params.agent == 'pretrained':
                action, _ = model.predict(custom_state)
            else:
                action = env.action_space.sample()

            # take action
            custom_state, reward, done, info = env.step(action)
            print(reward)
            env.render()

    env.close()
    print('testing finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    params.batch_size = 1
    params.timesteps = 2048
    params.rl_agent = './ppo_localize_agent'

    params.show_plot = False
    params.store_plot = False
    params.out_folder = './episode_runs/'
    Path(params.out_folder).mkdir(parents=True, exist_ok=True)

    train_localization_agent(params)

    # test_action_sampler(params)
