#!/usr/bin/env python3

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gibson2.envs.igibson_env import iGibsonEnv
from stable_baselines3 import PPO
from utils import datautils
import tensorflow as tf
import torch.nn as nn
import numpy as np
import torch
import gym
import os

class NavigateGibsonEnv(iGibsonEnv):

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):

        super(NavigateGibsonEnv, self).__init__(config_file=config_file,
                        scene_id=scene_id,
                        mode=mode,
                        action_timestep=action_timestep,
                        physics_timestep=physics_timestep,
                        device_idx=device_idx,
                        render_to_tensor=render_to_tensor,
                        automatic_reset=automatic_reset)

        output_size = 18 + np.prod((56, 56, 3))

        self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(output_size, ),
                dtype=np.float32)

    def step(self, action):
        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])

        custom_state = np.concatenate([self.robots[0].calc_state(),
                            np.reshape(rgb, [-1])], 0)

        return custom_state, reward, done, info

    def reset(self):
        state = super(NavigateGibsonEnv, self).reset()

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])

        custom_state = np.concatenate([self.robots[0].calc_state(),
                            np.reshape(rgb, [-1])], 0)

        return custom_state

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

        # to determine device dynamically
        self.dummy_param = nn.Parameter(torch.empty(0))

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

def train_action_sampler(device, timesteps):

    # create gym env
    config_filename = os.path.join('./configs/', 'turtlebot_navigate.yaml')
    env = NavigateGibsonEnv(config_file=config_filename, mode='headless')
    state = env.reset()

    # Train the agent
    policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
                                robot_state_size=18,
                                image_shape=(56, 56, 3),
                            ),
    net_arch=[256, 256],
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device)
    model.learn(total_timesteps=timesteps)
    model.save("ppo_navigate_agent")

    del model # remove to demonstrate saving and loading
    env.close()

    print('training finished')

def test_action_sampler(device):

    # fix seed
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # create gym env
    config_filename = os.path.join('./configs/', 'turtlebot_navigate.yaml')
    env = NavigateGibsonEnv(config_file=config_filename, mode='gui')

    # Test the agent
    policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
                                robot_state_size=18,
                                image_shape=(56, 56, 3),
                            ),
    net_arch=[256, 256],
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device)
    model = PPO.load('ppo_navigate_agent')

    for _ in range(5):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                print(reward)
                break

    env.close()

    print('testing finished')

if __name__ == '__main__':
    train_action_sampler(device=0, timesteps=50000)
    # test_action_sampler(device=0)
