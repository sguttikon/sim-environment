#!/usr/bin/env python3

import os
import gym
import torch
import argparse
import datetime
import cv2 as cv
import numpy as np
import tensorflow as tf
from pathlib import Path
from networks import CustomCNN
from stable_baselines3 import PPO
from stable_baselines3 import SAC
# from matplotlib import pyplot as plt
from gibson2.envs.igibson_env import iGibsonEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson')
# set_path('/home/guttikon/awesome_robotics/sim-environment/src/tensorflow/igibson')
from utils import datautils

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


        # self.observation_space = gym.spaces.Box(
        #         low=-np.inf, high=np.inf,
        #         shape=(20, ),
        #         dtype=np.float32)
        output_size = 20 + np.prod((56, 56, 3))  # [H, W, C]
        self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(output_size, ),
                dtype=np.float32)

    def step(self, action):
        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        # process image for training
        # rgb_obs = datautils.process_raw_image(state['rgb'])
        rgb_obs = state['rgb']

        # # visualize
        # plt.imshow(rgb_obs)
        # plt.show()

        custom_state = np.concatenate([
                        self.task.get_task_obs(self)[:-2], # goal x,y relative distance
                        self.robots[0].calc_state(),    # proprioceptive state
                        np.reshape(rgb_obs, [-1])   # rgb observation
                    ], 0)

        return custom_state, reward, done, info

    def reset(self):
        if np.random.uniform() < 0.5:
            self.task.target_pos = np.array([-0.5, 0.9, 0.0])
        else:
            self.task.target_pos = np.array([-0.5, -0.2, 0.0])

        state = super(NavigateGibsonEnv, self).reset()

        # process image for training
        # rgb_obs = datautils.process_raw_image(state['rgb'])
        rgb_obs = state['rgb']

        custom_state = np.concatenate([
                        self.task.get_task_obs(self)[:-2], # goal x,y relative distance
                        self.robots[0].calc_state(),    # proprioceptive state
                        np.reshape(rgb_obs, [-1])   # rgb observation
                    ], 0)

        return custom_state

def train_action_sampler(params):

    rootdir = './runs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    Path(rootdir).mkdir(parents=True, exist_ok=True)

    # create gym env
    config_filename = os.path.join(
                        '../configs/',
                        'turtlebot_navigate.yaml'
    )
    env = NavigateGibsonEnv(
                    config_file=config_filename,
                    mode='headless',
                    action_timestep=1.0 / 60.0,
                    physics_timestep=1.0 / 60.0,
                    device_idx=params.gpu_num
    )

    # # necessary: wrap the environment
    # vec_env = VecTransposeImage(DummyVecEnv([lambda: env]))

    # Use deterministic actions for evaluation
    logdir = os.path.join(rootdir, 'logs')
    savedir = os.path.join(rootdir, 'best_models')
    eval_callback = EvalCallback(
                    eval_env=env,
                    n_eval_episodes=params.n_eval,
                    eval_freq=params.eval_freq,
                    log_path=logdir,
                    best_model_save_path=savedir,
                    deterministic=True,
                    render=False,
    )

    # Create the callback list
    callback_list = CallbackList([eval_callback])

    # policy_kwargs = dict(
    #     net_arch=[512, 512],
    # )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
                        features_dim=512
        ),
        net_arch=[256, 256],
    )
    model = SAC(
                policy='MlpPolicy',
                env=env,
                tensorboard_log=logdir,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=params.seed,
                device=params.gpu_num,
    )
    model.learn(
                total_timesteps=params.n_train,
                callback=callback_list
    )
    # model.save(rootdir + '/ppo_baselines3_agent')

    del model # remove to demonstrate saving and loading
    env.close()

    print('training finished')

def test_action_sampler(params):

    rootdir = './runs/20210413-090921'
    savedir = os.path.join(rootdir, 'best_models')

    # create gym env
    config_filename = os.path.join(
                        '../configs/',
                        'turtlebot_navigate.yaml'
    )
    env = NavigateGibsonEnv(
                    config_file=config_filename,
                    mode='headless',
                    action_timestep=1.0 / 60.0,
                    physics_timestep=1.0 / 60.0,
                    device_idx=params.gpu_num
    )

    # # necessary: wrap the environment
    # vec_env = VecTransposeImage(DummyVecEnv([lambda: env]))

    # policy_kwargs = dict(
    #     net_arch=[512, 512],
    # )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
                        features_dim=512
        ),
        net_arch=[256, 256],
    )
    model = SAC(
                policy='MlpPolicy',
                env=env,
                batch_size=32,
                buffer_size=5000,
                tensorboard_log=None,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=params.seed,
                device=params.gpu_num)

    custom_objects = dict(
            batch_size=32,
            buffer_size=5000,
    )
    model = SAC.load(
                path=savedir + '/best_model',
                custom_objects=custom_objects)
    print(f'====> loaded pretrained model : {savedir + "/best_model"}')

    # mean_reward, std_reward = evaluate_policy(model, env)
    # print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    for _ in range(5):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                print(f'episode end reward: {reward}')
                break

    env.close()

    print('testing finished')

def parse_args():
    """
    parse command line arguments

    :return dict: dictionary of parameters
    """

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--n_train', type=int, default=1e6)
    argparser.add_argument('--n_eval', type=int, default=5)
    argparser.add_argument('--eval_freq', type=int, default=1e4)
    argparser.add_argument('--seed', type=int, default='42', help='Fix the random seed of numpy and tensorflow.')
    argparser.add_argument('--gpu_num', type=int, default='0', help='use gpu no. to train')

    params = argparser.parse_args()

    params.n_train = int(params.n_train)
    params.n_eval = int(params.n_eval)
    params.eval_freq = int(params.eval_freq)

    # fix seed
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert params.gpu_num < len(gpus)
    if gpus:
        # restrict TF to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[params.gpu_num], 'GPU')
            torch.device(f'cuda:{params.gpu_num}')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            print(e)

    return params

if __name__ == '__main__':
    params = parse_args()

    train_action_sampler(params)

    # test_action_sampler(params)
