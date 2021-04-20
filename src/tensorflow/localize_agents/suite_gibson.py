# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from gibson2.envs.igibson_env import iGibsonEnv
import gibson2
import gym
import numpy as np
import os

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
# path to custom tf_agents
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/stanford/agents')
# set_path('/home/guttikon/awesome_robotics/sim-environment/src/tensorflow/stanford/agents')

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.policies import random_tf_policy

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

        observation_space = OrderedDict()
        IMG_WIDTH = 56
        IMG_HEIGHT = 56
        TASK_OBS_DIM = 20

        # observation_space['task_obs'] = gym.spaces.Box(
        #         low=-np.inf, high=+np.inf,
        #         shape=(TASK_OBS_DIM,),    # task_obs + proprioceptive_obs
        #         dtype=np.float32)
        observation_space['rgb'] = gym.spaces.Box(
                low=-1.0, high=+1.0,
                shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

    def step(self, action):
        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        custom_state = OrderedDict()
        # custom_state['task_obs'] = np.concatenate([
        #                 self.task.get_task_obs(self)[:-2], # goal x,y relative distance
        #                 self.robots[0].calc_state(),    # proprioceptive state
        #             ], 0)
        custom_state['rgb'] = state['rgb']  # [0, 1] range rgb image

        return custom_state, reward, done, info

    def reset(self):
        if np.random.uniform() < 0.5:
            self.task.target_pos = np.array([0.3, 0.9, 0.0])# np.array([-0.5, 0.9, 0.0])
        else:
            self.task.target_pos = np.array([0.2, -0.2, 0.0])# np.array([-0.5, -0.2, 0.0])

        state = super(NavigateGibsonEnv, self).reset()

        custom_state = OrderedDict()
        # custom_state['task_obs'] = np.concatenate([
        #                 self.task.get_task_obs(self)[:-2], # goal x,y relative distance
        #                 self.robots[0].calc_state(),    # proprioceptive state
        #             ], 0)
        custom_state['rgb'] = state['rgb']  # [0, 1] range rgb image

        return custom_state

def load(config_file,
         model_id=None,
         env_mode='headless',
         action_timestep=1.0 / 10.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    env = NavigateGibsonEnv(config_file=config_file,
                     scene_id=model_id,
                     mode=env_mode,
                     action_timestep=action_timestep,
                     physics_timestep=physics_timestep,
                     device_idx=device_idx)

    discount = env.config.get('discount_factor', 0.99)
    max_episode_steps = env.config.get('max_step', 500)

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
    )


def wrap_env(env,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=wrappers.TimeLimit,
             env_wrappers=(),
             spec_dtype_map=None,
             auto_reset=True):
    for wrapper in gym_env_wrappers:
        env = wrapper(env)
    env = gym_wrapper.GymWrapper(
        env,
        discount=discount,
        spec_dtype_map=spec_dtype_map,
        match_obs_space_dtype=True,
        auto_reset=auto_reset,
        simplify_box_bounds=True
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env

if __name__ == '__main__':
    eval_py_env = load(
        config_file=os.path.join('../configs/', 'turtlebot_navigate.yaml'),
        env_mode='gui',
        device_idx=0,
    )
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    rnd_policy = random_tf_policy.RandomTFPolicy(
                    time_step_spec=eval_tf_env.time_step_spec(),
                    action_spec=eval_tf_env.action_spec())

    for _ in range(5):
        time_step = eval_tf_env.reset()
        for _ in range(100):
            action_step = rnd_policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
