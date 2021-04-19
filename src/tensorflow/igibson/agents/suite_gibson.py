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

from gibson2.envs.igibson_env import iGibsonEnv
import gibson2

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers

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

    def step(self, action):
        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        return state, reward, done, info

    def reset(self):
        state = super(NavigateGibsonEnv, self).reset()

        return state

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
