#!/usr/bin/env python3

from collections import OrderedDict
from gibson2.envs.igibson_env import iGibsonEnv
import numpy as np
import gym

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
