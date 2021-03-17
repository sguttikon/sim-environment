#!/usr/bin/env python3

from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.utils import l2_distance
from utils import datautils
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

        # distance based reward
        reward = reward - l2_distance(
            self.robots[0].get_position()[:2],
            self.task.target_pos[:2]
        )
        return custom_state, reward, done, info

    def reset(self):
        state = super(NavigateGibsonEnv, self).reset()

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])

        custom_state = np.concatenate([self.robots[0].calc_state(),
                            np.reshape(rgb, [-1])], 0)

        return custom_state
