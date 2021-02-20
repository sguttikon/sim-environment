#!/usr/bin/env python3

from gym.envs.registration import register

register(
    id='iGibson-v0',
    entry_point='custom_env.env:iGibsonEnv',
)
