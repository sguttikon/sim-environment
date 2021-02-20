#!/usr/bin/env python3

import os
import numpy as np
import pybullet as p
from iGibson_env import iGibsonEnv

np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':
    config_filename = os.path.join('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson/configs/', 'turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    print(env.observation_space)
    print(env.action_space)

    for episode in range(5):
        print(f'Episode: {episode}')
        state = env.reset()
        print(f'robot pose: {state["pose"]}')
        for _ in range(20):  # 2 seconds
            action = [0.5, 0.5] # env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print(f'robot pose: {state["pose"]}')
    env.close()
