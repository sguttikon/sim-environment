#!/usr/bin/env python3

import os
import numpy as np
import pybullet as p
from utils.iGibson_env import iGibsonEnv

np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':
    config_filename = os.path.join('./configs/', 'turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless', show_plot=True)
    print(env.observation_space)
    print(env.action_space)

    np.random.seed(42)

    for episode in range(1):
        print(f'Episode: {episode}')
        state = env.reset()
        robot_state = env.get_robot_state()
        print(f'robot pose: {robot_state["pose"]}')
        for _ in range(50):  # 5 seconds
            env.render()
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            robot_state = env.get_robot_state()
            print(f'robot pose: {robot_state["pose"]}, reward: {reward}, done: {done}')
    env.close()
