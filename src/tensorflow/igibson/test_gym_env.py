#!/usr/bin/env python3

import gym
import custom_env

if __name__ == '__main__':

    # create a new gym environment
    env = gym.make('custom_env:iGibson-v0')

    for _  in range(10):
        robot_pose, init_particles = env.reset()
        print(robot_pose)

    env.close()
