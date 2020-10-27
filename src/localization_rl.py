#!/usr/bin/env python3

import os
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from transforms3d.euler import quat2euler
from DPF.dpf import DMCL
import numpy as np
import utils

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
dmcl = DMCL()

def get_pose(robot) -> np.ndarray:
    """
    """
    position = robot.get_position()
    euler = quat2euler(robot.get_orientation())
    pose = np.array([position[0], position[2], utils.wrap_angle(euler[0])])
    return pose

def train_network(env):
    """
    """
    num_epochs = 1
    epoch_len = 100
    epoch = 0

    turtlebot = env.robots[0]
    while epoch < num_epochs:
        epoch += 1

        obs = env.reset()
        init_pose = get_pose(turtlebot)
        particles = dmcl.initialize_particles(init_pose)

        # motion update
        action = env.action_space.sample()
        particles = dmcl.motion_update(action, particles)

        # measurement update
        probs = dmcl.measurement_update(obs, particles)

if __name__ == '__main__':
    # configuration file contains: robot, scene, etc. details
    config_file_path = os.path.join(curr_dir_path, 'turtlebot.yaml')

    mode = 'headless' # []'headless', 'gui']
    render_to_tensor=True
    nav_env = NavigateEnv(config_file=config_file_path,
                            mode=mode,
                            render_to_tensor=render_to_tensor)

    train_network(nav_env)
