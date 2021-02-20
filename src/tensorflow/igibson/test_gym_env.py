#!/usr/bin/env python3

import os
from iGibson_env import iGibsonEnv

if __name__ == '__main__':
    config_filename = os.path.join('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson/configs/', 'turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    env.close()
