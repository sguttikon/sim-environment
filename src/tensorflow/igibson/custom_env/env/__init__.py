#!/usr/bin/env python3

import sys

def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'custom_env' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson/custom_env')

from custom_env.env.iGibsonEnv import iGibsonEnv
