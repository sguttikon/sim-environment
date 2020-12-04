#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'openai_ros' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src')

import measurement as m
import utils.constants as constants
import numpy as np
import torch

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)
np.set_printoptions(precision=3)

if __name__ == '__main__':
    print('running measurement model training')
    measurement = m.Measurement()
    train_epochs = 500
    eval_epochs = 5
    measurement.train(train_epochs, eval_epochs)
