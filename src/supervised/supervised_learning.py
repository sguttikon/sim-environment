#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'sim-environment' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src')

import measurement as m
import utils.constants as constants
import numpy as np
import torch
import random
from pathlib import Path

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)
np.set_printoptions(precision=3)

Path("saved_models").mkdir(parents=True, exist_ok=True)
Path("best_models").mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    print('running measurement model training')
    measurement = m.Measurement(render=False, pretrained=False)
    train_epochs = 500
    eval_epochs = 5
    measurement.train(train_epochs, eval_epochs)

    # file_name = '../bckp/dec_13/best_models/likelihood_mse_best.pth'
    # measurement.test(file_name)

    del measurement
