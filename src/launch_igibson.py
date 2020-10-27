#!/usr/bin/env python3

import os
from gibson2.envs.locomotor_env import NavigateEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_floor(config_data: dict) -> np.ndarray:
    """
    """

    print(type(config_data))
    model_id = config_data['model_id']
    erosion = config_data['trav_map_erosion']
    floors = 0

    model_path = get_model_path(model_id)
    with open(os.path.join(model_path, 'floors.txt'), 'r') as f:
        floors = sorted(list( map(float, f.readlines()) ))

    # default assuming ground floor map
    idx = 0
    trav_map = cv2.imread(os.path.join(model_path,
                            'floor_trav_{0}.png'.format(idx))
                            )
    obs_map = Image.open(os.path.join(model_path,
                            'floor_{0}.png'.format(idx))
                            )

    # trav_map[obs_map == 0] = 0
    # trav_map = cv2.erode(trav_map, np.ones( (erosion, erosion) ))

    # plt.figure(idx, figsize=(8, 8))
    # plt.imshow(trav_map, cmap='gray')
    #
    # plt.show()

    return trav_map

def main():
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))

    # configuration file contains: robot, scene, etc. details
    config_file_path = os.path.join(curr_dir_path, 'turtlebot.yaml')
    config_data = parse_config(config_file_path)

    trav_map = load_floor(config_data)
    print(np.unique(trav_map))

    nav_env = NavigateEnv(config_file=config_file_path, mode='gui')
    state = nav_env.reset()
    print(state['sensor'])

if __name__ == '__main__':
    main()
