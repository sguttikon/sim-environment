#!/usr/bin/env python3

from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
from gibson2.core.render.profiler import Profiler
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from transforms3d.euler import quat2euler

def load_floor(config_data: dict) -> np.ndarray:
    """
    """

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

def getKey() -> int:
    key = input('Enter key: ')
    if key == '':
        key = -1
    else:
        key = ord(key)
    return key

def wrap_angle(angle):
    return ( (angle-np.pi) % (2*np.pi) ) - np.pi

def get_pose(robot):
    position = robot.get_position()
    euler = quat2euler(robot.get_orientation())
    pose = np.array([position[0], position[2], wrap_angle(euler[0])])
    return pose

def teleop_robot(nav_env: NavigateEnv):
    """
    """

    msg = """
Moving around:
    w
a       d
    s
anything else : stop
q to quit
"""
    print(msg)
    state = nav_env.reset()
    turtlebot = nav_env.robots[0]
    for i in range(200):
        key = getKey()
        if key == 113:
            # quit
            break
        elif (key, ) in turtlebot.keys_to_action.keys():
            action = turtlebot.keys_to_action[(key, )]
        else:
            action = turtlebot.keys_to_action[()]

        state, reward, done, info = nav_env.step(action)
        episode_data = {
            'step': i,
            'state':{
                'sensor': state['sensor'],
                'rgb': state['rgb'],
                'depth': state['depth'],
                'scan': state['depth'],
                'pose': get_pose(turtlebot),
            },
            'reward': reward,
            'done': done,
        }
        print(turtlebot.get_position(), quat2euler(turtlebot.get_orientation()))
        if done:
            print('Episode finished')
            break

def main():
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))

    # configuration file contains: robot, scene, etc. details
    config_file_path = os.path.join(curr_dir_path, 'turtlebot.yaml')
    config_data = parse_config(config_file_path)

    trav_map = load_floor(config_data)

    nav_env = NavigateEnv(config_file=config_file_path, mode='gui')
    teleop_robot(nav_env)

def collect_data():
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))

    # configuration file contains: robot, scene, etc. details
    config_file_path = os.path.join(curr_dir_path, 'turtlebot.yaml')

    nav_env = NavigateEnv(config_file=config_file_path, mode='gui')
    for episode_idx in range(10):
        nav_env.reset()
        for step_idx in range(100):
            with Profiler('Env action step'):
                action = nav_env.action_space.sample()
                state, reward, done, info = nav_env.step(action)
                if done:
                    print('Episode finished after {0} timesteps'.format(step_idx + 1))
                    break

if __name__ == '__main__':
    main()
    #collect_data()
