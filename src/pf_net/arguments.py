#!/usr/bin/env python3

import argparse
import os
from pfnet import PFNet

def parse_args():

    argparser = argparse.ArgumentParser()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(curr_dir, 'turtlebot.yaml')
    # refer http://svl.stanford.edu/igibson/docs/environments.html
    argparser.add_argument('--config_filepath', type=str, default=filepath, help='path to the environment .yaml file')
    argparser.add_argument('--env_mode', type=str, default='headless', help='choice are [headless, gui]')
    argparser.add_argument('--map_pixel_in_meters', type=float, default=0.1, help='the width (and height) of a pixel of the map in meters')

    argparser.add_argument('--num_particles', type=int, default=5000, help='number of particles')
    argparser.add_argument('--transition_std', nargs='*', default=[0.1, 0.1], help='standard deviations for transition model: translation std and rotation std')
    argparser.add_argument('--seed', type=int, default=42, help='random seed value to set')

    params = argparser.parse_args()

    return params

if __name__ == '__main__':
    params = parse_args()

    pf_net = PFNet(params)
    del pf_net
