#!/usr/bin/env python3

from pfnet import PFNet
import argparse
import torch
import os

def parse_args():

    argparser = argparse.ArgumentParser()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(curr_dir, 'turtlebot.yaml')
    # refer http://svl.stanford.edu/igibson/docs/environments.html
    argparser.add_argument('--config_filepath', type=str, default=filepath, help='path to the environment .yaml file')
    argparser.add_argument('--env_mode', type=str, default='headless', help='choice are [headless, gui]')
    argparser.add_argument('--map_pixel_in_meters', type=float, default=0.1, help='the width (and height) of a pixel of the map in meters')

    argparser.add_argument('--pretrained_model', type=bool, default=True, help='use pretrained models')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='trade-off parameter for soft-resampling 0.0 < alpha <= 1.0' )
    argparser.add_argument('--num_particles', type=int, default=5000, help='number of particles')
    argparser.add_argument('--transition_std', nargs='*', default=[0.1, 0.1], help='standard deviations for transition model: translation std and rotation std')
    argparser.add_argument('--seed', type=int, default=42, help='random seed value to set')

    if torch.cuda.is_available():
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')

    params = argparser.parse_args()

    return params

if __name__ == '__main__':
    params = parse_args()

    pf_net = PFNet(params)
    del pf_net
