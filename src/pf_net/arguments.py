#!/usr/bin/env python3

from pathlib import Path
from pfnet import PFNet
import numpy as np
import argparse
import random
import torch
import os

Path("saved_models").mkdir(parents=True, exist_ok=True)

def parse_args():

    argparser = argparse.ArgumentParser()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(curr_dir, 'turtlebot.yaml')
    # refer http://svl.stanford.edu/igibson/docs/environments.html
    argparser.add_argument('--config_filepath', type=str, default=filepath, help='path to the environment .yaml file')
    argparser.add_argument('--env_mode', type=str, default='headless', help='choice are [headless, gui]')
    argparser.add_argument('--map_pixel_in_meters', type=float, default=0.1, help='the width (and height) of a pixel of the map in meters')
    argparser.add_argument('--init_particles_std', nargs='*', default=[0.1, 0.1], help='standard deviations for generated initial particles: translation std and rotation std')
    argparser.add_argument('--init_particles_model', type=str, default='GAUSS', help='choice are [GAUSS, UNIFORM]')

    argparser.add_argument('--pretrained_model', type=bool, default=True, help='use pretrained models')
    argparser.add_argument('--render', type=bool, default=False, help='to render the plots')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='trade-off parameter for soft-resampling 0.0 < alpha <= 1.0' )
    argparser.add_argument('--num_particles', type=int, default=5000, help='number of particles')
    argparser.add_argument('--transition_std', nargs='*', default=[0.05, 0.05], help='standard deviations for transition model: translation std and rotation std')
    argparser.add_argument('--seed', type=int, default=42, help='random seed value to set')
    argparser.add_argument('--num_eps', type=int, default=1000, help='number of episodes to train')
    argparser.add_argument('--eps_len', type=int, default=50, help='length of each episode')

    params = argparser.parse_args()

    if torch.cuda.is_available():
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')

    # set common seed value
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    return params

if __name__ == '__main__':
    params = parse_args()

    pf_net = PFNet(params)
    pf_net.run_training()

    # file_name = '../bckp/dec_23/saved_models/pfnet_eps_990.pth'
    # pf_net.run_validation(file_name)

    # pf_net.run_manual()

    del pf_net
