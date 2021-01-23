#!/usr/bin/env python3

from torch.utils.data import DataLoader
from matplotlib.patches import Wedge
from matplotlib.patches import Arrow
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import pf

def load(model, file_name):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['pf_cell'])
    return model

def draw_map(global_map):
    map_plt = plt_ax.imshow(global_map)
    return map_plt

def draw_robot(robot_plt, robot_state, clr):
    x, y, theta = robot_state

    radius = 10
    length = radius + 10
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    if 'robot' not in robot_plt:
        robot_plt['robot'] = Wedge((x, y), radius, 0, 360, color=clr, alpha=.8)
        plt_ax.add_artist(robot_plt['robot'])
    else:
        robot_plt['robot'].set_center((x, y))
        robot_plt['heading'].set_alpha(.4)    # oldpose

    robot_plt['heading'] = Arrow(x, y, dx, dy, width=10, fc=clr, alpha=.8)
    plt_ax.add_artist(robot_plt['heading'])    # newpose

    return robot_plt

def visualize(params):
    model = pf.PFCell(params).to(params.rank)
    model = load(model, params.file_name)

    # data loader
    composed = transforms.Compose([
                pf.ToTensor(),
    ])
    dataset = pf.House3DTrajDataset(params, params.train_file, transform=composed)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=params.num_workers)

    episode_batch = next(iter(data_loader))

    true_states = episode_batch['true_states'].to(params.rank)
    odometries = episode_batch['odometry'].to(params.rank)
    global_maps = episode_batch['global_map'].to(params.rank)
    observations = episode_batch['observation'].to(params.rank)
    init_particle_states = episode_batch['init_particles'].to(params.rank)

    trajlen = observations.shape[1]
    batch_size, num_particles = episode_batch['init_particles'].shape[:2]
    init_particle_weights = torch.full((batch_size, num_particles), np.log(1.0/num_particles)).to(params.rank)

    # start with episode trajectory state with init particles and weights
    state = init_particle_states, init_particle_weights

    t_particle_states = []
    t_particle_weights = []

    # plot map
    global_map = global_maps[0, 0].detach().cpu().numpy()
    map_plt = draw_map(global_map)

    gt_plt = {}
    est_plt = {}

    # iterate over trajectory_length
    for traj in range(trajlen):
        true_state = true_states[:, traj, :]
        obs = observations[:, traj, :, :, :]
        odom = odometries[:, traj, :]
        inputs = obs, odom, global_maps

        output, state = model(inputs, state)

        t_particle_states.append(output[0].unsqueeze(1))
        t_particle_weights.append(output[1].unsqueeze(1))

        # plot true robot pose
        gt_state = true_state[0].detach().cpu().numpy()
        gt_plt = draw_robot(gt_plt, gt_state, '#064b67')

        # plot est robot pose
        lin_weights = torch.nn.functional.softmax(output[1], dim=-1)
        est_state = torch.sum(torch.mul(output[0][:, :, :], lin_weights[:, :, None]), dim=1)
        est_state = est_state[0].detach().cpu().numpy()
        est_plt = draw_robot(est_plt, est_state, '#78097e')
        # print(np.linalg.norm(gt_state-est_state))


    t_particle_states = torch.cat(t_particle_states, axis=1)
    t_particle_weights = torch.cat(t_particle_weights, axis=1)

    outputs = t_particle_states, t_particle_weights

    loss = loss_fn(outputs[0], outputs[1], true_states, params)
    print(loss)

    return outputs

def loss_fn(particle_states, particle_weights, true_states, params):

    lin_weights = torch.nn.functional.softmax(particle_weights, dim=-1)

    true_coords = true_states[:, :, :2]
    mean_coords = torch.sum(torch.mul(particle_states[:, :, :, :2], lin_weights[:, :, :, None]), dim=2)
    coord_diffs = mean_coords - true_coords

    # convert from pixel coordinates to meters
    coord_diffs *= params.map_pixel_in_meters

    # coordinate loss component: (x-x')^2 + (y-y')^2
    loss_coords = torch.sum(torch.square(coord_diffs), axis=2)

    true_orients = true_states[:, :, 2]
    orient_diffs = particle_states[:, :, :, 2] - true_orients[:, :, None]

    # normalize between [-pi, +pi]
    orient_diffs = pf.normalize(orient_diffs, isTensor=True)

    # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
    loss_orient = torch.square(torch.sum(orient_diffs * lin_weights, axis=2))

    # combine translational and orientation losses
    loss_combined = loss_coords + 0.36 * loss_orient

    losses = {}
    losses['loss_coords'] = torch.mean(loss_coords)
    losses['loss_total'] = torch.mean(loss_combined)

    return losses

def str2bool(v):
    if isinstance(v, bool):
        return v

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--train_file', type=str, default='../data/valid.tfrecords', help='path to the training .tfrecords')
    argparser.add_argument('--type', type=str, default='valid', help='type of .tfrecords')
    argparser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False, help='use resampling during training')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='alpha=0: uniform sampling (ignoring weights) and alpha=1: standard hard sampling (produces zero gradients)')
    argparser.add_argument('--batch_size', type=int, default=4, help='batch size used for training')
    argparser.add_argument('--num_workers', type=int, default=0, help='workers used for data loading')
    argparser.add_argument('--num_particles', type=int, default=30, help='number of particles used for training')
    argparser.add_argument('--transition_std', nargs='*', default=['0.0', '0.0'], help='std for motion model, translation std (meters), rotatation std (radians)')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--n_gpu', type=int, default=-1, help='number of gpus to train')
    argparser.add_argument('--use_lfc', type=str2bool, nargs='?', const=True, default=False, help='use LocallyConnected2d')
    argparser.add_argument('--dataparallel', type=str2bool, nargs='?', const=True, default=False, help='get parallel data training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    params.trajlen = 24
    params.map_pixel_in_meters = 0.02
    params.init_particles_distr = 'gaussian'
    params.init_particles_std = ['0.3', '0.523599']  # 30cm, 30degrees

    # convert multi-input fileds to numpy arrays
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.transition_std = np.array(params.transition_std, np.float32)

    # set common seed value
    # torch.cuda.manual_seed(params.seed)
    # torch.manual_seed(params.seed)
    # np.random.seed(params.seed)
    # random.seed(params.seed)

    print("#########################")
    print(params)
    print("#########################")

    if torch.cuda.is_available():
        params.rank = torch.device('cuda')
    else:
        params.rank = torch.device('cpu')

    fig = plt.figure(figsize=(7, 7))
    plt_ax = fig.add_subplot(111)

    params.file_name = './bckp/jan_23_1/saved_models/pfnet_eps_00049.pth'
    visualize(params)

    plt.show()
