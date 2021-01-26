#!/usr/bin/env python3

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from matplotlib.patches import Wedge
from matplotlib.patches import Arrow
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import numpy as np
import argparse
import random
import torch
import cv2
import pf

np.set_printoptions(precision=3, suppress=True)

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
        robot_plt['robot'] = Wedge((x, y), radius, 0, 360, color=clr, alpha=.9)
        plt_ax.add_artist(robot_plt['robot'])
    else:
        robot_plt['robot'].set_center((x, y))
        # oldpose
        robot_plt['heading'].set_alpha(.0)

    robot_plt['heading'] = Arrow(x, y, dx, dy, width=10, fc=clr, alpha=.9)
    plt_ax.add_artist(robot_plt['heading'])    # newpose

    return robot_plt

def draw_particles(robot_plt, particles, particle_weights):
    colors = cm.rainbow(particle_weights)
    positions = particles[:, 0:2]
    if 'particles' not in robot_plt:
        robot_plt['particles'] = plt.scatter(positions[:, 0], positions[:, 1], s=10, c=colors, alpha=.25)
    else:
        robot_plt['particles'].set_offsets(positions[:, 0:2])
        robot_plt['particles'].set_color(colors)
    return robot_plt

def visualize(params):
    model = pf.PFCell(params).to(params.rank)
    model = load(model, params.checkpoint)
    model.eval()

    # data loader
    composed = transforms.Compose([
                pf.ToTensor(),
    ])
    dataset = pf.House3DTrajDataset(params, transform=composed)
    data_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    print(f'validation file: {params.data_file} has {len(dataset)} records')

    episode_batch = next(iter(data_loader))

    true_states = episode_batch['true_states'].to(params.rank)
    odometries = episode_batch['odometry'].to(params.rank)
    global_maps = episode_batch['global_map'].to(params.rank)
    org_map_shape = episode_batch['org_map_shape']
    observations = episode_batch['observation'].to(params.rank)
    init_particle_states = episode_batch['init_particles'].to(params.rank)

    trajlen = params.trajlen
    batch_size, num_particles = init_particle_states.shape[:2]
    init_particle_weights =  torch.full((batch_size, num_particles), np.log(1.0/num_particles)).to(params.rank)

    # start with episode trajectory state with init particles and weights
    state = init_particle_states, init_particle_weights

    t_particle_states = []
    t_particle_weights = []

    # plot map
    shape = org_map_shape[0]
    org_global_maps = global_maps[0][:shape[2], :shape[0], :shape[1]]
    global_map = org_global_maps[0].detach().cpu().numpy()
    map_plt = draw_map(global_map)

    gt_plt = {}
    est_plt = {}

    # plot init est pose particles
    particles = state[0][0].detach().cpu().numpy()
    lin_weights = torch.nn.functional.softmax(state[1], dim=-1)
    particle_weights = lin_weights[0].detach().cpu().numpy()
    draw_particles(est_plt, particles, particle_weights)

    images = []
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
        gt_plt = draw_robot(gt_plt, gt_state, '#7B241C')

        # plot est robot pose
        lin_weights = torch.nn.functional.softmax(output[1], dim=-1)
        est_state = torch.sum(torch.mul(output[0][:, :, :], lin_weights[:, :, None]), dim=1)
        est_state = est_state[0].detach().cpu().numpy()
        est_plt = draw_robot(est_plt, est_state, '#515A5A')
        # print(np.linalg.norm(gt_state-est_state))

        # plot est pose particles
        particles = output[0][0].detach().cpu().numpy()
        particle_weights = lin_weights[0].detach().cpu().numpy()
        draw_particles(est_plt, particles, particle_weights)

        plt_ax.legend([gt_plt['robot'], est_plt['robot']], ["gt_pose", "est_pose"])

        canvas.draw()
        img = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)


    t_particle_states = torch.cat(t_particle_states, axis=1)
    t_particle_weights = torch.cat(t_particle_weights, axis=1)

    eps_outputs = t_particle_states, t_particle_weights

    losses = loss_fn(eps_outputs, true_states, params)

    if params.use_loss == 'pfnet_loss':
        t_loss = losses['pfnet_loss']
    elif params.use_loss == 'dpf_loss':
        t_loss = losses['dpf_loss']

    # MSE at end of episode
    lin_weights = torch.nn.functional.softmax(eps_outputs[1][:, -1, :], dim=-1)
    l_mean_state = torch.sum(torch.mul(eps_outputs[0][:, -1, :, :], lin_weights[:, :, None]), dim=1)
    l_true_state = true_states[:, -1, :]
    rescales = [params.map_pixel_in_meters, params.map_pixel_in_meters, 0.36]
    t_mse_last = torch.mean(compute_sq_distance(l_mean_state, l_true_state, rescales)).item()

    print(f't_loss: {t_loss.item():03.3f} and t_mse_last: {t_mse_last:03.3f}')

    gt_state[:2] *= params.map_pixel_in_meters
    est_state[:2] *= params.map_pixel_in_meters
    print(f'gt pose:  {gt_state}\nest pose: {est_state} in (mts, radians)')

    return images

def loss_fn(outputs, true_states, params):
    particle_states, particle_weights = outputs

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
    pfnet_loss = loss_coords + 0.36 * loss_orient

    # negative log likelihood loss
    rescales = [params.map_pixel_in_meters, params.map_pixel_in_meters, 0.36]
    sq_distance = compute_sq_distance(particle_states[:, :, :, :], true_states[:, :, None, :], rescales)
    std = 0.01
    particle_std=0.2
    log_likelihood = torch.sum(lin_weights - np.log(np.sqrt(2 * np.pi * std ** 2)) - (sq_distance / (2.0 * particle_std ** 2)), axis=2)
    dpf_loss = -log_likelihood

    losses = {}
    losses['pfnet_loss'] = torch.mean(pfnet_loss)
    losses['dpf_loss'] = torch.mean(dpf_loss)

    return losses

def compute_sq_distance(a, b, rescales):
    result = 0.0
    for i in range(a.shape[-1]):
        diff = a[..., i] - b[..., i]
        if i == 2:
            diff = pf.normalize(diff, isTensor=True)
        result += (diff * rescales[i]) ** 2
    return result

def str2bool(v):
    if isinstance(v, bool):
        return v

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_file', type=str, default='../data/test.tfrecords', help='path to the training/validation .tfrecords')
    argparser.add_argument('--eval', type=str2bool, nargs='?', const=True, default=True, help='validate network')
    argparser.add_argument('--checkpoint', type=str, default='./saved_models/pfnet_eps_00000.pth', help='load pretrained model *.pth checkpoint')
    argparser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train/eval')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False, help='use resampling during training')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='alpha=0: uniform sampling (ignoring weights) and alpha=1: standard hard sampling (produces zero gradients)')
    argparser.add_argument('--resample_threshold', type=float, default=0.5, help='resample_threshold=1 means resample every step and resample_threshold=0.01 means almost never')
    argparser.add_argument('--batch_size', type=int, default=1, help='batch size used for training')
    argparser.add_argument('--num_workers', type=int, default=0, help='workers used for data loading')
    argparser.add_argument('--num_particles', type=int, default=200, help='number of particles used for training')
    argparser.add_argument('--trajlen', type=int, default=25, help='trajectory length to train [max 100]')
    argparser.add_argument('--init_particles_distr', type=str, default='gaussian', help='options: [gaussian, one-room]')
    argparser.add_argument('--init_particles_std', nargs='*', default=['0.3', '0.523599'], help='std for init distribution, position std (meters), rotatation std (radians)')
    argparser.add_argument('--transition_std', nargs='*', default=['0.0', '0.0'], help='std for motion model, translation std (meters), rotatation std (radians)')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--global_map_size', nargs='*', default=(3500, 3500), help='shape of local map')
    # argparser.add_argument('--use_gpus', type=str, nargs='*', default=None, help='gpu(s) to train')
    argparser.add_argument('--use_loss', type=str, default='pfnet_loss', help='options: [pfnet_loss, dpf_loss]')
    argparser.add_argument('--use_lfc', type=str2bool, nargs='?', const=True, default=False, help='use LocallyConnected2d')
    argparser.add_argument('--dataparallel', type=str2bool, nargs='?', const=True, default=False, help='get parallel data training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    params.map_pixel_in_meters = 0.02
    assert 0.0 < params.resample_threshold <= 1.0

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

    fig = plt.figure(figsize=(7, 7), dpi=300)
    plt_ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    images = visualize(params)

    size = (images[0].shape[0], images[0].shape[1])
    folder = './output/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(folder + 'result.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for i in range(len(images)):
        out.write(images[i])
        cv2.imwrite(folder + f'result_img_{i}.png', images[i])
    out.release()

    #plt.show()
