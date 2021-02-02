#!/usr/bin/env python3

from matplotlib.backends.backend_agg import FigureCanvasAgg
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.utils import parse_config
from matplotlib.patches import Wedge
from matplotlib.patches import Arrow
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import numpy as np
import argparse
import igibson
import random
import torch
import scipy
import cv2
import pf
import os

out_folder = './output/'
Path(out_folder).mkdir(parents=True, exist_ok=True)

def draw_map(plt_ax, global_map):
    rows, cols = global_map.shape
    extent = [-cols/2, cols/2, -rows/2, rows/2]
    # map_plt = plt_ax.imshow(global_map)
    map_plt = plt_ax.imshow(global_map, origin='upper', extent=extent)
    return map_plt

def draw_particles(robot_plt, particles, particle_weights, params):
    colors = cm.rainbow(particle_weights)
    size = params.rows
    res = params.res
    positions = particles[:, 0:2] * size * res
    if 'particles' not in robot_plt:
        robot_plt['particles'] = plt.scatter(positions[:, 0], positions[:, 1], s=10, c=colors, alpha=.25)
    else:
        robot_plt['particles'].set_offsets(positions[:, 0:2])
        robot_plt['particles'].set_color(colors)
    return robot_plt

def draw_robot(plt_ax, robot_plt, robot_state, clr, params):
    x, y, theta = robot_state
    size = params.rows
    res = params.res

    x *= size * res
    y *= size * res
    radius = 0.1 * size * res
    length = 2 * radius
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    if 'robot' not in robot_plt:
        robot_plt['robot'] = Wedge((x, y), radius, 0, 360, color=clr, alpha=.9)
        plt_ax.add_artist(robot_plt['robot'])
    else:
        robot_plt['robot'].set_center((x, y))
        # oldpose
        robot_plt['heading'].set_alpha(.0)

    robot_plt['heading'] = Arrow(x, y, dx, dy, width=radius, fc=clr, alpha=.9)
    plt_ax.add_artist(robot_plt['heading'])    # newpose

    return robot_plt

def run_pfnet(params):

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std[0] = particle_std[0] / params.map_pixel_in_meters
    particle_var = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_var[(0, 0, 1),]) # 3x3 matrix

    model = pf.PFCell(params).to(params.rank)
    model = load(model, params.checkpoint)
    model.eval()

    config_filename = os.path.join('.', 'turtlebot_demo.yaml')
    params.config_data = parse_config(config_filename)
    env = iGibsonEnv(config_file=config_filename, mode='gui')

    # # set common seed value
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    env.seed(params.seed)

    with torch.no_grad():

        old_obs = env.reset()
        old_pose = igibson.get_gt_pose(env)
        params.floor_idx = env.task.floor_num
        particles, particle_weights = igibson.random_particles(env, old_pose, params)

        init_particle_states = torch.from_numpy(particles).float().unsqueeze(0).to(params.rank)
        init_particle_weights = torch.from_numpy(particle_weights).float().unsqueeze(0).to(params.rank)
        floor_map = igibson.get_floor_map(params.config_data['scene_id'], params.floor_idx)
        global_maps = torch.from_numpy(floor_map).float().unsqueeze(0).unsqueeze(0).to(params.rank) # [1, 1, 1000, 1000]

        trajlen = params.trajlen
        batch_size, num_particles = init_particle_states.shape[:2]

        # start with episode trajectory state with init particles and weights
        state = init_particle_states, init_particle_weights

        t_particle_states = []
        t_particle_weights = []
        t_true_states = []
        # iterate over trajectory_length
        for traj in tqdm(range(trajlen)):
            action = env.action_space.sample()

            new_obs, _, _, _ = env.step(action)
            new_pose = igibson.get_gt_pose(env)

            obs = np.transpose(old_obs['rgb'], axes=[2, 0, 1])
            obs = torch.from_numpy(obs.copy()).float().unsqueeze(0).to(params.rank) # [1, 3, 56, 56]
            true_state = torch.from_numpy(old_pose.copy()).float().unsqueeze(0).to(params.rank)
            odom = pf.calc_odometry(old_pose, new_pose)
            odom = torch.from_numpy(odom.copy()).float().unsqueeze(0).to(params.rank)   # [1, 3]

            inputs = obs, odom, global_maps

            output, state = model(inputs, state)

            t_particle_states.append(output[0])
            t_particle_weights.append(output[1])

            old_pose = new_pose
            old_obs = new_obs
            t_true_states.append(true_state)

        t_particle_states = torch.stack(t_particle_states, axis=1)    # [1, trajlen, num_particles, 3]
        t_particle_weights = torch.stack(t_particle_weights, axis=1)  # [1, trajlen, num_particles]
        t_true_states = torch.stack(t_true_states, axis=1)   # [1, trajlen, 3]

        eps_outputs = t_particle_states, t_particle_weights, None

        losses = igibson.loss_fn(eps_outputs, t_true_states, params)

        if params.use_loss == 'pfnet_loss':
            t_loss = losses['pfnet_loss']
        elif params.use_loss == 'dpf_loss':
            t_loss = losses['dpf_loss']

        # MSE at end of episode
        lin_weights = torch.nn.functional.softmax(eps_outputs[1][:, -1, :], dim=-1)
        l_mean_state = torch.sum(torch.mul(eps_outputs[0][:, -1, :, :], lin_weights[:, :, None]), dim=1)
        l_true_state = t_true_states[:, -1, :]
        rescales = [params.map_pixel_in_meters, params.map_pixel_in_meters, 0.36]
        t_mse_last = torch.mean(igibson.compute_sq_distance(l_mean_state, l_true_state, rescales)).item()

        print(f't_loss: {t_loss.item():03.3f} and t_mse_last: {t_mse_last:03.3f}')
        print(f'gt pose:  {l_true_state[0].detach().cpu().numpy()}\nest pose: {l_mean_state[0].detach().cpu().numpy()} in (mts, radians)')

        data = {
            'global_maps': global_maps.detach().cpu().numpy(),
            'particle_states': t_particle_states.detach().cpu().numpy(),
            'particle_weights': t_particle_weights.detach().cpu().numpy(),
            'true_states': t_true_states.detach().cpu().numpy(),
        }

        with open(out_folder+'data.pkl','wb') as f:
            pkl.dump(data, f)

def visualize(params):
    with open(out_folder+'data.pkl','rb') as f:
        data = pkl.load(f)

    global_maps = data['global_maps']
    particle_states = data['particle_states']
    particle_weights = data['particle_weights']
    true_states = data['true_states']

    fig = plt.figure(figsize=(7, 7), dpi=300)
    plt_ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    # plot map
    global_map = global_maps[0][0]
    map_plt = draw_map(plt_ax, global_map)
    params.rows = global_map.shape[0]
    params.res = 0.1

    trajlen = params.trajlen

    gt_plt = {}
    est_plt = {}

    images = []
    for traj in range(trajlen):
        true_state = true_states[:, traj, :]
        particle_state = particle_states[:, traj, :, :]
        particle_weight = particle_weights[:, traj, :]

        # plot true robot pose
        gt_plt = draw_robot(plt_ax, gt_plt, true_state[0], '#7B241C', params)

        # plot est robot pose
        lin_weights = scipy.special.softmax(particle_weight, axis=-1)
        est_state = np.sum(particle_state[:, :, :] * lin_weights[:, :, None], axis=1)
        est_plt = draw_robot(plt_ax, est_plt, est_state[0], '#515A5A', params)
        # print(np.linalg.norm(gt_state-est_state))

        # plot est pose particles
        draw_particles(est_plt, particle_state[0], lin_weights[0], params)

        plt_ax.legend([gt_plt['robot'], est_plt['robot']], ["gt_pose", "est_pose"])

        canvas.draw()
        img = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    size = (images[0].shape[0], images[0].shape[1])
    out = cv2.VideoWriter(out_folder + 'result.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for i in range(len(images)):
        out.write(images[i])
        cv2.imwrite(out_folder + f'result_img_{i}.png', images[i])
    out.release()

    # plt.show()

def load(model, file_name):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['pf_cell'])
    print('=====> checkpoint loaded')
    return model

def str2bool(v):
    if isinstance(v, bool):
        return v

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--checkpoint', type=str, default='./saved_models/pfnet_eps_00000.pth', help='load pretrained model *.pth checkpoint')
    argparser.add_argument('--num_particles', type=int, default=30, help='number of particles used for training')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False, help='use resampling during training')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='alpha=0: uniform sampling (ignoring weights) and alpha=1: standard hard sampling (produces zero gradients)')
    argparser.add_argument('--resample_threshold', type=float, default=0.5, help='resample_threshold=1 means resample every step and resample_threshold=0.01 means almost never')
    argparser.add_argument('--trajlen', type=int, default=25, help='trajectory length to train [max 100]')
    argparser.add_argument('--init_particles_distr', type=str, default='gaussian', help='options: [gaussian, uniform]')
    argparser.add_argument('--init_particles_std', nargs='*', default=['0.3', '0.523599'], help='std for init distribution, position std (meters), rotatation std (radians)')
    argparser.add_argument('--transition_std', nargs='*', default=['0.0', '0.0'], help='std for motion model, translation std (meters), rotatation std (radians)')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--use_gpus', type=str, nargs='*', default=None, help='gpu(s) to train')
    argparser.add_argument('--use_loss', type=str, default='pfnet_loss', help='options: [pfnet_loss, dpf_loss]')
    argparser.add_argument('--use_lfc', type=str2bool, nargs='?', const=True, default=False, help='use LocallyConnected2d')
    argparser.add_argument('--dataparallel', type=str2bool, nargs='?', const=True, default=False, help='get parallel data training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    params.map_pixel_in_meters = 1.0    # hardcoded
    assert 0.0 < params.resample_threshold <= 1.0

    # convert multi-input fileds to numpy arrays
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.transition_std = np.array(params.transition_std, np.float32)

    print("#########################")
    print(params)
    print("#########################")

    use_cuda = params.use_gpus is not None and torch.cuda.is_available()
    if use_cuda:
        params.rank = torch.device('cuda:0')
    else:
        params.rank = torch.device('cpu')

    run_pfnet(params)
    visualize(params)
