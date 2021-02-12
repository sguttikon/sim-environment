#!/usr/bin/env python3

from torch.nn.parallel import DistributedDataParallel as DDP
from gibson2.utils.assets_utils import get_scene_path
from torch.utils.tensorboard import SummaryWriter
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.utils import parse_config
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import gibson2
import random
import torch
import cv2
import os
import pf

np.set_printoptions(precision=5, suppress=True)

# logger
writer = SummaryWriter()

def get_gt_pose(env, idx=0):
    robot = env.robots[idx] #hardcoded

    position = robot.get_position()
    euler_orientation = pf.normalize(robot.get_rpy())
    gt_pose = np.array([
        position[0],        # x
        position[1],        # y
        euler_orientation[2] # yaw
    ])
    return gt_pose

def get_floor_map(scene_id, floor_idx):
    filename = os.path.join(get_scene_path(scene_id), f'floor_{floor_idx}.png')

    floor_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    floor_map = cv2.flip(floor_map, 0)  #HACK flip image for this scene

    return floor_map    # [WIDTH, HEIGHT]

def collect_batch_data(env, params):
    batch_samples = {}

    old_obs = env.reset()
    old_pose = get_gt_pose(env)
    params.floor_idx = env.task.floor_num
    particles, particle_weights = random_particles(env, old_pose, params)

    batch_samples['init_particles'] = torch.from_numpy(particles).float().unsqueeze(0).to(params.rank)
    batch_samples['init_particle_weights'] = torch.from_numpy(particle_weights).float().unsqueeze(0).to(params.rank)
    floor_map = get_floor_map(params.config_data['scene_id'], params.floor_idx)
    batch_samples['global_map'] = torch.from_numpy(floor_map).float().unsqueeze(0).unsqueeze(0).to(params.rank)
    observations = []
    odometrys = []
    true_states = []
    for i in range(params.trajlen):
        observations.append(np.transpose(old_obs['rgb'], axes=[2, 0, 1]))
        true_states.append(old_pose)

        if i >= params.trajlen-2:
            action = [0.0, 0.0] #HACK do nothing at end of trajectory
        else:
            action = env.action_space.sample()

        new_obs, _, _, _ = env.step(action)
        new_pose = get_gt_pose(env)

        odom = pf.calc_odometry(old_pose, new_pose)
        odometrys.append(odom)

        old_pose = new_pose
        old_obs = new_obs

    batch_samples['odometry'] = torch.from_numpy(np.array(odometrys)).float().unsqueeze(0).to(params.rank)
    batch_samples['observation'] = torch.from_numpy(np.array(observations)).float().unsqueeze(0).to(params.rank)
    batch_samples['true_states'] = torch.from_numpy(np.array(true_states)).float().unsqueeze(0).to(params.rank)

    return batch_samples

def run_episode(model, episode_batch):
    odometries = episode_batch['odometry']
    global_maps = episode_batch['global_map']
    observations = episode_batch['observation']
    particle_states = episode_batch['particle_states']
    particle_weights = episode_batch['particle_weights']

    state = particle_states, particle_weights

    t_particle_states = []
    t_particle_weights = []
    t_n_eff = []

    # iterate over shoter segment_length
    seglen = observations.shape[1]
    for seg in range(seglen):
        obs = observations[:, seg, :, :, :]
        odom = odometries[:, seg, :]
        inputs = obs, odom, global_maps

        outputs, state = model(inputs, state)

        t_particle_states.append(outputs[0].unsqueeze(1))
        t_particle_weights.append(outputs[1].unsqueeze(1))
        t_n_eff.append(outputs[2])

    t_particle_states = torch.cat(t_particle_states, axis=1)
    t_particle_weights = torch.cat(t_particle_weights, axis=1)

    eps_outputs = t_particle_states, t_particle_weights, t_n_eff
    eps_state = state

    return eps_outputs, eps_state

def run_pfnet(rank, params):
    print(f'Training on GPepisode_batchU rank {rank}.')
    Path("train_models").mkdir(parents=True, exist_ok=True)

    # create model and move it to GPU with id rank
    if rank != torch.device('cpu'):
        setup(rank, params.world_size)

    model = pf.PFCell(params).to(rank)
    params.rank = rank

    if rank != torch.device('cpu'):
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # define optimizer
    model_params =  list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=2e-4, weight_decay=0.01)

    config_filename = os.path.join('.', 'turtlebot_demo.yaml')
    params.config_data = parse_config(config_filename)
    env = iGibsonEnv(config_file=config_filename, mode='headless')
    env.seed(params.seed)

    trajlen, seglen, num_particles = params.trajlen, params.seglen, params.num_particles
    assert trajlen % seglen == 0
    num_segments = trajlen // seglen
    # iterate over num_train_epochs
    for epoch in range(params.num_epochs):
        b_loss = []
        b_mse_last = []
        model.train()
        # iterate over num_batches
        for batch_idx in tqdm(range(params.num_batches)):
            batch_samples = collect_batch_data(env, params)

            # sanity check
            assert list(batch_samples['true_states'].shape)[1:] == [trajlen, 3]
            assert list(batch_samples['odometry'].shape)[1:] == [trajlen, 3]
            assert list(batch_samples['global_map'].shape)[1:] == [1, 1000, 1000]
            assert list(batch_samples['observation'].shape)[1:] == [trajlen, 3, 56, 56]
            assert list(batch_samples['init_particles'].shape)[1:] == [num_particles, 3]
            assert list(batch_samples['init_particle_weights'].shape)[1:] == [num_particles]

            # traj loss is sum of multiple seg loss
            t_loss = 0

            # start each episode trajectory with init particles and weights state
            episode_batch = {}
            episode_batch['particle_states'] = batch_samples['init_particles']
            episode_batch['particle_weights'] = batch_samples['init_particle_weights']
            episode_batch['global_map'] = batch_samples['global_map']
            # iterate over shorter segments
            for i in range(num_segments):
                # use latest inputs per segment
                episode_batch['odometry'] = batch_samples['odometry'][:, i*seglen : (i+1)*seglen, :]
                episode_batch['observation'] = batch_samples['observation'][:, i*seglen : (i+1)*seglen, :]

                # sanity check
                assert list(episode_batch['odometry'].shape)[1:] == [seglen, 3]
                assert list(episode_batch['observation'].shape)[1:] == [seglen, 3, 56, 56]
                eps_outputs, eps_state = run_episode(model, episode_batch)

                # [batch_size, trajlen, [num_particles], ..]
                labels = batch_samples['true_states'][:, i*seglen : (i+1)*seglen, :]
                seg_losses = loss_fn(eps_outputs, labels, params)

                if params.use_loss == 'pfnet_loss':
                    loss = seg_losses['pfnet_loss']
                elif params.use_loss == 'dpf_loss':
                    loss = seg_losses['dpf_loss']

                loss.backward()

                # visualize gradient flow
                # pf.plot_grad_flow(pf_cell.named_parameters())

                # update parameters based on gradients
                optimizer.step()

                # clear gradients
                optimizer.zero_grad()

                # continue with previous episode state for same trajectory
                # detach() since backprop only to current segment
                episode_batch['particle_states'] = eps_state[0].detach()
                episode_batch['particle_weights'] = eps_state[1].detach()

                t_loss += loss.item()
            t_loss /= num_segments  # mean

            # MSE at end of episode
            lin_weights = torch.nn.functional.softmax(episode_batch['particle_weights'], dim=-1)
            l_mean_state = torch.sum(torch.mul(episode_batch['particle_states'][:, :, :], lin_weights[:, :, None]), dim=1)
            l_true_state = labels[:, -1, :]
            rescales = [params.map_pixel_in_meters, params.map_pixel_in_meters, 0.36]
            t_mse_last = torch.mean(compute_sq_distance(l_mean_state, l_true_state, rescales)).item()

            # log per epoch batch stats (only for gpu:0 or cpu)
            if rank == torch.device('cpu') or rank == 0:
                b_loss.append(t_loss)
                b_mse_last.append(t_mse_last)

                # print('train epoch: {0:05d}, batch: {1:05d}, b_loss: {2:03.3f}, mse_last: {3:03.3f}'.format(epoch, batch_idx, t_loss, t_mse_last))
                writer.add_scalars(f'epoch-{epoch:03d}_train_stats', {
                    't_loss': t_loss,
                    't_mse_last': t_mse_last,
                }, batch_idx)

        # log per epoch mean stats (only for gpu:0 or cpu)
        if rank == torch.device('cpu') or rank == 0:
            print(f'epoch: {epoch:05d}, mean_loss: {np.mean(b_loss):03.3f}, mean_mse_last: {np.mean(b_mse_last):03.3f}')
            writer.add_scalars('train_stats', {
                    'mean_loss': np.mean(b_loss),
                    'mean_mse_last': np.mean(b_mse_last),
            }, epoch)

        # save (only for gpu:0 or cpu)
        file_name = 'train_models/' + 'pfnet_train_eps_{0:05d}.pth'.format(epoch)
        if rank == torch.device('cpu'):
            save(model, file_name)
        elif rank == 0:
            save(model.module, file_name)

    if rank != torch.device('cpu'):
        cleanup()

    print('training finished')

    env.close()

def save(model, file_name):
    torch.save({
        'pf_cell': model.state_dict(),
    }, file_name)

def load(model, file_name):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['pf_cell'])
    return model

def loss_fn(outputs, true_states, params):
    particle_states, particle_weights, _ = outputs

    lin_weights = torch.nn.functional.softmax(particle_weights, dim=-1)

    true_coords = true_states[:, :, :2]
    mean_coords = torch.sum(torch.mul(particle_states[:, :, :, :2], lin_weights[:, :, :, None]), dim=2)
    coord_diffs = mean_coords - true_coords

    # convert from pixel coordinates to meters
    coord_diffs *= params.map_pixel_in_meters

    # coordinate loss component: (x-x')^2 + (y-y')^2
    loss_coords = torch.sum(torch.square(coord_diffs), axis=2)

    # normalize between [-pi, +pi]
    true_orients = true_states[:, :, 2]
    orient_diffs = pf.normalize(particle_states[:, :, :, 2], isTensor=True) - \
                    pf.normalize(true_orients[:, :, None], isTensor=True)

    # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
    loss_orient = torch.square(torch.sum(orient_diffs * lin_weights, axis=2))

    # combine translational and orientation losses
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
    losses['loss_coords'] = loss_coords

    return losses

def compute_sq_distance(a, b, rescales):
    result = 0.0
    for i in range(a.shape[-1]):
        diff = a[..., i] - b[..., i]
        if i == 2:
            diff = pf.normalize(diff, isTensor=True)
        result += (diff * rescales[i]) ** 2
    return result

def random_particles(env, init_state, params):
    distr = params.init_particles_distr
    assert distr in ["gaussian", "uniform"]
    num_particles = params.num_particles
    init_cov = params.init_particles_cov

    if distr == 'gaussian':
        # sample offset from the gaussian
        center = np.random.multivariate_normal(mean=init_state, cov=init_cov)

        # sample particles from gaussian centered around the offset
        particles = np.random.multivariate_normal(mean=center, cov=init_cov, size=num_particles)
    elif distr == 'uniform':
        sample_i = 0
        rnd_particles = []
        while sample_i < num_particles:
            _, initial_pos = env.scene.get_random_point(floor=params.floor_idx)
            initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            rnd_pose = [initial_pos[0], initial_pos[1], initial_orn[2]]
            rnd_particles.append(rnd_pose)
            sample_i += 1
        particles = np.array(rnd_particles)

    particle_weights = np.full(num_particles, np.log(1.0/num_particles))
    particles[:, 2] = pf.normalize(particles[:, 2])
    return particles, particle_weights

def draw_map(global_map):
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    #dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size,rank=rank)

def cleanup():
    dist.destroy_process_group()

def str2bool(v):
    if isinstance(v, bool):
        return v

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num_particles', type=int, default=30, help='number of particles used for training')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False, help='use resampling during training')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='alpha=0: uniform sampling (ignoring weights) and alpha=1: standard hard sampling (produces zero gradients)')
    argparser.add_argument('--resample_threshold', type=float, default=0.5, help='resample_threshold=1 means resample every step and resample_threshold=0.01 means almost never')
    argparser.add_argument('--trajlen', type=int, default=25, help='trajectory length to train [max 100]')
    argparser.add_argument('--seglen', type=int, default=5, help='short train segement length to train [max 100]')
    argparser.add_argument('--init_particles_distr', type=str, default='gaussian', help='options: [gaussian, uniform]')
    argparser.add_argument('--init_particles_std', nargs='*', default=['0.3', '0.523599'], help='std for init distribution, position std (meters), rotatation std (radians)')
    argparser.add_argument('--transition_std', nargs='*', default=['0.0', '0.0'], help='std for motion model, translation std (meters), rotatation std (radians)')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train')
    argparser.add_argument('--num_batches', type=int, default=500, help='number of episode batched per epochs to train')
    argparser.add_argument('--use_gpus', type=str, nargs='*', default=None, help='gpu(s) to train')
    argparser.add_argument('--use_loss', type=str, default='pfnet_loss', help='options: [pfnet_loss, dpf_loss]')
    argparser.add_argument('--use_lfc', type=str2bool, nargs='?', const=True, default=False, help='use LocallyConnected2d')
    argparser.add_argument('--dataparallel', type=str2bool, nargs='?', const=True, default=False, help='get parallel data training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    # convert multi-input fileds to numpy arrays
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.transition_std = np.array(params.transition_std, np.float32)

    params.map_pixel_in_meters = 1.0    # hardcoded

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std[0] = particle_std[0] / params.map_pixel_in_meters
    particle_var = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_var[(0, 0, 1),]) # 3x3 matrix

    # set common seed value
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    use_cuda = params.use_gpus is not None and torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    if not use_cuda:
        print('switching to CPU')
        params.n_gpu = -1
    elif n_gpus < len(params.use_gpus):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(elem) for elem in range(n_gpus)])
        params.n_gpu = n_gpus
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([elem for elem in params.use_gpus])
        params.n_gpu = len(params.use_gpus)
    params.world_size = params.n_gpu

    print("#########################")
    print(params)
    print("#########################")

    if use_cuda:
        # multi-process run
        print(f'Avaialble GPUs - cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}')
        mp.spawn(run_pfnet, nprocs=params.n_gpu, args=(params,))
    else:
        # normal run
        rank = torch.device('cpu')
        run_pfnet(rank, params)
