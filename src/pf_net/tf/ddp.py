#!/usr/bin/env python3

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import numpy as np
import argparse
import random
import torch
import pf
import os

np.set_printoptions(precision=5, suppress=True)
Path("train_models").mkdir(parents=True, exist_ok=True)
Path("eval_models").mkdir(parents=True, exist_ok=True)

# logger
writer = SummaryWriter()

def run_episode(model, episode_batch):
    odometries = episode_batch['odometry']
    global_maps = episode_batch['global_map']
    observations = episode_batch['observation']
    particle_states = episode_batch['particle_states']
    particle_weights = episode_batch['particle_weights']

    state = particle_states, particle_weights

    t_particle_states = []
    t_particle_weights = []

    # iterate over shoter segment_length
    seglen = observations.shape[1]
    for seg in range(seglen):
        obs = observations[:, seg, :, :, :]
        odom = odometries[:, seg, :]
        inputs = obs, odom, global_maps

        outputs, state = model(inputs, state)

        t_particle_states.append(outputs[0].unsqueeze(1))
        t_particle_weights.append(outputs[1].unsqueeze(1))

    t_particle_states = torch.cat(t_particle_states, axis=1)
    t_particle_weights = torch.cat(t_particle_weights, axis=1)

    eps_outputs = t_particle_states, t_particle_weights
    eps_state = state

    return eps_outputs, eps_state

def run_training(rank, params):
    print(f"Training on GPU rank {rank}.")

    # create model and move it to GPU with id rank
    if rank != torch.device('cpu'):
        setup(rank, params.world_size)
        torch.cuda.set_device(rank)
    model = pf.PFCell(params).to(rank)
    params.rank = rank

    # define optimizer
    model_params =  list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=2e-4, weight_decay=0.01)

    if rank != torch.device('cpu'):
        # wrap the model
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # data loader
    composed = transforms.Compose([
                pf.ToTensor(),
    ])
    train_dataset = pf.House3DTrajDataset(params, 'train', transform=composed)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    print(f'training file: {params.train_file} has {len(train_dataset)} records')

    valid_dataset = pf.House3DTrajDataset(params, 'valid', transform=composed)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    print(f'validation file: {params.valid_file} has {len(valid_dataset)} records')

    # train_dataset = TensorDataset(config.masked_sentences, config.original_sentences)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=config.world_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,sampler=train_sampler, shuffle=False)

    trajlen, seglen = params.trajlen, params.seglen
    assert trajlen % seglen == 0
    num_segments = trajlen // seglen
    # iterate over num_train_epochs
    for epoch in range(params.num_train_epochs):
        b_loss = []
        b_mse_last = []
        model.train()
        # iterate over num_batches
        for batch_idx, batch_samples in enumerate(train_loader):
            batch_samples['true_states'] = batch_samples['true_states'].to(params.rank)
            batch_samples['odometry'] = batch_samples['odometry'].to(params.rank)
            batch_samples['global_map'] = batch_samples['global_map'].to(params.rank)
            batch_samples['observation'] = batch_samples['observation'].to(params.rank)

            batch_size, num_particles = batch_samples['init_particles'].shape[:2]
            batch_samples['init_particles'] = batch_samples['init_particles'].to(params.rank)
            batch_samples['init_particle_weights'] = torch.full((batch_size, num_particles), np.log(1.0/num_particles)).to(params.rank)

            # sanity check
            assert list(batch_samples['true_states'].shape)[1:] == [trajlen, 3]
            assert list(batch_samples['odometry'].shape)[1:] == [trajlen, 3]
            # assert list(batch_samples['global_map'].shape)[1:] == [1, 3000, 3000]
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

                # cleat gradients
                optimizer.zero_grad()

                # continue with previous episode state for same trajectory
                # detach() since backprop only to current segment
                episode_batch['particle_states'] = eps_state[0].detach()
                episode_batch['particle_weights'] = eps_state[1].detach()

                t_loss += loss.item()

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

                print('train epoch: {0:05d}, batch: {1:05d}, b_loss: {2:03.3f}, mse_last: {3:03.3f}'.format(epoch, batch_idx, t_loss, t_mse_last))
                writer.add_scalars('epoch-{0:03d}_train_stats'.format(epoch), {
                    't_loss': t_loss,
                    't_mse_last': t_mse_last
                }, batch_idx)

        # log per epoch mean stats (only for gpu:0 or cpu)
        if rank == torch.device('cpu') or rank == 0:
            print('train epoch: {0:05d}, mean_loss: {1:03.3f}, mean_mse_last: {2:03.3f}'.format(epoch, np.mean(b_loss), np.mean(b_mse_last)))
            writer.add_scalars('train_stats', {
                    'mean_loss': np.mean(b_loss_total),
                    'mean_mse_last': np.mean(b_mse_last),
            }, epoch)

        # validate
        if epoch%5 == 0:
            run_validation(model, valid_loader, params)

        # save (only for gpu:0 or cpu)
        file_name = 'train_models/' + 'pfnet_train_eps_{0:05d}.pth'.format(epoch)
        if rank == torch.device('cpu'):
            save(model, file_name)
        elif rank == 0:
            save(model.module, file_name)
    cleanup()

    print('training finished')

def run_validation(model, valid_loader, params):

    # iterate over num_valid_epochs
    for epoch in range(params.num_valid_epochs):
        b_loss_total = []
        b_loss_coords = []
        model.eval()
        # iterate over num_batches
        for batch_idx, batch_samples in enumerate(valid_loader):
            episode_batch = {}
            episode_batch['true_states'] = batch_samples['true_states'].to(params.rank)
            episode_batch['odometry'] = batch_samples['odometry'].to(params.rank)
            episode_batch['global_map'] = batch_samples['global_map'].to(params.rank)
            episode_batch['observation'] = batch_samples['observation'].to(params.rank)
            episode_batch['particle_states'] = batch_samples['init_particles'].to(params.rank)

            batch_size, num_particles = episode_batch['particle_states'].shape[:2]
            episode_batch['particle_weights'] = torch.full((batch_size, num_particles), np.log(1.0/num_particles)).to(params.rank)

            #HACK validating on entire trajectory instead of running shorter segments
            eps_outputs, _ = run_episode(model, episode_batch)

            labels = episode_batch['true_states']
            losses = loss_fn(eps_outputs, labels, params)

            # log per epoch batch stats (only for gpu:0 or cpu)
            if rank == torch.device('cpu') or rank == 0:
                loss_total = losses['loss_total'].item()
                loss_coords = losses['loss_coords'].item()

                b_loss_total.append(loss_total)
                b_loss_coords.append(loss_coords)

                print(' eval epoch: {0:05d}, batch: {1:05d}, b_loss_coords: {2:03.3f}, b_loss_total: {3:03.3f}'.format(epoch, batch_idx, loss_coords, loss_total))
                writer.add_scalars('epoch-{0:03d}_eval_stats'.format(epoch), {
                    'b_total_loss': loss_total,
                    'b_coords_loss': loss_coords
                }, batch_idx)

        # log per epoch mean stats (only for gpu:0 or cpu)
        if rank == torch.device('cpu') or rank == 0:
            print(' eval epoch: {0:05d}, mean_loss_coords: {1:03.3f}, mean_loss_total: {2:03.3f}'.format(epoch, np.mean(b_loss_coords), np.mean(b_loss_total)))
            writer.add_scalars('train_stats', {
                    'mean_total_loss': np.mean(b_loss_total),
                    'mean_coords_loss': np.mean(b_loss_coords)
            }, epoch)

        # save (only for gpu:0 or cpu)
        file_name = 'eval_models/' + 'pfnet_eval_eps_{0:03.3f}.pth'.format(np.mean(b_loss_total))
        if rank == torch.device('cpu'):
            save(model, file_name)
        elif rank == 0:
            save(model.module, file_name)

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

def save(model, file_name):
    torch.save({
        'pf_cell': model.state_dict(),
    }, file_name)

def load(model, file_name):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['pf_cell'])
    return model

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

# reference: https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--train_file', type=str, default='../data/valid.tfrecords', help='path to the training .tfrecords')
    argparser.add_argument('--valid_file', type=str, default='../data/test.tfrecords', help='path to the validating .tfrecords')
    argparser.add_argument('--num_train_epochs', type=int, default=20, help='number of epochs to train')
    argparser.add_argument('--num_valid_epochs', type=int, default=1, help='number of epochs to eval')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False, help='use resampling during training')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=0.5, help='alpha=0: uniform sampling (ignoring weights) and alpha=1: standard hard sampling (produces zero gradients)')
    argparser.add_argument('--batch_size', type=int, default=4, help='batch size used for training')
    argparser.add_argument('--num_workers', type=int, default=0, help='workers used for data loading')
    argparser.add_argument('--num_particles', type=int, default=30, help='number of particles used for training')
    argparser.add_argument('--trajlen', type=int, default=25, help='trajectory length to train [max 100]')
    argparser.add_argument('--seglen', type=int, default=5, help='short train segement length to train [max 100]')
    argparser.add_argument('--init_particles_distr', type=str, default='gaussian', help='options: [gaussian, one-room]')
    argparser.add_argument('--init_particles_std', nargs='*', default=['0.3', '0.523599'], help='std for init distribution, position std (meters), rotatation std (radians)')
    argparser.add_argument('--transition_std', nargs='*', default=['0.0', '0.0'], help='std for motion model, translation std (meters), rotatation std (radians)')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--global_map_size', nargs='*', default=(3500, 3500), help='shape of local map')
    argparser.add_argument('--use_gpus', type=str, nargs='*', default=None, help='gpu(s) to train')
    argparser.add_argument('--use_loss', type=str, default='pfnet_loss', help='options: [pfnet_loss, dpf_loss]')
    argparser.add_argument('--use_lfc', type=str2bool, nargs='?', const=True, default=False, help='use LocallyConnected2d')
    argparser.add_argument('--dataparallel', type=str2bool, nargs='?', const=True, default=False, help='get parallel data training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    params.map_pixel_in_meters = 0.02

    # convert multi-input fileds to numpy arrays
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.transition_std = np.array(params.transition_std, np.float32)

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
        print(f'Available GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        mp.spawn(run_training, nprocs=params.n_gpu, args=(params,))
    else:
        # normal run
        rank = torch.device('cpu')
        run_training(rank, params)
