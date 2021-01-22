#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import argparse
import random
import torch
import pf
import os

np.set_printoptions(precision=5, suppress=True)
Path("saved_models").mkdir(parents=True, exist_ok=True)

class PFNet(object):
    def __init__(self, params):
        self.params = params
        self.writer = SummaryWriter()

    def run_episode(self, model, episode_batch):
        trajlen = self.params.trajlen

        odometries = episode_batch['odometry']
        global_maps = episode_batch['global_map']
        observations = episode_batch['observation']
        init_particle_states = episode_batch['init_particles']
        init_particle_weights = episode_batch['init_particle_weights']

        # start with episode trajectory state with init particles and weights
        state = init_particle_states, init_particle_weights

        t_particle_states = []
        t_particle_weights = []

        # iterate over trajectory_length
        for traj in range(trajlen):
            obs = observations[:, traj, :, :, :]
            odom = odometries[:, traj, :]
            inputs = obs, odom, global_maps

            outputs, state = model(inputs, state)

            t_particle_states.append(outputs[0].unsqueeze(1))
            t_particle_weights.append(outputs[1].unsqueeze(1))

        t_particle_states = torch.cat(t_particle_states, axis=1)
        t_particle_weights = torch.cat(t_particle_weights, axis=1)

        outputs = t_particle_states, t_particle_weights

        return outputs

    def preprocess_data(self, batch_samples):
        episode_batch = {}

        batch_size, num_particles = batch_samples['init_particles'].shape[:2]
        episode_batch['init_particle_weights'] = torch.full((batch_size, num_particles), np.log(1.0/num_particles)).to(self.params.device)
        episode_batch['init_particles'] = batch_samples['init_particles'].to(self.params.device)
        episode_batch['true_states'] = batch_samples['true_states'].to(self.params.device)
        episode_batch['observation'] = batch_samples['observation'].to(self.params.device)
        episode_batch['global_map'] = batch_samples['global_map'].to(self.params.device)
        episode_batch['odometry'] = batch_samples['odometry'].to(self.params.device)

        return episode_batch

    def run_training(self):
        trajlen = self.params.trajlen
        batch_size = self.params.batch_size
        num_particles = self.params.num_particles

        # data loader
        composed = transforms.Compose([
                    pf.ToTensor(),
        ])
        train_dataset = pf.House3DTrajDataset(params, params.train_file, transform=composed)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True)

        # define model
        model = pf.PFCell(params)

        self.params.batch_size //= self.params.device_count
        if params.multiple_gpu:
            model = torch.nn.DataParallel(model)
        model.to(self.params.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.01)

        print('dataloader initialized')
        # iterate over num_epochs
        for epoch in range(self.params.num_epochs):
            b_loss_total = []
            b_loss_coords = []
            model.train()
            # iterate over num_batches
            for batch_idx, batch_samples in enumerate(train_loader):
                episode_batch = self.preprocess_data(batch_samples)

                # skip if batch_size doesn't match
                labels = episode_batch['true_states']
                if batch_size != labels.shape[0]:
                    break

                outputs = self.run_episode(model, episode_batch)

                # [batch_size, trajlen, [num_particles], ..]
                losses = self.loss_fn(outputs[0], outputs[1], labels)
                losses['loss_total'].backward()

                # visualize gradient flow
                # pf.plot_grad_flow(self.pf_cell.named_parameters())

                # update parameters based on gradients
                self.optimizer.step()

                # cleat gradients
                self.optimizer.zero_grad()

                loss_total = losses['loss_total'].item()
                loss_coords = losses['loss_coords'].item()

                b_loss_total.append(loss_total)
                b_loss_coords.append(loss_coords)

                # log per epoch batch stats
                print('epoch: {0:05d}, batch: {1:05d}, b_loss_coords: {2:03.3f}, b_loss_total: {3:03.3f}'.format(epoch, batch_idx, loss_coords, loss_total))
                self.writer.add_scalars('epoch-{0:03d}_train_stats'.format(epoch), {
                    'b_total_loss': loss_total,
                    'b_coords_loss': loss_coords
                }, batch_idx)

            # log per epoch mean stats
            print('epoch: {0:05d}, mean_loss_coords: {1:03.3f}, mean_loss_total: {2:03.3f}'.format(epoch, np.mean(b_loss_coords), np.mean(b_loss_total)))
            self.writer.add_scalars('train_stats', {
                    'mean_total_loss': np.mean(b_loss_total),
                    'mean_coords_loss': np.mean(b_loss_coords)
            }, epoch)

            # save
            file_name = 'saved_models/' + 'pfnet_eps_{0:05d}.pth'.format(epoch)
            self.save(file_name)

        print('training finished')

    def loss_fn(self, particle_states, particle_weights, true_states):

        lin_weights = torch.nn.functional.softmax(particle_weights, dim=-1)

        true_coords = true_states[:, :, :2]
        mean_coords = torch.sum(torch.mul(particle_states[:, :, :, :2], lin_weights[:, :, :, None]), dim=2)
        coord_diffs = mean_coords - true_coords

        # convert from pixel coordinates to meters
        coord_diffs *= self.params.map_pixel_in_meters

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

    def save(self, model, file_name):
        torch.save({
            'pf_cell': model.state_dict(),
        }, file_name)

    def load(self, model, file_name):
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint['pf_cell'])

    def test(self):
        file_name = 'saved_models/pfnet_eps_0.pth'
        self.load(file_name)
        self.set_eval_mode()

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
    argparser.add_argument('--use_cpu', type=str2bool, nargs='?', const=True, default=False, help='cpu training')
    argparser.add_argument('--multiple_gpu', type=str2bool, nargs='?', const=True, default=False, help='use multiple gpu for training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    print("#########################")
    print(params)
    print("#########################")

    params.device_count = 1
    if not params.use_cpu and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(elem) for elem in range(torch.cuda.device_count())])
        params.device = torch.device('cuda:0')
        print('GPU detected: ', os.environ['CUDA_VISIBLE_DEVICES'])
        params.device_count = torch.cuda.device_count()
    else:
        params.use_cpu = True
        params.device = torch.device('cpu')
        print('No GPU. switching to CPU')

    params.trajlen = 24
    params.map_pixel_in_meters = 0.02
    params.init_particles_distr = 'gaussian'
    params.init_particles_std = ['0.3', '0.523599']  # 30cm, 30degrees

    # convert multi-input fileds to numpy arrays
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.transition_std = np.array(params.transition_std, np.float32)

    # set common seed value
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    pf_net = PFNet(params)

    pf_net.run_training()

    # pf_net.test()
