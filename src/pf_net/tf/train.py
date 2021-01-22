#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter
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

        composed = transforms.Compose([
                    pf.ToTensor(),
        ])
        train_dataset = pf.House3DTrajDataset(params, params.train_file, transform=composed)
        self.train_data_loader = pf.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

        self.pf_cell = pf.PFCell(params).to(params.device)

        model_params =  list(self.pf_cell.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=2e-4, weight_decay=0.01)

        self.writer = SummaryWriter()

    def run_episode(self, episode_batch):
        trajlen = self.params.trajlen
        batch_size = self.params.batch_size
        num_particles = self.params.num_particles

        odometries = episode_batch['odometry'].to(params.device)
        global_maps = episode_batch['global_map'].to(params.device)
        observations = episode_batch['observation'].to(params.device)

        init_particle_states = episode_batch['init_particles'].to(params.device)
        init_particle_weights = torch.full((batch_size, num_particles), np.log(1.0/float(num_particles))).to(params.device)

        # start with episode trajectory state with init particles and weights
        state = init_particle_states, init_particle_weights

        t_particle_states = []
        t_particle_weights = []

        # iterate over trajectory_length
        for traj in range(trajlen):
            obs = observations[:, traj, :, :, :]
            odom = odometries[:, traj, :]
            inputs = obs, odom, global_maps

            outputs, state = self.pf_cell(inputs, state)

            t_particle_states.append(outputs[0].unsqueeze(1))
            t_particle_weights.append(outputs[1].unsqueeze(1))

        t_particle_states = torch.cat(t_particle_states, axis=1)
        t_particle_weights = torch.cat(t_particle_weights, axis=1)

        outputs = t_particle_states, t_particle_weights

        return outputs

    def run_training(self):
        batch_size = self.params.batch_size
        num_particles = self.params.num_particles

        # iterate over num_epochs
        for epoch in range(self.params.num_epochs):
            b_loss_total = []
            b_loss_coords = []
            self.set_train_mode()
            # iterate over num_batches
            for batch_idx, batch_samples in enumerate(self.train_data_loader):
                episode_batch = batch_samples
                labels = episode_batch['true_states'].to(params.device)

                # skip if batch_size doesn't match
                if batch_size != labels.shape[0]:
                    break

                outputs = self.run_episode(episode_batch)

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

                # log per batch stats
                print('epoch: {0:05d}, batch: {1:05d}, b_loss_coords: {2:03.3f}, b_loss_total: {3:03.3f}'.format(epoch, batch_idx, loss_coords, loss_total))
                self.writer.add_scalar('training/b_loss_total_eps{0}'.format(epoch), np.mean(loss_total), batch_idx)
                self.writer.add_scalar('training/b_loss_coords_eps{0}'.format(epoch), np.mean(loss_coords), batch_idx)

            # log per epoch mean stats
            print('epoch: {0:05d}, mean_loss_coords: {1:03.3f}, mean_loss_total: {2:03.3f}'.format(epoch, np.mean(b_loss_coords), np.mean(b_loss_total)))
            self.writer.add_scalar('training/mean_loss_total', np.mean(b_loss_total), epoch)
            self.writer.add_scalar('training/mean_loss_coords', np.mean(b_loss_coords), epoch)

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

    def set_train_mode(self):
        self.pf_cell.train()

    def set_eval_mode(self):
        self.pf_cell.eval()

    def save(self, file_name):
        torch.save({
            'pf_cell': self.pf_cell.state_dict(),
        }, file_name)

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        self.pf_cell.load_state_dict(checkpoint['pf_cell'])

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
    argparser.add_argument('--batch_size', type=int, default=4, help='batch size used for training')
    argparser.add_argument('--num_workers', type=int, default=0, help='workers used for data loading')
    argparser.add_argument('--num_particles', type=int, default=30, help='number of particles used for training')
    argparser.add_argument('--transition_std', nargs='*', default=[0, 0], help='error in motion execution')
    argparser.add_argument('--local_map_size', nargs='*', default=(28, 28), help='shape of local map')
    argparser.add_argument('--use_cpu', type=str2bool, nargs='?', const=True, default=False, help='cpu training')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')

    params = argparser.parse_args()

    print("#########################")
    print(params)
    print("#########################")

    if not params.use_cpu and torch.cuda.is_available():
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # tensorflow

    params.trajlen = 24
    params.map_pixel_in_meters = 0.02
    params.init_particles_distr = 'gaussian'
    params.init_particles_std = [0.3, 0.523599]  # 30cm, 30degrees

    # set common seed value
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    pf_net = PFNet(params)

    pf_net.run_training()

    # pf_net.test()
