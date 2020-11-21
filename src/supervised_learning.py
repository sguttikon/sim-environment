#!/usr/bin/env python3

import pickle
import numpy as np
import torch
import torch.nn as nn
import random
import networks.networks as nets
import utils.constants as constants
import utils.helpers as helpers
from torch.utils.tensorboard import SummaryWriter

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)
np.set_printoptions(precision=3)

def get_motion_data(idx, batch_size=25):
    file_name = 'sup_data/rnd_pose_data/data_{:04d}.pkl'.format(idx)
    motion_data = {
        'start_pose': [],
        'action': [],
        'end_pose': [],
        'delta_t': [],
    }
    with open(file_name,'rb') as file:
        data = pickle.load(file)
        old_pose = None
        for eps_data in data:
            if old_pose is None:
                old_pose = eps_data['pose']
                continue

            new_pose = eps_data['pose']
            motion_data['end_pose'].append(new_pose)
            motion_data['start_pose'].append(old_pose)
            motion_data['action'].append(eps_data['vel_cmd'])
            motion_data['delta_t'].append(eps_data['delta_t'])

            old_pose = new_pose
    rnd_indices = np.random.randint(0, 99, size=batch_size)
    for key in motion_data.keys():
        motion_data[key] = np.asarray(motion_data[key])[rnd_indices]
    return motion_data

def train_motion_model():
    motion_net = nets.MotionNetwork().to(constants.DEVICE)
    motion_net.train()
    mse_loss = nn.MSELoss()
    params = list(motion_net.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-4)
    writer = SummaryWriter()

    num_epochs = 1000
    num_data_files = 75
    train_idx = 0
    for j in range(num_epochs):
        total_loss = 0
        for idx in range(num_data_files):
            motion_data = get_motion_data(idx)
            gt_old_poses = motion_data['start_pose']
            gt_new_poses = motion_data['end_pose']
            gt_actions = motion_data['action']
            gt_delta_t = motion_data['delta_t']

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, learn_noise=True, simple_model=True)

            loss = mse_loss(est_new_poses, helpers.to_tensor(gt_new_poses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/mse_loss', loss.item(), train_idx)
            train_idx = train_idx + 1
            total_loss = total_loss + loss.item()
        print('mean mse loss: {0}'.format(total_loss/num_data_files))
    file_name = 'model.pt'
    torch.save({
        'motion_net': motion_net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name)

def test_motion_model():
    motion_net = nets.MotionNetwork().to(constants.DEVICE)
    motion_net.eval()

    file_name = 'model.pt'
    checkpoint = torch.load(file_name)
    motion_net.load_state_dict(checkpoint['motion_net'])

    with torch.no_grad():
        num_epochs = 10
        num_data_files = 75
        for j in range(num_epochs):
            rnd_idx = np.random.randint(0, num_data_files)
            motion_data = get_motion_data(rnd_idx, batch_size=1)
            gt_old_poses = motion_data['start_pose']
            gt_new_poses = motion_data['end_pose']
            gt_actions = motion_data['action']
            gt_delta_t = motion_data['delta_t']

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, learn_noise=True, simple_model=True)
            print(gt_new_poses, est_new_poses)


if __name__ == '__main__':
    train_motion_model()
    test_motion_model()

    # motion_data = get_motion_data(10)
    # for idx in range(1,2):
    #     old_pose = motion_data['start_pose'][idx]
    #     new_pose = motion_data['end_pose'][idx]
    #     vel_cmd = motion_data['action'][idx]
    #     delta_t = motion_data['delta_t'][idx]
    #
    #     print(new_pose, helpers.sample_motion_model_velocity(vel_cmd, old_pose, delta_t, use_noise=True), helpers.sample_motion_model_velocity(vel_cmd, old_pose, delta_t))
