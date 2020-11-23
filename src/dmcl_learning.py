#!/usr/bin/env python3

# import os
# from DPF.dpf import DMCL
# import matplotlib.pyplot as plt
#
# curr_dir_path = os.path.dirname(os.path.abspath(__file__))
# config_file_path = os.path.join(curr_dir_path, 'config/turtlebot.yaml')
#
#
# dmcl = DMCL(config_file_path)
# dmcl.train()
#
# # to prevent plot from closing
# plt.ioff()
# plt.show()


import os
from DPF.dmcl import DMCL
import utils.helpers as helpers
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import torch

def train():
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(curr_dir_path, 'config/turtlebot.yaml')
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("best_models").mkdir(parents=True, exist_ok=True)

    dmcl = DMCL(config_filename, render=True, agent='RANDOM')
    writer = SummaryWriter()

    num_epochs = 2000
    epoch_len = 250
    train_idx = 0
    eval_idx = 0
    best_err = np.inf
    for curr_epoch in range(num_epochs):
        # train
        gt_pose, particles = dmcl.init_particles(is_uniform=True)
        triplet_losses = []
        mse_losses = []
        for curr_step in range(epoch_len):
            train_idx = train_idx + 1

            gt_pose, particles, info = dmcl.step(particles)

            writer.add_scalar('train/triplet_loss', info['triplet_loss'].item(), train_idx)
            writer.add_scalar('train/mse_loss', info['mse_loss'].item(), train_idx)
            triplet_losses.append(info['triplet_loss'].item())
            mse_losses.append(info['mse_loss'].item())

        mean_particles = np.mean(particles, axis=0)
        curr_err = helpers.eucld_dist(gt_pose, mean_particles, use_numpy=True)
        writer.add_scalar('train/pose_err', curr_err.item(), train_idx)

        print('train=> mean triplet_loss: {0:.4f}, mean mse_loss: {1:.4f}, pose error: {2:.4f}'.\
                format(np.array(triplet_losses).mean(), np.array(mse_losses).mean(), curr_err))

        if curr_epoch%10 == 0:
            file_path = 'saved_models/train_model_{}.pt'.format(curr_epoch)
            dmcl.save(file_path)

            # eval
            gt_pose, particles = dmcl.init_particles(is_uniform=True)
            triplet_losses = []
            mse_losses = []
            for curr_step in range(epoch_len):
                eval_idx = eval_idx + 1

                gt_pose, particles, info = dmcl.predict(particles)

                writer.add_scalar('eval/triplet_loss', info['triplet_loss'].item(), eval_idx)
                writer.add_scalar('eval/mse_loss', info['mse_loss'].item(), eval_idx)
                triplet_losses.append(info['triplet_loss'].item())
                mse_losses.append(info['mse_loss'].item())

            mean_particles = np.mean(particles, axis=0)
            curr_err = helpers.eucld_dist(gt_pose, mean_particles, use_numpy=True)
            writer.add_scalar('eval/pose_err', curr_err.item(), eval_idx)

            print('eval=====> mean triplet_loss: {0:.4f}, mean mse_loss: {1:.4f}, pose error: {2:.4f}'.\
                format(np.array(triplet_losses).mean(), np.array(mse_losses).mean(), curr_err))

            if curr_err < best_err:
                best_err = curr_err
                file_path = 'best_models/model.pt'
                dmcl.save(file_path)

    writer.close()

def test():
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(curr_dir_path, 'config/turtlebot.yaml')
    dmcl = DMCL(config_filename, render=True, agent='RANDOM')

    file_path = 'best_models/model.pt'
    dmcl.load(file_path)

    num_epochs = 5
    epoch_len = 50
    for curr_epoch in range(num_epochs):
        particles, particles_probs = dmcl.init_particles()
        for curr_step in range(epoch_len):
            particles, info = dmcl.predict(particles)
        print('episode entropy: {0:.4f}, episode mse: {1:.4f}'.format(info['entropy'].item(), info['mse'].item()))

if __name__ == '__main__':
    train()
    #test()
