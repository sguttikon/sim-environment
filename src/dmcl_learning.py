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

    dmcl = DMCL(config_filename, render=False, agent='RANDOM')
    writer = SummaryWriter()

    num_epochs = 500
    epoch_len = 250
    train_idx = 0
    eval_idx = 0
    acc = np.inf
    ent = np.inf
    for curr_epoch in range(num_epochs):
        # train
        particles, particles_probs = dmcl.init_particles()
        cum_loss = 0.
        cum_acc = 0.
        for curr_step in range(epoch_len):
            train_idx = train_idx + 1

            particles, info = dmcl.step(particles)

            writer.add_scalar('train/total_loss', info['total_loss'].item(), train_idx)
            writer.add_scalar('train/measurement_loss', info['loss'].item(), train_idx)
            writer.add_scalar('train/entropy', info['entropy'].item(), train_idx)
            writer.add_scalar('train/mse', info['mse'].item(), train_idx)
            cum_loss = cum_loss + info['total_loss'].item()
            cum_acc = cum_acc + info['mse'].item()
        writer.add_scalar('train/eps_end_entropy', info['entropy'].item(), train_idx)
        writer.add_scalar('train/eps_end_mse', info['mse'].item(), train_idx)

        mean_loss = cum_loss / epoch_len
        mean_acc = cum_acc / epoch_len
        print('mean loss: {0:.4f}, mean mse: {1:.4f}'.format(mean_loss, mean_acc))
        print('episode entropy: {0:.4f}, episode mse: {1:.4f}'.format(info['entropy'].item(), info['mse'].item()))

        if curr_epoch%10 == 0:
            file_path = 'saved_models/train_model_{}.pt'.format(curr_epoch)
            dmcl.save(file_path)

            # eval
            particles, particles_probs = dmcl.init_particles()
            cum_acc = 0.
            for curr_step in range(epoch_len):
                eval_idx = eval_idx + 1

                particles, info = dmcl.predict(particles)

                writer.add_scalar('eval/entropy', info['entropy'].item(), eval_idx)
                writer.add_scalar('eval/mse', info['mse'].item(), eval_idx)
                cum_acc = cum_acc + info['mse'].item()
            writer.add_scalar('eval/eps_end_entropy', info['entropy'].item(), train_idx)
            writer.add_scalar('eval/eps_end_mse', info['mse'].item(), train_idx)

            mean_acc = cum_acc / epoch_len
            if mean_acc < acc and info['entropy'].item() < ent:
                acc = mean_acc
                ent = info['entropy'].item()
                file_path = 'best_models/model.pt'
                dmcl.save(file_path)
                print('=> best accuracy: {0:.4f}, entropy: {1:.4f}'.format(acc, ent))

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
