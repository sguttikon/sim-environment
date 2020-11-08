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
from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.utils import parse_config
import utils.helpers as helpers
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
config_filename = os.path.join(curr_dir_path, 'config/turtlebot.yaml')
Path("saved_models").mkdir(parents=True, exist_ok=True)
Path("best_models").mkdir(parents=True, exist_ok=True)

env = NavigateRandomEnv(config_file = config_filename,
            mode = 'headless',  # ['headless', 'gui']
)
config_data = parse_config(config_filename)
scale = config_data['trav_map_resolution']
robot = env.robots[0]
dmcl = DMCL(robot)
writer = SummaryWriter()

num_epochs = 2000
epoch_len = 50
train_idx = 0
eval_idx = 0
acc = np.inf
for curr_epoch in range(num_epochs):
    obs = env.reset()
    gt_pose = helpers.get_gt_pose(robot)
    particles, particles_probs = dmcl.init_particles(gt_pose)
    for curr_step in range(epoch_len):
        train_idx = train_idx + 1

        # TODO: get action from trained agent
        action = env.action_space.sample() * 1.5

        obs, reward, done, info = env.step(action)
        particles, total_loss, mse = \
                dmcl.step(obs['rgb'], action, particles)

        writer.add_scalar('train/loss', total_loss.item(), train_idx)
        writer.add_scalar('train/mse', mse.item(), train_idx)
        if curr_step%10 == 0:
            print('loss: {0}, mse: {1}'.format(total_loss.item(), mse.item()))

    if curr_epoch%10 == 0:
        file_path = 'saved_models/train_model_{}.pt'.format(curr_epoch)
        dmcl.save(file_path)

        obs = env.reset()
        gt_pose = helpers.get_gt_pose(robot)
        particles, particles_probs = dmcl.init_particles(gt_pose)
        total_err = 0.
        for curr_step in range(epoch_len):
            eval_idx = eval_idx + 1

            # TODO: get action from trained agent
            action = env.action_space.sample() * 1.5

            obs, reward, done, info = env.step(action)
            particles, mse = dmcl.predict(obs['rgb'], action, particles)

            writer.add_scalar('eval/mse', mse.item(), eval_idx)
            total_err = total_err + mse.item()

        mean_acc = total_err / epoch_len
        if mean_acc < acc:
            acc = mean_acc
            file_path = 'best_models/model.pt'
            dmcl.save(file_path)
            print('=> best accuracy: {0}'.format()acc)

writer.close()
