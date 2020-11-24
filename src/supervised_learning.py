#!/usr/bin/env python3

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import networks.networks as nets
import utils.constants as constants
import utils.helpers as helpers
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses
import cv2
import os
from scipy.stats import norm

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)
np.set_printoptions(precision=3)
(w, h) = (256, 256)
num_particles = 500
batch_size = 25

def get_motion_data(idx):
    file_name = 'sup_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(idx)
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

def get_rnd_particles_data(idx):
    file_name = 'sup_data/rnd_particles_data/particles_{:04d}.pkl'.format(idx)
    with open(file_name,'rb') as file:
        particles = pickle.load(file)
        rnd_particles = []
        for idx in range(batch_size):
            rnd_indices = np.random.randint(0, 4999, size=num_particles)
            rnd_particles.append(particles[rnd_indices])
        rnd_particles = np.asarray(rnd_particles)
        # rnd_indices = np.random.randint(0, 4999, size=num_particles)
        # rnd_particles = particles[rnd_indices]
    return rnd_particles

def get_observation_data(idx):
    file_name = 'sup_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(idx)
    observation_data = {
        'obs_pose': [],
        'obs_rgb': [],
        'env_map': [],
    }

    floor_idx = 0
    model_path = '/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/envs/iGibson/gibson2/dataset/Rs'
    trav_map = cv2.imread(os.path.join(model_path, 'floor_trav_{0}.png'.format(floor_idx)))
    env_map = cv2.resize(trav_map, (w, h))

    with open(file_name,'rb') as file:
        data = pickle.load(file)
        for eps_data in data:
            observation_data['obs_pose'].append(eps_data['pose'])
            observation_data['obs_rgb'].append(eps_data['state']['rgb'])
            observation_data['env_map'].append(env_map)
    rnd_indices = np.random.randint(0, 99, size=batch_size)
    for key in observation_data.keys():
        observation_data[key] = np.asarray(observation_data[key])[rnd_indices]
    return observation_data

def train_motion_model():
    motion_net = nets.MotionNetwork().to(constants.DEVICE)
    motion_net.train()
    loss_fn = nn.MSELoss()
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

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, learn_noise=True, simple_model=False)

            est_new_poses = helpers.transform_poses(est_new_poses)
            gt_new_poses = helpers.transform_poses(helpers.to_tensor(gt_new_poses))

            loss = loss_fn(est_new_poses, gt_new_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('motion_train/mse_loss', loss.item(), train_idx)
            train_idx = train_idx + 1
            total_loss = total_loss + loss.item()
        print('mean mse loss: {0}'.format(total_loss/num_data_files))
    file_name = 'motion_model.pt'
    torch.save({
        'motion_net': motion_net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name)

def test_motion_model():
    motion_net = nets.MotionNetwork().to(constants.DEVICE)
    motion_net.eval()
    loss_fn = nn.MSELoss()

    file_name = 'motion_model_complex.pt'
    checkpoint = torch.load(file_name)
    motion_net.load_state_dict(checkpoint['motion_net'])

    with torch.no_grad():
        num_epochs = 10
        num_data_files = 75
        for j in range(num_epochs):
            rnd_idx = np.random.randint(0, num_data_files)
            motion_data = get_motion_data(rnd_idx)
            gt_old_poses = motion_data['start_pose']
            gt_new_poses = motion_data['end_pose']
            gt_actions = motion_data['action']
            gt_delta_t = motion_data['delta_t']

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, learn_noise=True, simple_model=False)

            gt_new_poses = helpers.transform_poses(helpers.to_tensor(gt_new_poses))
            est_new_poses = helpers.transform_poses(est_new_poses)

            loss = loss_fn(est_new_poses, gt_new_poses)
            print(loss)

def compute_labels(gt_poses, rnd_particles):
    particles_batch = []
    labels_batch = []
    for idx in range(gt_poses.shape[0]):
        dist = helpers.eucld_dist(gt_poses[idx], rnd_particles[idx])
        sorted, indices = torch.sort(dist, descending=True)
        labels = torch.arange(0, rnd_particles.shape[1], dtype=torch.float32)
        labels = labels / labels.sum()

        particles_batch.append(rnd_particles[idx][indices])
        labels_batch.append(labels)
    return torch.stack(particles_batch), torch.stack(labels_batch)

def get_triplet_labels(gt_pose, rnd_particles, desc=False):
    dist = helpers.eucld_dist(gt_pose, rnd_particles)
    _, indices = torch.sort(dist, descending=desc)
    return indices

def get_mse_labels(gt_pose, rnd_particles, std=1.0):
    gt_pose = helpers.to_numpy(gt_pose)
    rnd_particles = helpers.to_numpy(rnd_particles)
    labels = norm.pdf(rnd_particles, loc=gt_pose, scale=std)
    labels = F.softmax(helpers.to_tensor(labels), dim=0)
    return labels

def get_mse_loss(gt_pose, rnd_particles, std=0.5):
    sqrt_dist = helpers.eucld_dist(gt_pose, rnd_particles)
    activations = (1/(rnd_particles.shape[0]*np.sqrt(2 *np.pi * std**2))) * torch.exp(-sqrt_dist/(2 * std**2))
    loss = torch.mean(-torch.log(1e-16 + activations))
    return loss

def train_measurement_model():
    vision_net = nets.VisionNetwork(w, h).to(constants.DEVICE)
    likeli_net = nets.LikelihoodNetwork().to(constants.DEVICE)
    vision_net.train()
    likeli_net.train()
    #loss_fn = losses.TripletMarginLoss()
    loss_fn = nn.L1Loss()
    params = list(likeli_net.parameters()) + list(vision_net.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-4)
    writer = SummaryWriter()

    num_epochs = 1000
    num_data_files = 75
    train_idx = 0
    for j in range(num_epochs):
        total_loss = 0
        for idx in range(num_data_files):

            obs_data = get_observation_data(idx)
            gt_poses = obs_data['obs_pose']
            rgbs = helpers.to_tensor(obs_data['obs_rgb'])
            env_maps = helpers.to_tensor(obs_data['env_map'])

            imgs = torch.cat([rgbs, env_maps], axis=-1).permute(0, 3, 1, 2) # from NHWC to NCHW
            encoded_imgs = vision_net(imgs)

            gt_trans_poses = helpers.transform_poses(helpers.to_tensor(gt_poses))
            particles = get_rnd_particles_data(idx)
            encoded_imgs = encoded_imgs.unsqueeze(1).repeat(1, particles.shape[1], 1)
            batch_loss = []

            #
            for b_idx in range(batch_size):
                b_particles = particles[b_idx]
                arg_poses = gt_poses[b_idx] + np.random.normal(0., 1.0, size=b_particles.shape)
                arg_poses[:, 2:3] = helpers.wrap_angle(arg_poses[:, 2:3], use_numpy=True)
                arg_trans_poses = helpers.transform_poses(helpers.to_tensor(arg_poses))
                b_trans_particles = helpers.transform_poses(helpers.to_tensor(b_particles))

                labels = get_mse_labels(gt_trans_poses[b_idx], b_trans_particles)
                arg_labels = get_mse_labels(gt_trans_poses[b_idx], arg_trans_poses)

                input_features = torch.cat([encoded_imgs[b_idx], b_trans_particles], axis=-1)
                _, likelihoods = likeli_net(input_features)

                input_features = torch.cat([encoded_imgs[b_idx], arg_trans_poses], axis=-1)
                _, arg_likelihoods = likeli_net(input_features)

                likelihoods = torch.cat([likelihoods, arg_likelihoods], dim=0)
                labels = torch.cat([labels, arg_labels], dim=0)

                loss = loss_fn(likelihoods, labels)
                writer.add_scalar('measurement_train/l1_loss', loss.item(), train_idx)
                train_idx = train_idx + 1
                total_loss = total_loss + loss.item()
                batch_loss.append(loss)

            # # TRIPLET LOSS
            # for b_idx in range(batch_size):
            #     rnd_particles = helpers.transform_poses(helpers.to_tensor(particles[b_idx]))
            #
            #     input_features = torch.cat([encoded_imgs[b_idx], rnd_particles], axis=-1)
            #     embeddings, _ = likeli_net(input_features)
            #     labels = get_triplet_labels(gt_poses[b_idx], rnd_particles)
            #
            #     arg_gt_poses = gt_poses[b_idx] + torch.normal(0., 0.1, \
            #                 size=rnd_particles.shape).to(constants.DEVICE)
            #     input_features = torch.cat([encoded_imgs[b_idx], arg_gt_poses], axis=-1)
            #     augmented, _ = likeli_net(input_features)
            #     arg_labels = get_triplet_labels(gt_poses[b_idx], arg_gt_poses)
            #
            #     embeddings = torch.cat([embeddings, augmented], dim=0)
            #     labels = torch.cat([labels, arg_labels], dim=0)
            #
            #     #mse_loss = get_mse_loss(gt_poses[b_idx], rnd_particles)
            #     triplet_loss = loss_fn(embeddings, labels)
            #     b_loss = triplet_loss #+ mse_loss
            #     batch_loss.append(b_loss)
            #
            #     writer.add_scalar('measurement_train/triplet_loss', b_loss.item(), train_idx)
            #     #writer.add_scalar('measurement_train/mse_loss', mse_loss.item(), train_idx)
            #     train_idx = train_idx + 1
            #     total_loss = total_loss + b_loss.item()

            loss = torch.stack(batch_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('mean triplet loss: {0}'.format(total_loss/(num_data_files*batch_size)))

    file_name = 'measurement_model_train.pt'
    torch.save({
        'vision_net': vision_net.state_dict(),
        'likeli_net': likeli_net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name)

def test_measurement_model():
    vision_net = nets.VisionNetwork(w, h).to(constants.DEVICE)
    likeli_net = nets.LikelihoodNetwork().to(constants.DEVICE)
    vision_net.eval()
    likeli_net.eval()
    #loss_fn = losses.TripletMarginLoss()
    loss_fn = nn.MSELoss()

    # file_name = 'measurement_model.pt'
    # checkpoint = torch.load(file_name)
    # vision_net.load_state_dict(checkpoint['vision_net'])
    # likeli_net.load_state_dict(checkpoint['likeli_net'])

    num_epochs = 1
    num_data_files = 75
    for j in range(num_epochs):
        rnd_idx = np.random.randint(0, num_data_files)
        obs_data = get_observation_data(rnd_idx)

        gt_poses = obs_data['obs_pose']
        rgbs = helpers.to_tensor(obs_data['obs_rgb'])
        env_maps = helpers.to_tensor(obs_data['env_map'])

        b_idx = np.random.randint(0, batch_size)
        x = gt_poses[b_idx] + np.random.normal(0., 1.0, size=3)
        print(gt_poses[b_idx], x)
        value = norm.pdf(x, loc=gt_poses[b_idx], scale=1.0)
        print(value)

        # imgs = torch.cat([rgbs, env_maps], axis=-1).permute(0, 3, 1, 2) # from NHWC to NCHW
        # encoded_imgs = vision_net(imgs)
        #
        # gt_poses = helpers.transform_poses(helpers.to_tensor(gt_poses))
        # particles = get_rnd_particles_data(rnd_idx)
        # encoded_imgs = encoded_imgs.unsqueeze(1).repeat(1, particles.shape[1], 1)
        #
        # b_idx = np.random.randint(0, batch_size)
        # rnd_particles = helpers.transform_poses(helpers.to_tensor(particles[b_idx]))
        #
        # input_features = torch.cat([encoded_imgs[b_idx], rnd_particles], axis=-1)
        # _, likelihoods = likeli_net(input_features)
        # labels = get_mse_labels(gt_poses[b_idx], rnd_particles)
        # loss = loss_fn(likelihoods, labels)
        # print(likelihoods.shape, labels.shape, loss)


if __name__ == '__main__':
    #train_motion_model()
    #test_motion_model()

    train_measurement_model()
    #test_measurement_model()
