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
from pathlib import Path

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)
torch.cuda.manual_seed(constants.RANDOM_SEED)
torch.manual_seed(constants.RANDOM_SEED)
np.set_printoptions(precision=3)
(w, h) = (256, 256)
num_particles = 500
batch_size = 25

Path("saved_models").mkdir(parents=True, exist_ok=True)
Path("best_models").mkdir(parents=True, exist_ok=True)

motion_net = nets.MotionNetwork().to(constants.DEVICE)
motion_params = list(motion_net.parameters())
motion_optim = torch.optim.Adam(motion_params, lr=2e-4)

vision_net = nets.VisionNetwork(w, h).to(constants.DEVICE)
likelihood_net = nets.LikelihoodNetwork(num_particles).to(constants.DEVICE)
measure_params = list(likelihood_net.parameters()) + list(vision_net.parameters())
measure_optim = torch.optim.Adam(measure_params, lr=2e-4)

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

def save_motion_model(file_name):
    torch.save({
        'motion_net': motion_net.state_dict(),
        'optimizer': motion_optim.state_dict(),
    }, file_name)

def train_motion_model(use_noise=True, simple_model=False):
    if simple_model:
        file_name = 'motion_model_simple_{0}.pt'
    else:
        file_name = 'motion_model_complex_{0}.pt'

    loss_fn = nn.MSELoss()
    writer = SummaryWriter()

    num_epochs = 2000
    num_data_files = 75
    train_idx = 0
    eval_idx = 0
    best_acc = np.inf

    for j in range(num_epochs):
        # TRAIN
        train_losses = []
        motion_net.train()
        for idx in range(num_data_files):
            motion_data = get_motion_data(idx)
            gt_old_poses = motion_data['start_pose']
            gt_new_poses = motion_data['end_pose']
            gt_actions = motion_data['action']
            gt_delta_t = motion_data['delta_t']

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, use_noise, simple_model)

            est_new_poses = helpers.transform_poses(est_new_poses)
            gt_new_poses = helpers.transform_poses(helpers.to_tensor(gt_new_poses))

            loss = loss_fn(est_new_poses, gt_new_poses)
            motion_optim.zero_grad()
            loss.backward()
            motion_optim.step()

            writer.add_scalar('motion_train/mse_loss', loss.item(), train_idx)
            train_idx = train_idx + 1
            train_losses.append(float(loss))
        print('mean mse loss: {0}'.format(np.mean(train_losses)))

        if j%50 == 0:
            save_motion_model('saved_models/' + file_name.format(j))

            # EVAL
            eval_losses = []
            motion_net.eval()
            for idx in range(5):
                rnd_idx = np.random.randint(0, num_data_files)
                motion_data = get_motion_data(rnd_idx)
                gt_old_poses = motion_data['start_pose']
                gt_new_poses = motion_data['end_pose']
                gt_actions = motion_data['action']
                gt_delta_t = motion_data['delta_t']

                est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, use_noise, simple_model)

                gt_new_poses = helpers.transform_poses(helpers.to_tensor(gt_new_poses))
                est_new_poses = helpers.transform_poses(est_new_poses)

                loss = loss_fn(est_new_poses, gt_new_poses)
                writer.add_scalar('motion_eval/mse_loss', loss.item(), eval_idx)
                eval_idx = eval_idx + 1
                eval_losses.append(float(loss))

            if np.mean(eval_losses) < best_acc:
                print('new best mse loss: {0}'.format(np.mean(eval_losses)))
                best_acc = np.mean(eval_losses)
                save_motion_model('best_models/' + file_name.format(0))
    writer.close()

def test_motion_model(use_noise=True, simple_model=False):
    if simple_model:
        file_name = 'motion_model_simple.pt'
    else:
        file_name = 'motion_model_complex.pt'
    checkpoint = torch.load(file_name)
    motion_net.load_state_dict(checkpoint['motion_net'])

    motion_net.eval()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        num_epochs = 1
        num_data_files = 75
        losses = []
        for j in range(num_epochs):
            rnd_idx = np.random.randint(0, num_data_files)
            motion_data = get_motion_data(rnd_idx)
            gt_old_poses = motion_data['start_pose']
            gt_new_poses = motion_data['end_pose']
            gt_actions = motion_data['action']
            gt_delta_t = motion_data['delta_t']

            est_new_poses = motion_net(gt_old_poses, gt_actions, gt_delta_t, use_noise, simple_model)

            gt_new_poses = helpers.transform_poses(helpers.to_tensor(gt_new_poses))
            est_new_poses = helpers.transform_poses(est_new_poses)

            loss = loss_fn(est_new_poses, gt_new_poses)
            losses.append(float(loss))
        print('mean mse loss: {0}'.format(np.mean(losses)))

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

def get_mse_labels(gt_pose, rnd_particles, mean=0, std=1.0):
    labels = []
    for b_idx in range(batch_size):
        eucld_dist = helpers.eucld_dist(gt_pose[b_idx], rnd_particles[b_idx])
        label = norm.pdf(helpers.to_numpy(eucld_dist), loc=mean, scale=std)
        label = F.softmax(helpers.to_tensor(label), dim=0)
        labels.append(label)
    return torch.stack(labels)

def get_mse_loss(gt_pose, rnd_particles, std=0.5):
    sqrt_dist = helpers.eucld_dist(gt_pose, rnd_particles)
    activations = (1/(rnd_particles.shape[0]*np.sqrt(2 *np.pi * std**2))) * torch.exp(-sqrt_dist/(2 * std**2))
    loss = torch.mean(-torch.log(1e-16 + activations))
    return loss

def save_measurement_model(file_name):
    torch.save({
        'vision_net': vision_net.state_dict(),
        'likelihood_net': likelihood_net.state_dict(),
        'optimizer': measure_optim.state_dict(),
    }, file_name)

def train_measurement_model():
    #file_name = 'measure_model_mse_{0}.pt'
    file_name = 'measure_model_triplet_{0}.pt'

    writer = SummaryWriter()
    loss_fn = losses.TripletMarginLoss()
    #loss_fn = nn.MSELoss()
    std = 0.5

    num_epochs = 1000
    num_data_files = 75
    train_idx = 0
    eval_idx = 0
    best_acc = np.inf

    for j in range(num_epochs):
        # TRAIN
        train_losses = []
        vision_net.train()
        likelihood_net.train()
        for idx in range(num_data_files):
            obs_data = get_observation_data(idx)
            gt_poses = obs_data['obs_pose']
            rgbs = obs_data['obs_rgb']
            env_maps = obs_data['env_map']

            trans_gt_poses = helpers.transform_poses(helpers.to_tensor(gt_poses))

            rnd_particles = get_rnd_particles_data(idx)
            arg_poses = np.expand_dims(gt_poses, axis=1) + np.random.normal(loc=0., scale=std, size=rnd_particles.shape)
            arg_poses[:, 2:3] = helpers.wrap_angle(arg_poses[:, 2:3], use_numpy=True)
            trans_b_particles = []
            trans_b_poses = []
            for b_idx in range(batch_size):
                trans_b_particles.append(helpers.transform_poses(helpers.to_tensor(rnd_particles[b_idx])))
                trans_b_poses.append(helpers.transform_poses(helpers.to_tensor(arg_poses[b_idx])))
            trans_rnd_particles = torch.stack(trans_b_particles)
            trans_arg_poses = torch.stack(trans_b_poses)

            # labels = get_mse_labels(trans_gt_poses, trans_rnd_particles)
            # arg_labels = get_mse_labels(trans_gt_poses, trans_arg_poses)

            trans_rnd_particles = torch.flatten(trans_rnd_particles, start_dim=1, end_dim=-1)
            trans_arg_poses = torch.flatten(trans_arg_poses, start_dim=1, end_dim=-1)

            imgs = np.concatenate([rgbs, env_maps], axis=-1)
            imgs = helpers.to_tensor(imgs).permute(0, 3, 1, 2) # from NHWC to NCHW
            encoded_imgs = vision_net(imgs)

            input_features = torch.cat([encoded_imgs, trans_rnd_particles], axis=-1)
            embeddings, likelihoods = likelihood_net(input_features)
            labels = torch.arange(embeddings.size(0))

            input_features = torch.cat([encoded_imgs, trans_arg_poses], axis=-1)
            augmented, arg_likelihoods = likelihood_net(input_features)

            embeddings = torch.cat([embeddings, augmented], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            loss = loss_fn(embeddings, labels)
            writer.add_scalar('measurement_train/triplet_loss', loss.item(), train_idx)

            # likelihoods = torch.cat([likelihoods, arg_likelihoods], dim=0)
            # labels = torch.cat([labels, arg_labels], dim=0)
            # loss = loss_fn(likelihoods, labels)
            # writer.add_scalar('measurement_train/mse_loss', loss.item(), train_idx)

            measure_optim.zero_grad()
            loss.backward()
            measure_optim.step()

            train_idx = train_idx + 1
            train_losses.append(float(loss))
        print('mean loss: {0}'.format(np.mean(train_losses)))

        if j%50 == 0:
            save_measurement_model('saved_models/' + file_name.format(j))

            # EVAL
            eval_losses = []
            vision_net.eval()
            likelihood_net.eval()
            for idx in range(5):
                rnd_idx = np.random.randint(0, num_data_files)
                obs_data = get_observation_data(rnd_idx)
                gt_poses = obs_data['obs_pose']
                rgbs = obs_data['obs_rgb']
                env_maps = obs_data['env_map']

                trans_gt_poses = helpers.transform_poses(helpers.to_tensor(gt_poses))

                rnd_particles = get_rnd_particles_data(rnd_idx)
                arg_poses = np.expand_dims(gt_poses, axis=1) + np.random.normal(loc=0., scale=std, size=rnd_particles.shape)
                arg_poses[:, 2:3] = helpers.wrap_angle(arg_poses[:, 2:3], use_numpy=True)
                trans_b_particles = []
                trans_b_poses = []
                for b_idx in range(batch_size):
                    trans_b_particles.append(helpers.transform_poses(helpers.to_tensor(rnd_particles[b_idx])))
                    trans_b_poses.append(helpers.transform_poses(helpers.to_tensor(arg_poses[b_idx])))
                trans_rnd_particles = torch.stack(trans_b_particles)
                trans_arg_poses = torch.stack(trans_b_poses)

                # labels = get_mse_labels(trans_gt_poses, trans_rnd_particles)
                # arg_labels = get_mse_labels(trans_gt_poses, trans_arg_poses)

                trans_rnd_particles = torch.flatten(trans_rnd_particles, start_dim=1, end_dim=-1)
                trans_arg_poses = torch.flatten(trans_arg_poses, start_dim=1, end_dim=-1)

                imgs = np.concatenate([rgbs, env_maps], axis=-1)
                imgs = helpers.to_tensor(imgs).permute(0, 3, 1, 2) # from NHWC to NCHW
                encoded_imgs = vision_net(imgs)

                input_features = torch.cat([encoded_imgs, trans_rnd_particles], axis=-1)
                embeddings, likelihoods = likelihood_net(input_features)
                labels = torch.arange(embeddings.size(0))

                input_features = torch.cat([encoded_imgs, trans_arg_poses], axis=-1)
                augmented, arg_likelihoods = likelihood_net(input_features)

                embeddings = torch.cat([embeddings, augmented], dim=0)
                labels = torch.cat([labels, labels], dim=0)
                loss = loss_fn(embeddings, labels)
                writer.add_scalar('measurement_eval/triplet_loss', loss.item(), eval_idx)

                # likelihoods = torch.cat([likelihoods, arg_likelihoods], dim=0)
                # labels = torch.cat([labels, arg_labels], dim=0)
                # loss = loss_fn(likelihoods, labels)
                # writer.add_scalar('measurement_eval/mse_loss', loss.item(), eval_idx)

                eval_idx = eval_idx + 1
                eval_losses.append(float(loss))

            if np.mean(eval_losses) < best_acc:
                print('new best loss: {0}'.format(np.mean(eval_losses)))
                best_acc = np.mean(eval_losses)
                save_measurement_model('best_models/' + file_name.format(0))
    writer.close()

def test_measurement_model():
    file_name = 'measure_model_triplet.pt'
    checkpoint = torch.load(file_name)
    vision_net.load_state_dict(checkpoint['vision_net'])
    likelihood_net.load_state_dict(checkpoint['likelihood_net'])

    vision_net.eval()
    likelihood_net.eval()

    #loss_fn = losses.TripletMarginLoss()
    loss_fn = nn.MSELoss()
    std = 0.5

    with torch.no_grad():
        num_epochs = 1
        num_data_files = 75
        losses = []
        for j in range(num_epochs):
            rnd_idx = np.random.randint(0, num_data_files)
            obs_data = get_observation_data(rnd_idx)
            gt_poses = obs_data['obs_pose']
            rgbs = obs_data['obs_rgb']
            env_maps = obs_data['env_map']

            trans_gt_poses = helpers.transform_poses(helpers.to_tensor(gt_poses))

            rnd_particles = get_rnd_particles_data(rnd_idx)
            arg_poses = np.expand_dims(gt_poses, axis=1) + np.random.normal(loc=0., scale=std, size=rnd_particles.shape)
            arg_poses[:, 2:3] = helpers.wrap_angle(arg_poses[:, 2:3], use_numpy=True)
            trans_b_particles = []
            trans_b_poses = []
            for b_idx in range(batch_size):
                trans_b_particles.append(helpers.transform_poses(helpers.to_tensor(rnd_particles[b_idx])))
                trans_b_poses.append(helpers.transform_poses(helpers.to_tensor(arg_poses[b_idx])))
            trans_rnd_particles = torch.stack(trans_b_particles)
            trans_arg_poses = torch.stack(trans_b_poses)

            labels = get_mse_labels(trans_gt_poses, trans_rnd_particles)
            arg_labels = get_mse_labels(trans_gt_poses, trans_arg_poses)

            trans_rnd_particles = torch.flatten(trans_rnd_particles, start_dim=1, end_dim=-1)
            trans_arg_poses = torch.flatten(trans_arg_poses, start_dim=1, end_dim=-1)

            imgs = np.concatenate([rgbs, env_maps], axis=-1)
            imgs = helpers.to_tensor(imgs).permute(0, 3, 1, 2) # from NHWC to NCHW
            encoded_imgs = vision_net(imgs)

            input_features = torch.cat([encoded_imgs, trans_rnd_particles], axis=-1)
            _, likelihoods = likelihood_net(input_features)

            input_features = torch.cat([encoded_imgs, trans_arg_poses], axis=-1)
            _, arg_likelihoods = likelihood_net(input_features)

            likelihoods = torch.cat([likelihoods, arg_likelihoods], dim=0)
            labels = torch.cat([labels, arg_labels], dim=0)

            _, indices = torch.sort(labels[0], descending=True)
            print(torch.sum(likelihoods[0][indices]), labels[0][indices])
            loss = loss_fn(likelihoods, labels)
            losses.append(float(loss))
        print('mean loss: {0}'.format(np.mean(losses)))

def plot_gt_trajectory(idx):
    file_name = 'sup_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(idx)
    writer = SummaryWriter('runs/data_{0}/gt_trajectory'.format(idx))
    with open(file_name,'rb') as file:
        data = pickle.load(file)
        gt_old_pose = None
        for eps_idx in range(len(data)):
            eps_data = data[eps_idx]
            print(eps_data['pose'], eps_data['vel_cmd'], eps_data['delta_t'])
            if gt_old_pose is None:
                gt_old_pose = eps_data['pose']
                continue

            gt_new_pose = eps_data['pose']
            gt_action = eps_data['vel_cmd']
            gt_delta_t = eps_data['delta_t']

            # plot trajectory as euclidean distance
            trans_gt_old_pose = helpers.transform_poses(helpers.to_tensor(gt_old_pose))
            trans_gt_new_pose = helpers.transform_poses(helpers.to_tensor(gt_new_pose))
            eucld_dist = helpers.eucld_dist(trans_gt_old_pose, trans_gt_new_pose, use_numpy=False)
            writer.add_scalar('eucld_dist', float(eucld_dist), eps_idx)

            gt_old_pose = gt_new_pose
    writer.close()

def plot_est_trajectory(idx, use_learned, use_noise, simple_model):
    if use_learned:
        motion_net = nets.MotionNetwork().to(constants.DEVICE)
        if simple_model:
            file_name = 'motion_model_simple.pt'
        else:
            file_name = 'motion_model_complex.pt'
        checkpoint = torch.load(file_name)
        motion_net.load_state_dict(checkpoint['motion_net'])
    else:
        motion_net = nets.SampleMotionModel().to(constants.DEVICE)
    motion_net.eval()

    file_name = 'sup_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(idx)

    noise_type = 'noisy' if use_noise else 'no_noise'
    model_type = 'learned' if use_learned else 'no_learned'
    motion_type = 'simple' if simple_model else 'complex'
    writer = SummaryWriter('runs/data_{0}/est_trajectory/{1}-{2}-{3}'.format(idx, model_type, noise_type, motion_type))
    with open(file_name,'rb') as file:
        data = pickle.load(file)
        est_old_poses = None
        for eps_idx in range(len(data)):
            eps_data = data[eps_idx]
            if est_old_poses is None:
                est_old_poses = np.asarray([eps_data['pose']])
                continue

            gt_actions = np.asarray([eps_data['vel_cmd']])
            gt_delta_t = np.asarray([eps_data['delta_t']])

            est_new_poses = motion_net(est_old_poses, gt_actions, gt_delta_t, use_noise, simple_model)

            if use_learned:
                est_new_poses = helpers.to_numpy(est_new_poses)

            # plot trajectory as euclidean distance
            trans_est_old_poses = helpers.transform_poses(helpers.to_tensor(est_old_poses))
            trans_est_new_poses = helpers.transform_poses(helpers.to_tensor(est_new_poses))
            eucld_dist = helpers.eucld_dist(trans_est_old_poses, trans_est_new_poses, use_numpy=False)
            writer.add_scalar('eucld_dist', float(eucld_dist), eps_idx)

            est_old_poses = est_new_poses
    writer.close()

if __name__ == '__main__':
    print('motion model')
    # train_motion_model(simple_model=False)
    # test_motion_model(simple_model=False)
    # train_motion_model(simple_model=True)
    # test_motion_model(simple_model=True)

    # num_data_files = 75
    # rnd_idx = np.random.randint(0, num_data_files)
    # plot_gt_trajectory(rnd_idx)
    # plot_est_trajectory(rnd_idx, use_learned=False, use_noise=False, simple_model=True)
    # plot_est_trajectory(rnd_idx, use_learned=False, use_noise=False, simple_model=False)
    # plot_est_trajectory(rnd_idx, use_learned=False, use_noise=True, simple_model=True)
    # plot_est_trajectory(rnd_idx, use_learned=False, use_noise=True, simple_model=False)
    # plot_est_trajectory(rnd_idx, use_learned=True, use_noise=True, simple_model=False)
    # plot_est_trajectory(rnd_idx, use_learned=True, use_noise=True, simple_model=True)

    print('measurement model')
    #train_measurement_model()
    test_measurement_model()
