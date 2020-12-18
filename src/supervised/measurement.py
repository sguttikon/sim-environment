#!/usr/bin/env python3

import numpy as np
import torch
from utils import helpers, constants, datautils, display
import networks.networks as nets
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

class Measurement(object):
    """
    """

    def __init__(self, vision_model_name='resnet34', loss='mse', render=False, pretrained=False):

        if vision_model_name == 'resnet34':
            self.vision_model_name = vision_model_name
            self.vision_model = models.resnet34(pretrained=pretrained).to(constants.DEVICE)
            # self.vision_model.fc = nn.Identity()

            for param in self.vision_model.parameters():
                param.requires_grad = False
            layers = ['layer4', 'avgpool']

        self.feature_extractor = nets.FeatureExtractor(self.vision_model, layers).to(constants.DEVICE)
        self.likelihood_net = nets.LikelihoodNetwork().to(constants.DEVICE)

        params = list(self.likelihood_net.parameters())
        self.optimizer = optim.Adam(params, lr=2e-4)

        if loss == 'mse':
            self.loss_fn_name = loss
            self.loss_fn = nn.MSELoss()

        self.writer = SummaryWriter()
        self.train_idx = 0
        self.eval_idx = 0

        self.best_eval_accuracy = np.inf

        self.num_data_files = 25

        if render:
            self.render = display.Render()
        else:
            self.render = None

    def get_obs_data_loader(self, file_idx, batch_size=constants.BATCH_SIZE):
        obs_file_name = 'igibson_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(file_idx)
        particles_file_name = 'igibson_data/rnd_particles_data/particles_{:04d}.pkl'.format(file_idx)

        # reference https://pytorch.org/docs/stable/torchvision/models.html
        composed = transforms.Compose([
                    datautils.Rescale(256),
                    datautils.RandomCrop(224),
                    datautils.ToTensor(),
                    datautils.Normalize()])

        obs_dataset = datautils.ObservationDataset(obs_pkl_file=obs_file_name,
                                    particles_pkl_file=particles_file_name,
                                    transform=composed)

        obs_data_loader = DataLoader(obs_dataset,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0)

        return obs_data_loader

    def set_train_mode(self):
        self.vision_model.train()
        self.feature_extractor.train()

    def set_eval_mode(self):
        self.vision_model.eval()
        self.feature_extractor.eval()

    def train(self, train_epochs=1, eval_epochs=1):
        eval_epoch = 5
        save_epoch = 10

        # iterate per epoch
        for epoch in range(train_epochs):
            # TRAIN
            self.set_train_mode()
            training_loss = []

            # iterate per pickle data file
            for file_idx in range(self.num_data_files):
                obs_data_loader = self.get_obs_data_loader(file_idx)

                # iterate per batch
                batch_loss = 0
                for _, batch_samples in enumerate(obs_data_loader):
                    self.optimizer.zero_grad()

                    # get data
                    batch_rgbs = batch_samples['state']['rgb'].to(constants.DEVICE)
                    batch_gt_poses = batch_samples['pose'].to(constants.DEVICE)
                    batch_gt_particles = batch_samples['gt_particles'].to(constants.DEVICE)
                    batch_gt_labels = batch_samples['gt_labels'].to(constants.DEVICE).squeeze()
                    batch_est_particles = batch_samples['est_particles'].to(constants.DEVICE)
                    batch_est_labels = batch_samples['est_labels'].to(constants.DEVICE).squeeze()

                    # transform particles from orientation angle to cosine and sine values
                    trans_batch_gt_particles = helpers.transform_poses(batch_gt_particles)
                    trans_batch_est_particles = helpers.transform_poses(batch_est_particles)

                    # get encoded image features
                    # features = self.vision_model(batch_rgbs)
                    features = self.feature_extractor(batch_rgbs)['avgpool']

                    # approach [p, img + 4]
                    img_features = features.view(batch_rgbs.shape[0], 1, -1)
                    repeat_img_features = img_features.repeat(1, batch_gt_particles.shape[1], 1)

                    # input_est_features = torch.cat([trans_batch_est_particles, repeat_img_features], axis=-1).squeeze()
                    # est_embeddings, est_likelihoods = self.likelihood_net(input_est_features)

                    input_gt_features = torch.cat([trans_batch_gt_particles, repeat_img_features], axis=-1).squeeze()
                    gt_embeddings, gt_likelihoods = self.likelihood_net(input_gt_features)

                    if self.loss_fn_name == 'mse':
                        # likelihoods = torch.cat([gt_likelihoods, est_likelihoods], dim=0)
                        # labels = torch.cat([batch_gt_labels, batch_est_labels], dim=0)
                        loss = self.loss_fn(gt_likelihoods.squeeze(), batch_gt_labels)

                    loss.backward()

                    # check gradient flow
                    # helpers.plot_grad_flow(self.vision_model.named_parameters())
                    # helpers.plot_grad_flow(self.likelihood_net.named_parameters())

                    self.optimizer.step()

                    batch_loss = batch_loss + loss

                # log
                self.writer.add_scalar('training/{}_b_loss'.format(self.loss_fn_name), batch_loss.item(), self.train_idx)
                self.train_idx = self.train_idx + 1
                training_loss.append(float(batch_loss))

            #
            print('mean loss: {0:4f}'.format(np.mean(training_loss)))

            if epoch%save_epoch == 0:
                file_name = 'saved_models/' + 'likelihood_{0}_idx_{1}.pth'.format(self.loss_fn_name, epoch)
                self.save(file_name)

            if epoch%eval_epoch == 0:
                self.eval(num_epochs=eval_epochs)

        print('training done')
        self.writer.close()

    def eval(self, num_epochs=1):
        rnd_files = 5
        total_loss = 0
        # iterate per epoch
        for epoch in range(num_epochs):
            # EVAL
            self.set_eval_mode()
            evaluation_loss = []

            rnd_indices = np.random.randint(0, self.num_data_files, size=rnd_files)
            # iterate per pickle data file
            for file_idx in rnd_indices:
                obs_data_loader = self.get_obs_data_loader(file_idx)

                # iterate per batch
                batch_loss = 0
                with torch.no_grad():
                    for _, batch_samples in enumerate(obs_data_loader):
                        # get data
                        batch_rgbs = batch_samples['state']['rgb'].to(constants.DEVICE)
                        batch_gt_poses = batch_samples['pose'].to(constants.DEVICE)
                        batch_gt_particles = batch_samples['gt_particles'].to(constants.DEVICE)
                        batch_gt_labels = batch_samples['gt_labels'].to(constants.DEVICE).squeeze()
                        batch_est_particles = batch_samples['est_particles'].to(constants.DEVICE)
                        batch_est_labels = batch_samples['est_labels'].to(constants.DEVICE).squeeze()

                        # transform particles from orientation angle to cosine and sine values
                        trans_batch_gt_particles = helpers.transform_poses(batch_gt_particles)
                        trans_batch_est_particles = helpers.transform_poses(batch_est_particles)

                        # get encoded image features
                        # features = self.vision_model(batch_rgbs)
                        features = self.feature_extractor(batch_rgbs)['avgpool']

                        # approach [p, img + 4]
                        img_features = features.view(batch_rgbs.shape[0], 1, -1)
                        repeat_img_features = img_features.repeat(1, batch_gt_particles.shape[1], 1)

                        # input_est_features = torch.cat([trans_batch_est_particles, repeat_img_features], axis=-1).squeeze()
                        # est_embeddings, est_likelihoods = self.likelihood_net(input_est_features)

                        input_gt_features = torch.cat([trans_batch_gt_particles, repeat_img_features], axis=-1).squeeze()
                        gt_embeddings, gt_likelihoods = self.likelihood_net(input_gt_features)

                        if self.loss_fn_name == 'mse':
                            # likelihoods = torch.cat([gt_likelihoods, est_likelihoods], dim=0)
                            # labels = torch.cat([batch_gt_labels, batch_est_labels], dim=0)
                            loss = self.loss_fn(gt_likelihoods.squeeze(), batch_gt_labels)

                        batch_loss = batch_loss + loss

                    # log
                    self.writer.add_scalar('evaluation/{}_b_loss'.format(self.loss_fn_name), batch_loss.item(), self.eval_idx)
                    self.eval_idx = self.eval_idx + 1
                    evaluation_loss.append(float(batch_loss))

            total_loss = total_loss + np.mean(evaluation_loss)

        if total_loss < self.best_eval_accuracy:
            self.best_eval_accuracy = total_loss
            print('new best loss: {0:4f}'.format(total_loss))

            file_name = 'best_models/' + 'likelihood_{0}_best.pth'.format(self.loss_fn_name)
            self.save(file_name)

    def test(self, file_name):
        self.load(file_name)
        self.set_eval_mode()

        rnd_idx = np.random.randint(0, self.num_data_files)
        obs_data_loader = self.get_obs_data_loader(rnd_idx, batch_size=1)
        with torch.no_grad():
            for _, batch_samples in enumerate(obs_data_loader):
                # get data
                batch_rgbs = batch_samples['state']['rgb'].to(constants.DEVICE)
                batch_gt_poses = batch_samples['pose'].to(constants.DEVICE)
                batch_gt_particles = batch_samples['gt_particles'].to(constants.DEVICE)
                batch_gt_labels = batch_samples['gt_labels'].to(constants.DEVICE).squeeze()
                batch_est_particles = batch_samples['est_particles'].to(constants.DEVICE)
                batch_est_labels = batch_samples['est_labels'].to(constants.DEVICE).squeeze()

                trans_batch_gt_poses = helpers.transform_poses(batch_gt_poses)
                trans_batch_gt_particles = helpers.transform_poses(batch_gt_particles)

                # get encoded image features
                # features = self.vision_model(batch_rgbs)
                features = self.feature_extractor(batch_rgbs)['avgpool']

                # approach [p, img + 4]
                img_features = features.view(batch_rgbs.shape[0], 1, -1)
                repeat_img_features = img_features.repeat(1, batch_gt_particles.shape[1], 1)

                input_gt_features = torch.cat([trans_batch_gt_particles, repeat_img_features], axis=-1)
                gt_embeddings, gt_likelihoods = self.likelihood_net(input_gt_features)

                if self.render is not None:
                    data = {
                        'occ_map': batch_samples['occ_map'],
                        'occ_map_res': batch_samples['occ_map_res'],
                        'robot_gt_pose': batch_samples['pose'],
                        'robot_gt_particles': batch_samples['gt_particles'],
                        'robot_gt_labels': batch_samples['gt_labels'],
                        'robot_est_labels': gt_likelihoods,
                    }
                    self.render.update_figures(data)
                loss = self.loss_fn(gt_likelihoods.squeeze(), batch_gt_labels)
                break

    def save(self, file_name):
        torch.save({
            'likelihood_net': self.likelihood_net.state_dict(),
            'vision_model': self.vision_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
        }, file_name)
        # print('=> created checkpoint')

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        self.likelihood_net.load_state_dict(checkpoint['likelihood_net'])
        self.vision_model.load_state_dict(checkpoint['vision_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        # print('=> loaded checkpoint')

    def __del__(self):
        del self.render

if __name__ == '__main__':
    measurement = Measurement()
    train_epochs=1
    eval_epochs=1
    # measurement.train(train_epochs, eval_epochs)
    # measurement.eval(eval_epochs)
    file_name = 'likelihood_mse_best.pth'
    measurement.test(file_name)
