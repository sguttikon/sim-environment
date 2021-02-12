#!/usr/bin/env python3

from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.assets_utils import get_model_path
from gibson2.utils.utils import parse_config

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import multivariate_normal
from transforms3d.euler import quat2euler
from skimage.viewer import ImageViewer
from torchvision import transforms
from skimage import io, transform
from display import Render
import networks as nets
import numpy as np
import datautils
import helpers
import scipy
import torch
import cv2
import os

class PFNet(object):

    def __init__(self, params):

        self.params = params
        self.env = NavigateRandomEnv(config_file = params.config_filepath,
                    mode = params.env_mode,
        )

        # set common seed value
        self.env.seed(params.seed)
        self.env.reset()

        self.robot = self.env.robots[0] #hardcoded

        config_data = parse_config(params.config_filepath)
        self.model_id = config_data['model_id']
        self.pixel_to_mts = config_data['trav_map_resolution'] # each map pixel in meter
        self.params.pixel_to_mts = self.pixel_to_mts

        if self.params.obs_model == 'OBS':
            self.observation_model = nets.ObservationModel(params)
            model_params = list(self.observation_model.resnet.parameters()) \
                         + list(self.observation_model.likelihood_net.parameters())
        elif self.params.obs_model == 'OBS_MAP':
            self.observation_model = nets.MapObservationModel(params)
            model_params = list(self.observation_model.obs_resnet.parameters()) \
                         + list(self.observation_model.map_feature_extractor.parameters()) \
                         + list(self.observation_model.map_obs_feature_extractor.parameters()) \
                         + list(self.observation_model.spatial_transform_net.parameters()) \
                         + list(self.observation_model.likelihood_net.parameters())

        self.resample = nets.ResamplingModel(params)
        self.transition_model = nets.TransitionModel(params)

        self.optimizer = torch.optim.Adam(model_params, lr=2e-4, weight_decay=4e-6)
        self.train_idx = 0

        self.global_floor_map, self.global_travs_map = self.get_env_map(config_data['trav_map_erosion'])
        layout_map = transform.resize(self.global_floor_map, (256, 256))
        self.layout_map = torch.from_numpy(layout_map).float().to(self.params.device).unsqueeze(0).unsqueeze(1)
        if params.render:
            self.render = Render()
            self.render.plot_map(self.global_floor_map, self.pixel_to_mts)
        else:
            self.render = None

    def init_particles(self):
        # reset environment
        env_obs = self.env.reset()

        rnd_particles = []
        num_particles = self.params.num_particles
        model = self.params.init_particles_model

        gt_pose = self.get_gt_pose()

        if model == 'GAUSS':
            mean = gt_pose # [0., 0., 0.]
            cov = [[0.5*0.5, 0, 0], [0, 0.5*0.5, 0], [0, 0, np.pi/12*np.pi/12]]  # diagonal covariance

            rnd_particles = multivariate_normal.rvs(mean, cov, size=num_particles)
            rnd_particles[:, 2] = helpers.normalize(rnd_particles[:, 2])
            rnd_particle_weights = multivariate_normal.logpdf(rnd_particles, mean, cov)
        elif model == 'UNIFORM':
            trav_map_size = self.global_travs_map.shape[0]
            trav_space = np.where(self.global_travs_map == 255)

            cnt = 0
            while cnt < num_particles:
                idx = np.random.randint(0, high=trav_space[0].shape[0])
                xy = np.array([trav_space[0][idx], trav_space[1][idx]])

                x, y = np.flip((xy - trav_map_size / 2.0) / trav_map_size / self.pixel_to_mts, axis=0)
                th = helpers.normalize(np.random.uniform(0, np.pi * 2) - np.pi)

                rnd_pose = [x, y, th]
                rnd_particles.append(rnd_pose)
                cnt = cnt + 1

            rnd_particles = np.array(rnd_particles)
            rnd_particle_weights = np.full(num_particles, np.log(1.0/num_particles))

        observation = env_obs['rgb']

        state = rnd_particles, rnd_particle_weights

        return state, observation

    def get_gt_pose(self):

        position = self.robot.get_position()
        euler_orientation = helpers.normalize(self.robot.get_rpy())

        gt_pose = np.array([
            position[0],        # x
            position[1],        # y
            euler_orientation[2] # yaw
        ])

        return gt_pose

    def get_est_pose(self, particle_states, particle_weights, isTensor=False):

        if isTensor:
            # shape [batch_size, particles, 3]
            lin_weights = torch.nn.functional.softmax(particle_weights, dim=-1)
            est_pose = torch.sum(torch.mul(particle_states[:, :, :3], lin_weights[:, :, None]), dim=1)
            est_pose[:, 2] = helpers.normalize(est_pose[:, 2], isTensor=True)

            covariance = []
            # iterate per batch
            for i in range(particle_states.shape[0]):
                cov = helpers.cov(particle_states[i], aweights=lin_weights[i])
                covariance.append(cov)

            covariance_matrix = torch.stack(covariance)
            isSingular = torch.isclose(torch.det(covariance_matrix), torch.zeros(1).to(self.params.device))[0]

            if isSingular:
                # covariance_matrix is singular
                entropy = torch.zeros(1).to(self.params.device) #TODO
            else:
                mvn = torch.distributions.multivariate_normal.MultivariateNormal(est_pose, covariance_matrix)
                entropy = mvn.entropy()
        else:
            # shape [particles, 3]
            lin_weights = scipy.special.softmax(particle_weights, axis=-1)
            est_pose = np.sum(np.multiply(particle_states[:, :3], lin_weights[:, None]), axis=0)
            est_pose[2] = helpers.normalize(est_pose[2], isTensor=False)

            covariance_matrix = np.cov(particle_states.T, aweights=lin_weights)
            isSingular = np.isclose(np.det(covariance_matrix), np.zeros(1))[0]

            if isSingular:
                # covariance_matrix is singular
                entropy = np.zeros(1) #TODO
            else:
                mvn = multivariate_normal(est_pose, covariance_matrix, seed=self.params.seed)
                entropy = mvn.entropy()
        return est_pose, covariance_matrix, entropy

    def get_env_map(self, trav_map_erosion):

        model_path = get_model_path(self.model_id)
        floor_idx = self.env.floor_num

        filename = os.path.join(model_path, 'floor_{}.png'.format(floor_idx))
        floor_map = io.imread(filename)

        filename = os.path.join(model_path, 'floor_trav_{}.png'.format(floor_idx))
        trav_map = io.imread(filename)
        trav_map[floor_map == 0] = 0
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))

        floor_map = cv2.flip(floor_map, 0)

        return floor_map, trav_map

    def transform_observation(self, rgb_img):

        # rescale
        new_h = new_w = 256

        rescaled = transform.resize(rgb_img, (new_h, new_w))

        # random crop
        h, w = new_h, new_w
        new_h = new_w = 224

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        rnd_crop = rescaled[top: top + new_h, left: left + new_w]

        # to tensor
        rnd_crop = rnd_crop.transpose((2, 0, 1))
        tensor_rgb_img = torch.from_numpy(rnd_crop).float()

        # normalize
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalizer = transforms.Normalize(mean=mean, std=std)

        new_rgb_img = normalizer(tensor_rgb_img)

        new_rgb_img = new_rgb_img.unsqueeze(0) # add batch dimension
        return new_rgb_img

    def transform_state(self, state):
        particle_states, particle_weights = state

        particle_states = torch.from_numpy(particle_states).float().to(self.params.device)
        particle_states = particle_states.unsqueeze(0) # add batch dimension

        particle_weights = torch.from_numpy(particle_weights).float().to(self.params.device)
        particle_weights = particle_weights.unsqueeze(0) # add batch dimension

        return particle_states, particle_weights

    def episode_run(self, inputs, state):
        particle_states, particle_weights = state
        observation, odometry, old_pose = inputs

        # observation update
        if self.params.learn_obs_model:
            if self.params.obs_model == 'OBS':
                lik = self.observation_model(particle_states, observation)
            elif self.params.obs_model == 'OBS_MAP':
                lik = self.observation_model(particle_states, observation, self.layout_map)
        else:
            mean = old_pose # [0., 0., 0.]
            cov = [[0.5*0.5, 0, 0], [0, 0.5*0.5, 0], [0, 0, np.pi/12*np.pi/12]]
            tmp_particles = particle_states.squeeze(0).detach().cpu().numpy()

            lik = multivariate_normal.logpdf(tmp_particles, mean, cov)
            lik = torch.from_numpy(lik).float().unsqueeze(0).to(self.params.device)

        particle_weights += lik  # unnormalized

        # resample
        if self.params.resample:
            particle_states, particle_weights = self.resample(particle_states, particle_weights)

        # construct output before motion update
        outputs = particle_states, particle_weights

        # motion update -> this will only affect the particle state input at the next step
        particle_states = self.transition_model(particle_states, odometry, old_pose)

        # construct new state
        state = particle_states.detach(), particle_weights.detach()

        return outputs, state

    def compute_loss(self, outputs, true_state):
        particle_states, particle_weights = outputs

        lin_weights = torch.nn.functional.softmax(particle_weights, dim=-1)

        true_coords = true_state[:, :2]
        mean_coords = torch.sum(torch.torch.mul(particle_states[:, :, :2], lin_weights[:, :, None]), dim=1)
        coord_diffs = mean_coords - true_coords

        # coordinate loss component: (x-x')^2 + (y-y')^2
        loss_coords = torch.sum(torch.square(coord_diffs), axis=1)

        true_orients = true_state[:, 2]
        orient_diffs = particle_states[:, :, 2] - true_orients[:, None]
        # normalize between -pi .. +pi
        orient_diffs = helpers.normalize(orient_diffs, isTensor=True)

        # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        loss_orient = torch.square(torch.sum(orient_diffs * lin_weights, dim=1))

        # combine translational and orientation losses
        loss_combined = loss_coords + 0.36 * loss_orient

        est_state, covariance, entropy = self.get_est_pose(particle_states, particle_weights, isTensor=True)

        total_loss = torch.mean(loss_combined) + 1.0 * entropy

        total_loss.backward(retain_graph=True)

        # view gradient flow
        # helpers.plot_grad_flow(self.observation_model.likelihood_net.named_parameters())
        # helpers.plot_grad_flow(self.observation_model.resnet.named_parameters())

        self.optimizer.step()

        self.optimizer.zero_grad()

        pose_mse = torch.norm(true_state-est_state)

        self.writer.add_scalar('training/loss' + ('_pretrained' if self.params.pretrained_model else ''), total_loss.item(), self.train_idx)
        self.train_idx = self.train_idx + 1
        return total_loss, pose_mse, entropy

    def set_train_mode(self):
        self.observation_model.set_train_mode()

    def set_eval_mode(self):
        self.observation_model.set_eval_mode()

    def save(self, file_name):
        if self.params.obs_model == 'OBS':
            torch.save({
                'likelihood_net': self.observation_model.likelihood_net.state_dict(),
                'resnet': self.observation_model.resnet.state_dict(),
            }, file_name)
        elif self.params.obs_model == 'OBS_MAP':
            torch.save({
                'likelihood_net': self.observation_model.likelihood_net.state_dict(),
                'obs_resnet': self.observation_model.obs_resnet.state_dict(),
                'map_feature_extractor': self.observation_model.map_feature_extractor.state_dict(),
                'map_obs_feature_extractor': self.observation_model.map_obs_feature_extractor.state_dict(),
                'spatial_transform_net': self.observation_model.spatial_transform_net.state_dict(),
            }, file_name)
        # print('=> created checkpoint')

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        if self.params.obs_model == 'OBS':
            self.observation_model.likelihood_net.load_state_dict(checkpoint['likelihood_net'])
        elif self.params.obs_model == 'OBS_MAP':
            self.observation_model.likelihood_net.load_state_dict(checkpoint['likelihood_net'])
            self.observation_model.obs_resnet.load_state_dict(checkpoint['obs_resnet'])
            self.observation_model.map_feature_extractor.load_state_dict(checkpoint['map_feature_extractor'])
            self.observation_model.map_obs_feature_extractor.load_state_dict(checkpoint['map_obs_feature_extractor'])
            self.observation_model.spatial_transform_net.load_state_dict(checkpoint['spatial_transform_net'])
        # print('=> checkpoint loaded')

    def plot_figures(self, gt_pose, est_pose, particle_states, particle_weights):
        data = {
            'gt_pose': gt_pose,
            'est_pose': est_pose,
            'particle_states': particle_states,
            'particle_weights': particle_weights,
        }
        if self.params.render:
            self.render.update_figures(data)

    def run_training(self):

        save_eps = 50
        self.writer = SummaryWriter()
        # iterate per episode
        for eps_idx in range(self.params.num_eps):
            self.set_train_mode()

            state, observation = self.init_particles()
            old_pose = self.get_gt_pose()

            # preprocess
            state = self.transform_state(state)

            # iterate per episode step
            for eps_step in range(self.params.eps_len):

                # take action in environment
                action = self.env.action_space.sample()
                #HACK avoid back movement
                action = 0 if action == 1 else action
                new_env_obs, _, done, _ = self.env.step(action)

                # preprocess
                new_pose = self.get_gt_pose()
                odometry = helpers.compute_odometry(old_pose, new_pose)
                observation = self.transform_observation(observation).to(self.params.device)

                inputs = observation, odometry, old_pose
                outputs, state = self.episode_run(inputs, state)

                true_state = torch.from_numpy(old_pose).float().to(self.params.device)
                true_state = true_state.unsqueeze(0) # add batch dimension

                # compute loss
                total_loss, pose_mse, entropy = self.compute_loss(outputs, true_state)

                # get latest observation
                old_pose = new_pose
                observation = new_env_obs['rgb']

                # early stop
                if pose_mse < self.params.stop_threshold[0] and entropy < self.params.stop_threshold[1]:
                    break

            self.writer.add_scalar('training/pose_mse', pose_mse.item(), eps_idx)
            self.writer.add_scalar('training/entropy', entropy.item(), eps_idx)

            if eps_idx%save_eps == 0:
                file_name = 'saved_models/' + 'pfnet_eps_{0}.pth'.format(eps_idx)
                self.save(file_name)

        print('training done')
        self.writer.close()

    def run_validation(self, file_name):

        self.load(file_name)

        # iterate per episode
        for eps in range(1):
            self.set_eval_mode()

            state, observation = self.init_particles()
            old_pose = self.get_gt_pose()

            # preprocess
            state = self.transform_state(state)

            # plot
            particle_states, particle_weights = state
            est_pose, covariance, entropy = self.get_est_pose(particle_states, particle_weights, isTensor=True)

            est_pose = est_pose.squeeze(0).detach().cpu().numpy()
            particle_states = particle_states.squeeze(0).detach().cpu().numpy()
            particle_weights = particle_weights.squeeze(0).detach().cpu().numpy()
            self.plot_figures(old_pose, est_pose, particle_states, particle_weights)

            # iterate per episode step
            with torch.no_grad():
                for eps_step in range(50):

                    # take action in environment
                    if self.params.manual_action:
                        # manual
                        value = input("\n step:{0} 0: Forward, 2: Right, 3:Left - ".format(eps_step))
                        try:
                            action = int(value)
                        except:
                            print("Invalid Input")
                            action = 0
                    else:
                        # random
                        action = self.env.action_space.sample()
                        #HACK avoid back movement
                        action = 0 if action == 1 else action

                    new_env_obs, _, done, _ = self.env.step(action)

                    # preprocess
                    new_pose = self.get_gt_pose()
                    odometry = helpers.compute_odometry(old_pose, new_pose)
                    observation = self.transform_observation(observation).to(self.params.device)

                    inputs = observation, odometry, old_pose
                    outputs, state = self.episode_run(inputs, state)

                    true_state = torch.from_numpy(old_pose).float().to(self.params.device)
                    true_state = true_state.unsqueeze(0) # add batch dimension

                    # plot
                    particle_states, particle_weights = outputs
                    est_state, covariance, entropy = self.get_est_pose(particle_states, particle_weights, isTensor=True)
                    pose_mse = torch.norm(true_state-est_state)

                    est_pose = est_state.squeeze(0).detach().cpu().numpy()
                    particle_states = particle_states.squeeze(0).detach().cpu().numpy()
                    particle_weights = particle_weights.squeeze(0).detach().cpu().numpy()
                    self.plot_figures(old_pose, est_pose, particle_states, particle_weights)

                    # get latest observation
                    old_pose = new_pose
                    observation = new_env_obs['rgb']

                    # early stop
                    if pose_mse < self.params.stop_threshold[0] and entropy < self.params.stop_threshold[1]:
                        break
                print('pose mse: {0}, entropy: {1}'.format(pose_mse, entropy.item()))
                print(' true_pose: {0} \n  est_pose: {1} \n covariance: {2}'.format(true_state, est_state, covariance))

    def show_local_map(self):
        state, observation = self.init_particles()
        old_pose = self.get_gt_pose()

        # preprocess
        state = self.transform_state(state)

        # plot
        particle_states, particle_weights = state
        est_pose, covariance, entropy = self.get_est_pose(particle_states, particle_weights, isTensor=True)

        est_pose = est_pose.squeeze(0).detach().cpu().numpy()
        particle_states = particle_states.squeeze(0).detach().cpu().numpy()
        particle_weights = particle_weights.squeeze(0).detach().cpu().numpy()
        self.plot_figures(old_pose, est_pose, particle_states, particle_weights)

        spatial_transformer = nets.SpatialTransformerNet(self.params)

        true_state = torch.from_numpy(old_pose).float().to(self.params.device)
        true_state = true_state.unsqueeze(0).unsqueeze(0).unsqueeze(0) # add batch dimension

        particles = state[0].unsqueeze(2).squeeze(0)
        layout_map = self.layout_map.repeat(particle_states.shape[0], 1, 1, 1)
        particles_local_map = spatial_transformer(true_state, self.layout_map)
        print(particles_local_map.shape)

        local_map = particles_local_map[0].squeeze()
        viewer = ImageViewer(local_map.cpu().detach().numpy())
        viewer.show()


    def __del__(self):
        del self.render
