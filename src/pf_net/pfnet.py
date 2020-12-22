#!/usr/bin/env python3

from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.assets_utils import get_model_path
from gibson2.utils.utils import parse_config

from torch.utils.tensorboard import SummaryWriter
from transforms3d.euler import quat2euler
from torchvision import transforms
from skimage import io, transform
from display import Render
import networks as nets
import numpy as np
import datautils
import helpers
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

        self.observation_model = nets.ObservationModel(params)
        self.resample = nets.ResamplingModel(params)
        self.transition_model = nets.TransitionModel(params)

        model_params = list(self.observation_model.resnet.parameters()) \
                        + list(self.observation_model.likelihood_net.parameters()) \

        self.optimizer = torch.optim.Adam(model_params, lr=2e-4, weight_decay=4e-6)
        self.train_idx = 0

        if params.render:
            self.render = Render()

            self.global_floor_map = self.get_env_map()
            self.render.plot_map(self.global_floor_map, self.pixel_to_mts)

            state, observation = self.init_particles()

            curr_gt_pose = self.get_gt_pose()
            self.render.plot_robot(curr_gt_pose, 'green')

            particle_states, particle_weights = state
            self.render.plot_particles(particle_states, 'coral')
        else:
            self.render = None

    def init_particles(self):
        # reset environment
        env_obs = self.env.reset()

        rnd_particles = []
        num_particles = self.params.num_particles

        lmt = 0.5
        gt_pose = self.get_gt_pose()
        bounds = np.array([
            [gt_pose[0] - lmt, gt_pose[0] + lmt],
            [gt_pose[1] - lmt, gt_pose[1] + lmt],
        ])

        cnt = 0
        translation_std = self.params.init_particles_std[0]
        rotation_std = self.params.init_particles_std[1]

        while cnt < num_particles:
            _, self.initial_pos = self.env.scene.get_random_point_floor(self.env.floor_num, self.env.random_height)#
            self.initial_pos = self.initial_pos + np.random.uniform(0, 1.0, size=self.initial_pos.shape) * translation_std
            self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            rnd_pose = [self.initial_pos[0], self.initial_pos[1], self.initial_orn[2]]

            if bounds[0][0] <= rnd_pose[0] <= bounds[0][1] and \
               bounds[1][0] <= rnd_pose[1] <= bounds[1][1]:
                rnd_particles.append(rnd_pose)
                cnt = cnt + 1
            else:
                continue

        rnd_particles = np.array(rnd_particles)
        rnd_particle_weights = np.full(num_particles, np.log(1.0/num_particles))

        observation = env_obs['rgb']

        state = rnd_particles, rnd_particle_weights

        return state, observation

    def get_gt_pose(self):

        position = self.robot.get_position()
        euler_orientation = quat2euler(self.robot.get_orientation())

        gt_pose = np.array([
            position[0],
            position[1],
            euler_orientation[0]
        ])

        return gt_pose

    def get_env_map(self):

        model_path = get_model_path(self.model_id)
        floor_idx = self.env.floor_num
        filename = os.path.join(model_path, 'floor_{}.png'.format(floor_idx))

        floor_map = cv2.flip(io.imread(filename), 0)
        return floor_map

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
        observation, odometry = inputs

        # observation update
        lik = self.observation_model(particle_states, observation)
        particle_weights += lik  # unnormalized

        # resample
        particle_states, particle_weights = self.resample(particle_states, particle_weights)

        # construct output before motion update
        outputs = particle_states, particle_weights

        # motion update -> this will only affect the particle state input at the next step
        particle_states = self.transition_model(particle_states, odometry)

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
        orient_diffs = datautils.wrap_angle(orient_diffs, isTensor=True)

        # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        loss_orient = torch.square(torch.sum(orient_diffs * lin_weights, axis=1))

        # combine translational and orientation losses
        loss_combined = loss_coords + 0.36 * loss_orient
        total_loss = torch.mean(loss_combined)

        total_loss.backward(retain_graph=True)

        # view gradient flow
        # helpers.plot_grad_flow(self.observation_model.likelihood_net.named_parameters())
        # helpers.plot_grad_flow(self.observation_model.resnet.named_parameters())

        self.optimizer.step()

        self.optimizer.zero_grad()

        self.writer.add_scalar('training/loss' + ('_pretrained' if self.params.pretrained_model else ''), total_loss.item(), self.train_idx)
        self.train_idx = self.train_idx + 1
        return total_loss

    def set_train_mode(self):
        self.observation_model.set_train_mode()

    def set_eval_mode(self):
        self.observation_model.set_eval_mode()

    def save(self, file_name):
        torch.save({
            'likelihood_net': self.observation_model.likelihood_net.state_dict(),
            'resnet': self.observation_model.resnet.state_dict(),
        }, file_name)
        # print('=> created checkpoint')

    def run_training(self):

        self.writer = SummaryWriter()
        # iterate per episode
        for eps in range(self.params.num_eps):
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
                odometry = datautils.calc_odometry(old_pose, new_pose)
                observation = self.transform_observation(observation).to(self.params.device)

                inputs = observation, odometry
                outputs, state = self.episode_run(inputs, state)

                true_state = torch.from_numpy(old_pose).float().to(self.params.device)
                true_state = true_state.unsqueeze(0) # add batch dimension

                # compute loss
                self.compute_loss(outputs, true_state)

                # get latest observation
                old_pose = new_pose
                observation = new_env_obs['rgb']

            file_name = 'saved_models/' + 'pfnet_eps_{0}.pth'.format(eps)
            self.save(file_name)

        print('training done')
        self.writer.close()

    def __del__(self):
        del self.render
