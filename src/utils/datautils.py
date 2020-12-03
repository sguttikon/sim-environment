#!/usr/bin/env python3

import os
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from skimage.color import gray2rgb
import utils.constants as constants
import utils.helpers as helpers
from scipy.stats import norm

class ObservationDataset(Dataset):
    """
    reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, obs_pkl_file, particles_pkl_file, transform=None):
        """
        """
        self.transform = transform
        with open(obs_pkl_file,'rb') as file:
            self.obs_pkl_data = pickle.load(file)

        with open(particles_pkl_file,'rb') as file:
            self.particles_pkl_data = pickle.load(file)

        floor_idx = 0
        model_path = '/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/envs/iGibson/gibson2/dataset/Rs'
        img_name = os.path.join(model_path, 'floor_trav_{0}.png'.format(floor_idx))
        self.env_map = io.imread(img_name)

    def __len__(self):
        return len(self.obs_pkl_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.obs_pkl_data[idx]
        sample['env_map'] = gray2rgb(self.env_map)

        std = 0.5
        gt_pose = sample['pose']
        gt_pose = np.expand_dims(gt_pose, axis=0)

        # add estimated particles
        sample['est_particles'] = self.particles_pkl_data

        eucld_dist = helpers.eucld_dist(gt_pose, sample['est_particles'], use_numpy=True)
        sample['est_labels'] = norm.pdf(eucld_dist, loc=0, scale=std).squeeze()

        # add gaussian particles around gt pose
        shape = sample['est_particles'].shape
        gt_particles = np.random.normal(loc=gt_pose, scale=std, size=shape)
        gt_particles[:, 2:3] = helpers.wrap_angle(gt_particles[:, 2:3], use_numpy=True) # wrap angle
        sample['gt_particles'] = gt_particles

        eucld_dist = helpers.eucld_dist(gt_pose, sample['gt_particles'], use_numpy=True)
        sample['gt_labels'] = norm.pdf(eucld_dist, loc=0, scale=std).squeeze()

        sample['pose'] = gt_pose

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    :params
        output_size (tuple or int): desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        rgb_img = sample['state']['rgb']
        env_map = sample['env_map']

        h, w = rgb_img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        new_rgb_img = transform.resize(rgb_img, (new_h, new_w))
        new_env_map = transform.resize(env_map, (new_h, new_w))

        sample['state']['rgb'] = new_rgb_img
        sample['env_map'] = new_env_map

        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    :params
        output_size (tuple or int): desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        rgb_img = sample['state']['rgb']

        h, w = rgb_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_rgb_img = rgb_img[top: top + new_h, left: left + new_w]

        sample['state']['rgb'] = new_rgb_img

        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        rgb_img = sample['state']['rgb']
        env_map = sample['env_map']
        pose = sample['pose']
        gt_particles = sample['gt_particles']
        gt_labels = sample['gt_labels']
        est_particles = sample['est_particles']
        est_labels = sample['est_labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        rgb_img = rgb_img.transpose((2, 0, 1))
        new_rgb_img = torch.from_numpy(rgb_img).float()

        env_map = env_map.transpose((2, 0, 1))
        new_env_map = torch.from_numpy(env_map).float()

        new_pose = torch.from_numpy(pose).float()
        new_gt_particles = torch.from_numpy(gt_particles).float()
        new_gt_labels = torch.from_numpy(gt_labels).float()
        new_est_particles = torch.from_numpy(est_particles).float()
        new_est_labels = torch.from_numpy(est_labels).float()

        sample['state']['rgb'] = new_rgb_img
        sample['env_map'] = new_env_map
        sample['pose'] = new_pose
        sample['gt_particles'] = new_gt_particles
        sample['gt_labels'] = new_gt_labels
        sample['est_particles'] = new_est_particles
        sample['est_labels'] = new_est_labels

        return sample

class Normalize(object):
    """
    Normalize the tensor image with mean and standard deviation
    """

    def __init__(self):
        #assert all(isinstance(x, float) for x in mean)
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        rgb_img = sample['state']['rgb']
        env_map = sample['env_map']

        new_rgb_img = self.normalize(rgb_img)
        new_env_map = self.normalize(env_map)

        sample['state']['rgb'] = new_rgb_img
        sample['env_map'] = new_env_map

        return sample
