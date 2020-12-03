#!/usr/bin/env python3

import os
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from skimage import io, transform
from skimage.color import gray2rgb

class ObservationDataset(Dataset):
    """
    reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, pkl_file, transform=None):
        """
        """
        self.transform = transform
        with open(pkl_file,'rb') as file:
            self.pkl_data = pickle.load(file)

        floor_idx = 0
        model_path = '/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/envs/iGibson/gibson2/dataset/Rs'
        img_name = os.path.join(model_path, 'floor_trav_{0}.png'.format(floor_idx))
        self.env_map = io.imread(img_name)

    def __len__(self):
        return len(self.pkl_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.pkl_data[idx]
        sample['env_map'] = gray2rgb(self.env_map)

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

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        rgb_img = rgb_img.transpose((2, 0, 1))
        new_rgb_img = torch.from_numpy(rgb_img)

        env_map = env_map.transpose((2, 0, 1))
        new_env_map = torch.from_numpy(env_map)

        sample['state']['rgb'] = new_rgb_img
        sample['env_map'] = new_env_map

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
