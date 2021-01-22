#!/usr/bin/env python3

from torch.utils.data import DataLoader
from skimage.viewer import ImageViewer
from torch.utils.data import Dataset
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, Tensor
import tensorflow as tf
import numpy as np
import argparse
import torch
import cv2
import os

test_data_size = 820
train_data_size = 74800
valid_data_size = 830

np.set_printoptions(precision=5, suppress=True)

fig = plt.figure(figsize=(7, 7))
plt_ax = fig.add_subplot(111)

class House3DTrajDataset(Dataset):

    def __init__(self, params, file, transform=None):
        self.params = params

        # build initial covariance matrix of particles, in pixels and radians
        if params.init_particles_distr == 'gaussian':
            particle_std = params.init_particles_std.copy()
            particle_std[0] = particle_std[0] / params.map_pixel_in_meters
            particle_var = np.square(particle_std)  # variance
            self.params.init_particles_cov = np.diag(particle_var[(0, 0, 1),]) # 3x3 matrix
        else:
            assert False

        self.map_shape = (3000, 3000, 1)

        self.raw_dataset = tf.data.TFRecordDataset(file)
        # self.raw_dataset_itr = list(self.raw_dataset.as_numpy_iterator())

        # iterate entries
        # self.raw_dataset = self.raw_dataset.apply(tf.data.experimental.assert_cardinality(self.test_data_size))
        # print(tf.data.experimental.cardinality(self.raw_dataset).numpy())

        # self.count = sum(1 for _ in self.raw_dataset)
        # print(self.count)

        self.transform = transform

    def __len__(self):
        if self.params.type == 'valid':
            return valid_data_size
        elif self.params.type == 'test':
            return test_data_size
        elif self.params.type == 'train':
            return train_data_size

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get idx element
        dataset = self.raw_dataset.skip(idx).take(1)
        data = list(dataset.as_numpy_iterator())[0]

        # costly
        # data = self.raw_dataset_itr[idx]

        result = tf.train.Example.FromString(data)
        features = result.features.feature

        # process maps
        map_wall = self.process_wall_map(features['map_wall'].bytes_list.value[0])
        global_map_list = [map_wall]
        #TODO other maps

        # input global map is a concatenation of semantic channels
        global_map = np.concatenate(global_map_list, axis=-1)

        # pad to have same global map size
        shape = global_map.shape
        new_global_map = np.zeros(self.map_shape, dtype=np.float32)
        new_global_map[:shape[0], :shape[1], :shape[2]] = global_map

        # process true states
        true_states = features['states'].bytes_list.value[0]
        true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

        # trajectory may be longer than what we use for training
        data_trajlen = true_states.shape[0]
        assert data_trajlen >= self.params.trajlen
        true_states = true_states[:self.params.trajlen]

        # process odometry
        odometry = features['odometry'].bytes_list.value[0]
        odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))
        odometry = odometry[:self.params.trajlen]

        # process observations
        observation = self.raw_images_to_array(list(features['rgb'].bytes_list.value)[:self.params.trajlen])

        # compute random init particles
        init_particles = self.random_particles(true_states[0], seed=self.get_sample_seed(self.params.seed, idx))

        sample = {}
        sample['true_states'] = true_states # np.array(np.split(true_states, self.num_segments))
        sample['global_map'] = new_global_map # np.expand_dims(new_global_map, axis=0)
        sample['init_particles'] = init_particles # np.expand_dims(init_particles, axis=0)
        sample['observation'] =  observation # np.array(np.split(observation, self.num_segments))
        sample['odometry'] = odometry # np.array(np.split(odometry, self.num_segments))
        # is_first = np.full(self.num_segments, False, dtype=bool)
        # is_first[0] = True
        # sample['is_first'] = is_first

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def process_wall_map(self, wallmap_feature):
        floormap = np.atleast_3d(self.decode_image(wallmap_feature))

        # transpose and invert
        floormap = 255 - np.transpose(floormap, axes=[1, 0, 2])
        # rescale to [-1, 1]
        floormap = floormap.astype(np.float32) * (2.0/255.0) - 1.0
        return floormap

    def get_sample_seed(self, seed, data_i):
        return (None if (seed is None or seed == 0) else ((data_i + 1) * 113 + seed))

    def random_particles(self, init_state, seed):
        particles = np.zeros((self.params.num_particles, 3), np.float32)

        if self.params.init_particles_distr == 'gaussian':
            # fix seed
            if seed is not None:
                random_state = np.random.get_state()
                np.random.seed(seed)

            # sample offset from the gaussian
            center = np.random.multivariate_normal(mean=init_state, cov=self.params.init_particles_cov)

            # restore random seed
            if seed is not None:
                np.random.set_state(random_state)

            # sample particles from gaussian centered around the offset
            particles = np.random.multivariate_normal(mean=center, cov=self.params.init_particles_cov, size=self.params.num_particles)
        else:
            assert False

        return particles

    def decode_image(self, img_str, resize=None):
        nparr = np.frombuffer(img_str, np.uint8)
        img_str = cv2.imdecode(nparr, -1)
        if resize is not None:
            img_str = cv2.resize(img_str, resize)
        return img_str

    def raw_images_to_array(self, images):
        image_list = []
        for image_str in images:
            image = np.atleast_3d(self.decode_image(image_str))
            # rescale to [-1, 1]
            image = image.astype(np.float32) * (2.0/255.0) - 1.0
            image_list.append(image)
        return np.stack(image_list, axis=0)

    def display_data(self, idx=0):

        # get idx element
        dataset = self.raw_dataset.skip(idx).take(1)
        data = list(dataset.as_numpy_iterator())[0]

        # costly
        # data = self.raw_dataset_itr[idx]

        result = tf.train.Example.FromString(data)
        features = result.features.feature

        map_wall = self.decode_image(features['map_wall'].bytes_list.value[0])

        true_states = features['states'].bytes_list.value[0]
        true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

        odometry = features['odometry'].bytes_list.value[0]
        odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

        rgb = self.raw_images_to_array(list(features['rgb'].bytes_list.value))

        init_particles = self.random_particles(true_states[0], seed=self.get_sample_seed(self.params.seed, idx))

        plt_ax.imshow(map_wall.transpose(), cmap='Greys',  interpolation='nearest')

        particles_plt = plt.scatter(init_particles[:, 0], init_particles[:, 1], s=10, c='blue', alpha=.5)

        for i in range(100):
            old_pose = true_states[i]
            odom = odometry[i]
            new_pose = sample_motion_odometry(old_pose, odom)

            x1, y1, _ = old_pose
            x2, y2, _ = new_pose
            # plt.scatter(x1, y1, s=50, c='blue', alpha=.5)

            dx = x2 - x1
            dy = y2 - y1
            if dx != 0 or dy != 0:
                plt_ax.arrow(x1, y1, dx, dy, width=1, head_width=1, head_length=1, fc='grey')
        pose_plt = Wedge((x2, y2), 5, 0, 360, color='blue', alpha=.75)
        plt_ax.add_artist(pose_plt)

        plt.show()

def normalize(angle, isTensor=False):
    '''
        wrap the give angle to range [-np.pi, np.pi]
    '''
    if isTensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    else:
        return np.arctan2(np.sin(angle), np.cos(angle))

def sample_motion_odometry(old_pose, odometry):
    x1, y1, th1 = old_pose
    odom_x, odom_y, odom_th = odometry

    sin = np.sin(th1)
    cos = np.cos(th1)

    x2 = x1 + (cos * odom_x - sin * odom_y)
    y2 = y1 + (sin * odom_x + cos * odom_y)
    th2 = normalize(th1 + odom_th)

    new_pose = np.array([x2, y2, th2])
    return new_pose

def calc_odometry(old_pose, new_pose):
    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    abs_x = (x2 - x1)
    abs_y = (y2 - y1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    odom_th = wrap_angle(th2 - th1)
    odom_x = cos * abs_x + sin * abs_y
    odom_y = cos * abs_y - sin * abs_x

    odometry = np.array([odom_x, odom_y, odom_th])
    return odometry

class ToTensor(object):

    def __call__(self, sample):

        true_states = sample['true_states']
        true_states = torch.from_numpy(true_states.copy()).float()
        sample['true_states'] = true_states

        init_particles = sample['init_particles']
        init_particles = torch.from_numpy(init_particles.copy()).float()
        sample['init_particles'] = init_particles

        odometry = sample['odometry']
        odometry = torch.from_numpy(odometry.copy()).float()
        sample['odometry'] = odometry

        # swap color axis because
        # numpy image: [trajlen] x H x W x C
        # torch image: [trajlen] x C x H x W

        global_map = sample['global_map']
        global_map = global_map.transpose((2, 0, 1))
        global_map = torch.from_numpy(global_map.copy()).float()
        sample['global_map'] = global_map

        observation = sample['observation']
        observation = observation.transpose((0, 3, 1, 2))
        observation = torch.from_numpy(observation.copy()).float()
        sample['observation'] =  observation

        return sample

class TransitionModel(nn.Module):

    def __init__(self, params):
        super(TransitionModel, self).__init__()
        self.params = params

    def forward(self, particle_states: Tensor, odometry: Tensor) -> Tensor:

        translation_std = self.params.transition_std[0] / self.params.map_pixel_in_meters  # in pixels
        rotation_std = self.params.transition_std[1]  # in radians

        part_x, part_y, part_th = torch.unbind(particle_states, dim=-1)

        odometry = odometry.unsqueeze(1)
        odom_x, odom_y, odom_th = torch.unbind(odometry, dim=-1)

        noise_th = torch.normal(mean=0.0, std=1.0, size=part_th.shape) * rotation_std
        if not self.params.use_cpu:
            noise_th = noise_th.cuda()

        # add orientation noise before translation
        part_th = part_th + noise_th

        cos_th = torch.cos(part_th)
        sin_th = torch.sin(part_th)
        delta_x = cos_th * odom_x - sin_th * odom_y
        delta_y = sin_th * odom_x + cos_th * odom_y
        delta_th = odom_th

        noise_x = torch.normal(mean=0.0, std=1.0, size=delta_x.shape) * translation_std
        noise_y = torch.normal(mean=0.0, std=1.0, size=delta_y.shape) * translation_std

        if not self.params.use_cpu:
            noise_x = noise_x.cuda()
            noise_y = noise_y.cuda()

        return torch.stack([part_x + delta_x + noise_x, part_y + delta_y + noise_y, part_th + delta_th], axis=-1)

class ObservationModel(nn.Module):

    def __init__(self):
        super(ObservationModel, self).__init__()

        block1_layers = [
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2, dilation=1, bias=True),
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=4, dilation=2, bias=True),
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=8, dilation=4, bias=True)
        ]
        self.block1 = nn.ModuleList(block1_layers)

        size = [384, 28, 28]
        block2_layers= [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(size),
            nn.ReLU()
        ]
        self.block2 = nn.ModuleList(block2_layers)

        size = [16, 14, 14]
        block3_layers= [
            nn.Conv2d(384, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(size),
            nn.ReLU()
        ]
        self.block3 = nn.ModuleList(block3_layers)

    def forward(self, observation: Tensor) -> Tensor:
        x = observation

        # block1
        convs = []
        for _, l in enumerate(self.block1):
            convs.append(l(x))
        x = torch.cat(convs, axis=1)

        # block2
        for _, l in enumerate(self.block2):
            x = l(x)

        # block3
        for _, l in enumerate(self.block3):
            x = l(x)

        # x = observation.cpu().detach().numpy()
        # x = tf.convert_to_tensor(x)
        # data_format = 'channels_last'
        # layer_i = 1
        # convs = [
        #         tf.keras.layers.Conv2D(
        #             128, (3, 3), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             128, (5, 5), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             64, (5, 5), activation=None, padding='same', data_format=data_format,
        #             dilation_rate=(2, 2), use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             64, (5, 5), activation=None, padding='same', data_format=data_format,
        #             dilation_rate=(4, 4), use_bias=True)(x),
        # ]
        # x = tf.concat(convs, axis=-1)
        # x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        #
        # x = tf.keras.layers.Conv2D(
        #     16, (3, 3), activation=None, padding='same', data_format=data_format,
        #         use_bias=True)(x)
        # x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # print(x.shape)

        return x # [batch_size, 16, 14, 14]

class MapModel(nn.Module):

    def __init__(self):
        super(MapModel, self).__init__()

        block1_layers = [
            nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=True),
            nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3, dilation=1, bias=True),
            nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=6, dilation=2, bias=True),
            nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=9, dilation=3, bias=True),
        ]
        self.block1 = nn.ModuleList(block1_layers)

        block2_layers= [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.block2 = nn.ModuleList(block2_layers)

        block3_layers= [
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(64, 4, kernel_size=5, stride=1, padding=2, dilation=1, bias=True),
        ]
        self.block3 = nn.ModuleList(block3_layers)

        size = [8, 14, 14]
        block4_layers= [
            nn.LayerNorm(size),
            nn.ReLU()
        ]
        self.block4 = nn.ModuleList(block4_layers)

    def forward(self, local_maps: Tensor) -> Tensor:
        x = local_maps

        # block1
        convs = []
        for _, l in enumerate(self.block1):
            convs.append(l(x))
        x = torch.cat(convs, axis=1)

        # block2
        for _, l in enumerate(self.block2):
            x = l(x)

        # block3
        convs = []
        for _, l in enumerate(self.block3):
            convs.append(l(x))
        x = torch.cat(convs, axis=1)

        # block4
        for _, l in enumerate(self.block4):
            x = l(x)

        # x = local_maps.cpu().detach().numpy()
        # x = np.transpose(x, (0, 2, 3, 1))
        # x = tf.convert_to_tensor(x)
        # data_format = 'channels_last'
        # layer_i = 1
        # convs = [
        #         tf.keras.layers.Conv2D(
        #             24, (3, 3), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             16, (5, 5), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             8, (7, 7), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             8, (7, 7), activation=None, padding='same', data_format=data_format,
        #             dilation_rate=(2, 2), use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             8, (7, 7), activation=None, padding='same', data_format=data_format,
        #             dilation_rate=(3, 3), use_bias=True)(x),
        # ]
        # x = tf.concat(convs, axis=-1)
        # x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        #
        # print(x.shape)
        # convs = [
        #         tf.keras.layers.Conv2D(
        #             4, (3, 3), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        #         tf.keras.layers.Conv2D(
        #             4, (5, 5), activation=None, padding='same', data_format=data_format,
        #             use_bias=True)(x),
        # ]
        # x = tf.concat(convs, axis=-1)
        # print(x.shape)
        return x # [batch_size*num_particles, 8, 14, 14]

class LikelihoodNet(nn.Module):

    def __init__(self):
        super(LikelihoodNet, self).__init__()

        output_size = (12, 12)

        block1_layers= [
            LocallyConnected2d(24, 8, output_size, kernel_size=3, stride=1, bias=True),
            nn.ReLU()
        ]
        self.block1 = nn.ModuleList(block1_layers)

        block2_layers= [
            nn.ZeroPad2d((1, 1, 1, 1)),
            LocallyConnected2d(24, 8, output_size, kernel_size=5, stride=1, bias=True),
            nn.ReLU()
        ]
        self.block2 = nn.ModuleList(block2_layers)

        block3_layers= [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        ]
        self.block3 = nn.ModuleList(block3_layers)

        in_channels = 16 * 5 * 5
        block4_layers= [
            nn.Linear(in_channels, 1, bias=True)
        ]
        self.block4 = nn.ModuleList(block4_layers)


    def forward(self, joint_features: Tensor) -> Tensor:
        x = joint_features
        total_samples = joint_features.shape[0]
        output_size = (12, 12)

        # block1
        x1 = x
        for _, l in enumerate(self.block1):
            x1 = l(x1)

        # block2
        x2 = x
        for _, l in enumerate(self.block2):
            x2 = l(x2)

        x = torch.cat([x1, x2], axis=1)

        # block3
        for _, l in enumerate(self.block3):
            x = l(x)

        # [batch_size*num_particles, 16, 5, 5]
        x = torch.reshape(x, (total_samples, -1))

        # block4
        for _, l in enumerate(self.block4):
            x = l(x)

        lik = x

        # x = joint_features.cpu().detach().numpy()
        # x = np.transpose(x, (0, 2, 3, 1))
        # x = tf.convert_to_tensor(x)
        # data_format = 'channels_last'
        # layer_i = 1
        #
        # # pad manually to match different kernel sizes
        # x_pad1 = tf.pad(x, paddings=tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]]))
        # convs = [
        #     tf.keras.layers.LocallyConnected2D(
        #         8, (3, 3), activation='relu', padding='valid', data_format=data_format,
        #         use_bias=True)(x),
        #     tf.keras.layers.LocallyConnected2D(
        #         8, (5, 5), activation='relu', padding='valid', data_format=data_format,
        #         use_bias=True)(x_pad1),
        # ]
        # x = tf.concat(convs, axis=-1)
        # x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        # x = tf.reshape(x, (total_samples, -1))
        # lik = tf.keras.layers.Dense(
        #         1, activation=None, use_bias=True)(x)
        # print(lik.shape)

        return lik # [batch_size*num_particles, 1]

# reference: https://discuss.pytorch.org/t/locally-connected-layers/26979/2
from torch.nn.modules.utils import _pair
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class SpatialTransformerNet(nn.Module):

    def __init__(self, params):
        super(SpatialTransformerNet, self).__init__()
        self.params = params

    def forward(self, particle_states: Tensor, global_maps: Tensor) -> Tensor:

        batch_size, num_particles = particle_states.shape[:2]
        total_samples = batch_size * num_particles
        flat_states = torch.reshape(particle_states, (total_samples, 3))

        zero = torch.full((total_samples, ), 0)
        one = torch.full((total_samples, ), 1)
        if not self.params.use_cpu:
            zero = zero.cuda()
            one = one.cuda()

        input_map_shape = global_maps.shape
        # affine transformation
        height_inverse = 1.0 / input_map_shape[2]
        width_inverse = 1.0 / input_map_shape[3]

        # 1. translate the global map s.t. the center is at the particle state
        translate_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0
        transm1 = torch.stack([one, zero, translate_x, zero, one, translate_y, zero, zero, one], axis=1)
        transm1 = torch.reshape(transm1, (total_samples, 3, 3))

        # 2. rotate the global map s.t. the oriantation matches the particle state
        # normalize orientations
        theta = normalize(flat_states[:, 2], isTensor=True)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        rotm = torch.stack([costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one], axis=1)
        rotm = torch.reshape(rotm, (total_samples, 3, 3))

        # 3. optional scale down the map
        window_scaler = 8
        scale_x = torch.full((total_samples, ), float(self.params.local_map_size[0] * window_scaler) * width_inverse)
        scale_y = torch.full((total_samples, ), float(self.params.local_map_size[1] * window_scaler) * height_inverse)
        if not self.params.use_cpu:
            scale_x = scale_x.cuda()
            scale_y = scale_y.cuda()
        scalem = torch.stack([scale_x, zero, zero, zero, scale_y, zero, zero, zero, one], axis=1)
        scalem = torch.reshape(scalem, (total_samples, 3, 3))

        # # 4, optional translate the local map s.t. the particle defines the bottom mid-point instead of the center
        # translate_y2 = torch.full((total_samples, ), -1.0)
        # transm2 = torch.stack([one, zero, zero, zero, one, translate_y2, zero, zero, one], axis=1)
        # transm2 = torch.reshape(transm2, (total_samples, 3, 3))

        # chain the transormation matrices
        transform_m = torch.matmul(transm1, rotm)   # translate and rotation
        transform_m = torch.matmul(transform_m, scalem) # scale
        # transform_m = torch.matmul(transform_m, transm2)

        # reshape to the format expected by the spatial transform network
        transform_m = torch.reshape(transform_m[:, :2], (batch_size, num_particles, 2, 3))

        output_list = []
        # iterate over num_particles
        for i in range(num_particles):
            grid_size = torch.Size((batch_size, input_map_shape[1], self.params.local_map_size[0], self.params.local_map_size[1]))
            grid = F.affine_grid(transform_m[:, i], grid_size, align_corners=False).float()
            local_map = F.grid_sample(global_maps, grid, align_corners=False)
            output_list.append(local_map)
        local_maps = torch.stack(output_list, axis=1)

        return local_maps # [batch_size, num_particles, 1, 28, 28]

class ResampleNet(nn.Module):

    def __init__(self, params):
        super(ResampleNet, self).__init__()
        self.params = params

    def forward(self, particle_states: Tensor, particle_weights: Tensor, alpha: int) -> Tensor:

        assert 0.0 < alpha <= 1.0
        batch_size, num_particles = particle_states.shape[:2]

        # normalize
        particle_weights = particle_weights - torch.logsumexp(particle_weights, dim=-1, keepdim=True)

        # construct uniform weights
        uniform_weights = torch.full((batch_size, num_particles), np.log(1.0/float(num_particles)))
        if not self.params.use_cpu:
            uniform_weights = uniform_weights.cuda()

        # build sampling distribution q(s) and update particle weights
        if alpha < 1.0:
            # soft resampling
            q_weights = torch.stack([particle_weights + np.log(alpha), uniform_weights + np.log(1.0-alpha)], axis=-1)
            q_weights = torch.logsumexp(q_weights, dim=-1, keepdim=False)
            q_weights = q_weights - torch.logsumexp(q_weights, dim=-1, keepdim=True)  # normalized
            particle_weights = particle_weights - q_weights  # this is unnormalized
        else:
            # hard resampling -> produces zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s) using q_weights
        m = torch.distributions.categorical.Categorical(logits=q_weights)
        idx = []
        # iterate over num_particles
        for _ in range(num_particles):
            sample = m.sample() #   [batch_size]
            idx.append(sample.unsqueeze(1))
        indices = torch.cat(idx, dim=-1)    #   [batch_size, num_particles]

        # index into particles
        helper = torch.arange(0, batch_size * num_particles, step=num_particles, dtype=torch.int64) # [batch_size]
        if not self.params.use_cpu:
            helper = helper.cuda()
        indices = indices + helper.unsqueeze(1)

        indices = torch.reshape(indices, (batch_size * num_particles, ))
        # gather new particle states based on indices
        particle_states = torch.reshape(particle_states, (batch_size * num_particles, 3))
        new_particle_states = particle_states[indices].view(batch_size, num_particles, 3)   # [batch_size, num_particles, 3]
        # gather new particle weights based on indices
        particle_weights = torch.reshape(particle_weights, (batch_size * num_particles, ))
        new_particle_weights = particle_weights[indices].view(batch_size, num_particles)    # [batch_size, num_particles]

        # uniform_weights = uniform_weights.cpu().detach().numpy()
        # uniform_weights = tf.convert_to_tensor(uniform_weights)
        # particle_states = particle_states.cpu().detach().numpy()
        # particle_states = tf.convert_to_tensor(particle_states)
        # particle_weights = particle_weights.cpu().detach().numpy()
        # particle_weights = tf.convert_to_tensor(particle_weights)
        # q_weights = tf.stack([particle_weights + np.log(alpha), uniform_weights + np.log(1.0-alpha)], axis=-1)
        # q_weights = tf.reduce_logsumexp(q_weights, axis=-1, keepdims=False)
        # indices = tf.cast(tf.random.categorical(q_weights, num_particles), tf.int32)
        #
        # helper = tf.range(0, batch_size*num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
        # indices = indices + tf.expand_dims(helper, axis=1)
        # particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
        # print(particle_states.shape, indices.shape)
        # particle_states = tf.gather(particle_states, indices=indices, axis=0)
        # print(particle_states.shape)

        return new_particle_states, new_particle_weights

class PFCell(nn.Module):
    def __init__(self, params):
        super(PFCell, self).__init__()

        self.params = params

        self.transition_model = TransitionModel(params)
        self.trans_map_model = SpatialTransformerNet(params)
        self.resample_model = ResampleNet(params)
        self.observation_model = ObservationModel()
        self.map_model = MapModel()
        self.likeli_net = LikelihoodNet()

    def forward(self, inputs, state):
        batch_size = self.params.batch_size
        num_particles = self.params.num_particles
        particle_states, particle_weights = state
        observation, odometry, global_maps = inputs

        # sanity check
        assert list(particle_states.shape) == [batch_size, num_particles, 3]
        assert list(particle_weights.shape) == [batch_size, num_particles]
        assert list(global_maps.shape) == [batch_size, 1, 3000, 3000]
        assert list(observation.shape) == [batch_size, 3, 56, 56]
        assert list(odometry.shape) == [batch_size, 3]

        # observation update
        lik = self.observation_update(global_maps, particle_states, observation)
        particle_weights += lik  # unnormalized

        # resample
        if self.params.resample:
            particle_states, particle_weights = self.resample(particle_states, particle_weights)

        # construct output before motion update
        outputs = particle_states, particle_weights

        # motion update
        particle_states = self.motion_update(particle_states, odometry)

        # construct new state
        state = particle_states, particle_weights

        return outputs, state

    def observation_update(self, global_maps, particle_states, observation):
        batch_size = self.params.batch_size
        num_particles = self.params.num_particles

        # [batch_size, K, 3], [batch_size, C, H, W]
        local_maps = self.trans_map_model(particle_states, global_maps)

        # flatten batch and particle dimensions
        local_maps = torch.reshape(local_maps, [batch_size * num_particles] + list(local_maps.shape[2:]))
        map_features = self.map_model(local_maps)

        # [batch_size, C, H, W]
        obs_features = self.observation_model(observation)

        # tile observation features and flatten batch and particle dimensions
        obs_features = obs_features.unsqueeze(1).repeat(1, num_particles, 1, 1, 1)
        obs_features = torch.reshape(obs_features, [batch_size * num_particles] + list(obs_features.shape[2:]))

        # merge features and process further
        joint_features = torch.cat([map_features, obs_features], axis=1)
        lik = self.likeli_net(joint_features)

        # [batch_size, num_particles] unflatten batch and particle dimensions
        lik = torch.reshape(lik, [batch_size, num_particles])

        return lik

    def resample(self, particle_states, particle_weights):

        # [batch_size, K, 3] [batch_size, K]
        new_particle_states, new_particle_weights = \
            self.resample_model(particle_states, particle_weights, self.params.alpha_resample_ratio)

        return new_particle_states, new_particle_weights

    def motion_update(self, particle_states, odometry):

        # [batch_size, 3]
        particle_states = self.transition_model(particle_states, odometry)

        return particle_states

# reference https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    params = argparser.parse_args()
    params.num_epochs = 1
    params.batch_size = 2
    params.num_particles = 30
    params.trajlen = 24
    params.map_pixel_in_meters = 0.02
    params.init_particles_distr = 'gaussian'
    params.init_particles_std = [0.3, 0.523599]  # 30cm, 30degrees
    params.transition_std = [0, 0]
    params.local_map_size = (28, 28)
    params.seed = 42
    params.type = 'valid'
    params.alpha_resample_ratio = 0.5
    params.resample = True

    params.device = torch.device('cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # tensorflow

    file = '../data/valid.tfrecords'

    composed = transforms.Compose([
                ToTensor(),
    ])
    house_dataset = House3DTrajDataset(params, file, transform=composed)
    house_data_loader = DataLoader(house_dataset, batch_size=params.batch_size, shuffle=True, num_workers=0)

    transition_model = TransitionModel(params)
    observation_model = ObservationModel()
    resample_model = ResampleNet(params)
    trans_map_model = SpatialTransformerNet(params)
    map_model = MapModel()
    likeli_net = LikelihoodNet()

    for _, batch_samples in enumerate(house_data_loader):
        # print(
        #     batch_samples['true_states'].shape, \
        #     batch_samples['odometry'].shape, \
        #     batch_samples['init_particles'].shape, \
        #     batch_samples['observation'].shape, \
        #     batch_samples['global_map'].shape
        # )
        observations = batch_samples['observation']
        particle_states = batch_samples['init_particles']
        global_maps = batch_samples['global_map']
        labels = batch_samples['true_states']
        odometries = batch_samples['odometry']
        particle_weights = torch.full((params.batch_size, params.num_particles), \
                np.log(1.0/float(params.num_particles)))

        batch_size, trajlen = observations.shape[:2]
        num_particles = particle_states.shape[1]
        for traj in range(trajlen):

            # visualize local map

            particle_state = labels[0, 0]
            particle_state[2] = 0
            global_map = global_maps[0, 0]

            local_maps = trans_map_model(particle_state.unsqueeze(0).unsqueeze(0), global_map.unsqueeze(0).unsqueeze(0))
            local_map = local_maps[0, 0, 0]

            x, y, _ = particle_state
            pose_plt = Wedge((x, y), 10, 0, 360, color='red', alpha=.75)
            plt_ax.add_artist(pose_plt)
            map_plt = plt_ax.imshow(global_map)

            # x, y = global_map.shape[0]/2, global_map.shape[1]/2
            # pose_plt = Wedge((x, y), 10, 0, 360, color='red', alpha=.75)
            # plt_ax.add_artist(pose_plt)
            # map_plt = plt_ax.imshow(local_map)

            plt.show()

            # observation update

            # [batch_size, K, 3], [batch_size, C, H, W]
            local_maps = trans_map_model(particle_states, global_maps)

            # flatten batch and particle dimensions
            local_maps = torch.reshape(local_maps, [batch_size * num_particles] + list(local_maps.shape[2:]))
            map_features = map_model(local_maps)

            # [batch_size, C, H, W]
            obs = observations[:, traj, :, :, :]
            obs_features = observation_model(obs)

            # tile observation features and flatten batch and particle dimensions
            obs_features = obs_features.unsqueeze(1).repeat(1, num_particles, 1, 1, 1)
            obs_features = torch.reshape(obs_features, [batch_size * num_particles] + list(obs_features.shape[2:]))

            # merge features and process further
            joint_features = torch.cat([map_features, obs_features], axis=1)
            lik = likeli_net(joint_features)

            # [batch_size, num_particles] unflatten batch and particle dimensions
            lik = torch.reshape(lik, [batch_size, num_particles])
            particle_weights += lik  # unnormalized

            # resample
            particle_states, particle_weights = resample_model(particle_states, particle_weights, params.alpha_resample_ratio)

            # motion update

            # [batch_size, 3]
            odometry = odometries[:, traj, :]
            particle_states = transition_model(particle_states, odometry)
            print(particle_states.shape)

            break
        break
    print('done')

    # idx = np.random.randint(0, valid_data_size)
    # # idx = 751
    # print(idx)
    # house_dataset.display_data(idx)
