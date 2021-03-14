#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

from utils import render, datautils, arguments, pfnet_loss
from gibson2.utils.assets_utils import get_scene_path
from gibson2.envs.igibson_env import iGibsonEnv
import tensorflow as tf
import torch.nn as nn
from PIL import Image
import pybullet as p
import numpy as np
import torch
import cv2
import gym
import os

# set programatically the path to 'pfnet' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/pfnet')
import pfnet

class LocalizeGibsonEnv(iGibsonEnv):

    def __init__(self, params):

        self.params = params
        super(LocalizeGibsonEnv, self).__init__(config_file=self.params.config_filename,
                        scene_id=None,
                        mode=self.params.mode,
                        action_timestep=1/10.0,
                        physics_timestep=1/240.0,
                        device_idx=self.params.gpu_num,
                        render_to_tensor=False,
                        automatic_reset=False)

        output_size = 18 + np.prod((56, 56, 3))

        self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(output_size, ),
                dtype=np.float32)

        # create pf model
        self.pfnet_model = pfnet.pfnet_model(self.params)

        # load model from checkpoint file
        if self.params.pfnet_load:
            print("=====> Loading pf model from " + self.params.pfnet_load)
            self.pfnet_model.load_weights(self.params.pfnet_load)

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        super(LocalizeGibsonEnv, self).load_miscellaneous_variables()

        self.obstacle_map = None
        self.pfnet_state = None
        self.floor_map = None
        self.old_pose = None
        self.old_obs = None

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        super(LocalizeGibsonEnv, self).reset_variables()

        self.obstacle_map = None
        self.pfnet_state = None
        self.floor_map = None
        self.old_pose = None
        self.old_obs = None

    def step(self, action):
        state, reward, done, info = super(LocalizeGibsonEnv, self).step(action)

        trajlen = 1
        batch_size = 1

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])

        floor_map = self.floor_map[0]
        robot_state = self.robots[0].calc_state()
        true_pose = self.get_robot_pose(robot_state, floor_map.shape)

        # calculate actual odometry b/w old pose and new pose
        old_pose = self.old_pose[0].numpy()
        odom = datautils.calc_odometry(old_pose, true_pose)

        obs = tf.expand_dims(
                    tf.convert_to_tensor(rgb, dtype=tf.float32)
                    , axis=0)
        odom = tf.expand_dims(
                    tf.convert_to_tensor(odom, dtype=tf.float32)
                    , axis=0)
        true_pose = tf.expand_dims(
                        tf.convert_to_tensor(true_pose, dtype=tf.float32)
                        , axis=0)

        # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
        # if non-stateful: pass the state explicity every step
        if self.params.stateful:
            self.pfnet_model.layers[-1].reset_states(state)    # RNN layer

        observation = tf.expand_dims(obs, axis=1)
        odometry = tf.expand_dims(odom, axis=1)
        # sanity check
        assert list(odometry.shape) == [batch_size, trajlen, 3]
        assert list(observation.shape) == [batch_size, trajlen, 56, 56, 3]

        input = [observation, odometry]
        model_input = (input, self.pfnet_state)

        # forward pass
        output, state = self.pfnet_model(model_input, training=False)

        self.pfnet_state = state
        self.old_pose = true_pose
        self.old_obs = obs

        custom_state = np.concatenate([robot_state, np.reshape(rgb, [-1])], 0)
        return custom_state, reward, done, info

    def reset(self):
        state = super(LocalizeGibsonEnv, self).reset()

        batch_size = 1
        map_size = params.global_map_size
        num_particles = self.params.num_particles
        particles_cov = self.params.init_particles_cov
        particles_distr = self.params.init_particles_distr

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])

        floor_map = self.get_floor_map()
        obstacle_map = self.get_obstacle_map()
        robot_state = self.robots[0].calc_state()
        true_pose = self.get_robot_pose(robot_state, floor_map.shape)

        obs = tf.expand_dims(
                    tf.convert_to_tensor(rgb, dtype=tf.float32)
                    , axis=0)
        true_pose = tf.expand_dims(
                        tf.convert_to_tensor(true_pose, dtype=tf.float32)
                        , axis=0)
        floor_map = tf.expand_dims(
                        tf.convert_to_tensor(floor_map, dtype=tf.float32)
                    , axis=0)
        obstacle_map = tf.expand_dims(
                        tf.convert_to_tensor(obstacle_map, dtype=tf.float32)
                    , axis=0)
        init_particles = tf.convert_to_tensor(
                            self.get_random_particles(
                                    num_particles,
                                    particles_distr,
                                    true_pose,
                                    floor_map,
                                    particles_cov)
                            , dtype=tf.float32)
        init_particle_weights = tf.constant(
                                    np.log(1.0/float(num_particles)),
                                    shape=(batch_size, num_particles),
                                    dtype=tf.float32)

        # sanity check
        assert list(true_pose.shape) == [batch_size, 3]
        assert list(obs.shape) == [batch_size, 56, 56, 3]
        assert list(init_particles.shape) == [batch_size, num_particles, 3]
        assert list(init_particle_weights.shape) == [batch_size, num_particles]
        assert list(floor_map.shape) == [batch_size, map_size[0], map_size[1], map_size[2]]
        assert list(obstacle_map.shape) == [batch_size, map_size[0], map_size[1], map_size[2]]

        self.pfnet_state = [init_particles, init_particle_weights, obstacle_map]
        self.obstacle_map = obstacle_map
        self.floor_map = floor_map
        self.old_pose = true_pose
        self.old_obs = obs

        custom_state = np.concatenate([robot_state, np.reshape(rgb, [-1])], 0)
        return custom_state

    def get_robot_pose(self, robot_state, floor_map_shape):
        robot_pos = robot_state[0:3]    # [x, y, z]
        robot_orn = robot_state[3:6]    # [r, p, y]

        # transform from co-ordinate space to pixel space
        robot_pos_xy = datautils.transform_pose(robot_pos[:2], floor_map_shape, self.scene.trav_map_resolution**2)  # [x, y]
        robot_pose = np.array([robot_pos_xy[0], robot_pos_xy[1], robot_orn[2]])  # [x, y, theta]

        return robot_pose

    def get_obstacle_map(self):
        """
        Get the scene obstacle map
        """
        obstacle_map = np.array(Image.open(
                    os.path.join(get_scene_path(self.config.get('scene_id')),
                            f'floor_{self.task.floor_num}.png')
                ))

        # process image for training
        obstacle_map = datautils.process_floor_map(obstacle_map)

        return obstacle_map

    def get_floor_map(self):
        """
        Get the scene floor map (traversability map + obstacle map)
        :return ndarray: floor map of current scene (H, W, 1)
        """

        obstacle_map = np.array(Image.open(
                    os.path.join(get_scene_path(self.config.get('scene_id')),
                            f'floor_{self.task.floor_num}.png')
                ))

        trav_map = np.array(Image.open(
                    os.path.join(get_scene_path(self.config.get('scene_id')),
                            f'floor_trav_{self.task.floor_num}.png')
                ))

        trav_map[obstacle_map == 0] = 0

        trav_map_erosion=self.config.get('trav_map_erosion', 2)
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
        trav_map[trav_map < 255] = 0

        # process image for training
        floor_map = datautils.process_floor_map(trav_map)

        return floor_map

    def get_random_particles(self, num_particles, particles_distr, robot_pose, scene_map, particles_cov):
        """
        Sample random particles based on the scene
        :param particles_distr: string type of distribution, possible value: [gaussian, uniform]
        :param robot_pose: ndarray indicating the robot pose ([batch_size], 3) in pixel space
            if None, random particle poses are sampled using unifrom distribution
            otherwise, sampled using gaussian distribution around the robot_pose
        :param particles_cov: for tracking Gaussian covariance matrix (3, 3)
        :param num_particles: integer indicating the number of random particles per batch
        :return ndarray: random particle poses  (batch_size, num_particles, 3) in pixel space
        """

        if robot_pose.ndim == 1:
            robot_pose = np.expand_dims(robot_pose, axis=0)

        assert list(robot_pose.shape)[1] == 3
        assert list(particles_cov.shape) == [3, 3]

        particles = []
        batches = robot_pose.shape[0]
        if particles_distr == 'uniform':
            # iterate per batch_size
            for b_idx in range(batches):
                sample_i = 0
                b_particles = []

                # get bounding box for more efficient sampling
                # rmin, rmax, cmin, cmax = self.bounding_box(self.floor_map)
                rmin, rmax, cmin, cmax = self.bounding_box(scene_map, robot_pose[b_idx])

                while sample_i < num_particles:
                    particle = np.random.uniform(low=(cmin, rmin, 0.0), high=(cmax, rmax, 2.0*np.pi), size=(3, ))
                    # reject if mask is zero
                    if not scene_map[int(np.rint(particle[1])), int(np.rint(particle[0]))]:
                        continue
                    b_particles.append(particle)

                    sample_i = sample_i + 1
                particles.append(b_particles)
        elif particles_distr == 'gaussian':
            # iterate per batch_size
            for b_idx in range(batches):
                # sample offset from the Gaussian
                center = np.random.multivariate_normal(mean=robot_pose[b_idx], cov=particles_cov)

                # sample particles from the Gaussian, centered around the offset
                particles.append(np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles))
        else:
            raise ValueError

        particles = np.stack(particles) # [batch_size, num_particles, 3]
        return particles

    def bounding_box(self, img, robot_pose=None, lmt=100):
        """
        Bounding box of non-zeros in an array.
        :param img: numpy array
        :param robot_pose: numpy array of robot pose
        :param lmt: integer representing width/length of bounding box
        :return (int, int, int, int): bounding box indices top_row, bottom_row, left_column, right_column
        """
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if robot_pose is not None:
            # futher constraint the bounding box
            x, y, _ = robot_pose

            rmin = np.rint(y-lmt) if (y-lmt) > rmin else rmin
            rmax = np.rint(y+lmt) if (y+lmt) < rmax else rmax
            cmin = np.rint(x-lmt) if (x-lmt) > cmin else cmin
            cmax = np.rint(x+lmt) if (x+lmt) < cmax else cmax

        return rmin, rmax, cmin, cmax

def train_localization_rl(params):
    """
    train rl agent for localization in iGibsonEnv with the parsed arguments
    """

    # create gym env
    env = LocalizeGibsonEnv(params)
    env.reset()
    env.seed(params.seed)
    env.action_space.np_random.seed(params.seed)

    for _ in range(params.trajlen-1):
        if params.agent == 'manual':
            action = datautils.get_discrete_action()
        else:
            # default random action
            action = env.action_space.sample()

        # take action and get new observation
        obs, reward, done, _ = env.step(action)

if __name__ == '__main__':
    params = arguments.parse_args()
    train_localization_rl(params)
