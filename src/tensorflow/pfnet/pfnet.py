#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
from utils import networks
from tensorflow import keras
from utils.spatial_transformer import transformer

class PFCell(keras.layers.AbstractRNNCell):
    """
    PF-Net custom implementation for localization with RNN interface
    Implements the particle set update: observation, tramsition models and soft-resampling

    Cell inputs: observation, odometry
    Cell states: particle_states, particle_weights
    Cell outputs: particle_states, particle_weights (updated)
    """
    def __init__(self, params, **kwargs):
        """
        :param params: parsed arguments
        """
        self.batch_size = params.batch_size
        self.num_particles = params.num_particles
        self.transition_std = params.transition_std
        self.map_pixel_in_meters = params.map_pixel_in_meters

        self.states_shape = (self.batch_size, self.num_particles, 3)
        self.weights_shape = (self.batch_size, self.num_particles)
        super(PFCell, self).__init__(**kwargs)

        # models
        self.obs_model = networks.obs_encoder()
        self.map_model = networks.map_encoder()
        self.joint_matrix_model = networks.map_obs_encoder()
        self.joint_vector_model = networks.likelihood_estimator()

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:])]

    @property
    def output_size(self):
        """
        Size(s) of output(s) produced by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:])]

    def call(self, input, state):
        """
        Implements a particle update
        :param input: observation (batch, 56, 56, ch), odometry (batch, 3), global_map (batch, H, W, 1)
            observation is the sensor reading at time t,
            odometry is the relative motion from time t to t+1,
            global map of environment
        :param state: particle_states (batch, k, 3), particle_weights (batch, k)
            weights are assumed to be in log space and unnormalized
        :return output: particle_states and particle_weights after the observation update.
            (but before the transition update)
        :return state: updated particle_states and particle_weights.
            (but after both observation and transition updates)
        """
        particle_states, particle_weights = state
        observation, odometry, global_map = input

        # observation update
        lik = self.observation_update(
                    global_map, particle_states, observation
        )
        particle_weights = particle_weights + lik # unnormalized

        # resample
        # TODO

        # construct output before motion update
        output = [particle_states, particle_weights]

        # motion update which affect the particle state input at next step
        particle_states = self.transition_model(particle_states, odometry)

        # construct new state after motion update
        state = [particle_states, particle_weights]

        return output, state

    def observation_update(self, global_map, particle_states, observation):
        """
        Implements a discriminative observation model for localization
        The model transforms global map to local maps for each particle,
        where a local map is a local view from state defined by the particle.
        :param global_map: global map input (batch, None, None, ch)
            assumes range[0, 2] were 0: occupied and 2: free space
        :param particle_states: particle states before observation update (batch, k, 3)
        :param observation: image observation (batch, 56, 56, ch)
        :return (batch, k): particle likelihoods in the log space (unnormalized)
        """

        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # transform global maps to local maps
        local_maps = self.transform_maps(global_map, particle_states, (28, 28))

        # rescale from [0, 2] to [-1, 1]    -> optional
        local_maps = -(local_maps - 1)

        # flatten batch and particle dimensions
        local_maps = tf.reshape(local_maps,
                [batch_size * num_particles] + local_maps.shape.as_list()[2:])

        # get features from local maps
        map_features = self.map_model(local_maps)

        # get features from observation
        obs_features = self.obs_model(observation)

        # tile observation features
        obs_features = tf.tile(tf.expand_dims(obs_features, axis=1), [1, num_particles, 1, 1, 1])
        obs_features = tf.reshape(obs_features,
                [batch_size * num_particles] + obs_features.shape.as_list()[2:])

        # sanity check
        assert obs_features.shape.as_list()[:-1] == map_features.shape.as_list()[:-1]

        # merge map and observation features
        joint_features = tf.concat([map_features, obs_features], axis=-1)
        joint_features = self.joint_matrix_model(joint_features)

        # reshape to a vector
        joint_features = tf.reshape(joint_features, [batch_size * num_particles, -1])
        lik = self.joint_vector_model(joint_features)
        lik = tf.reshape(lik, [batch_size, num_particles])

        return lik

    def resample(particle_states, particle_weights, alpha):
        # TODO
        pass

    def transition_model(self, particle_states, odometry):
        """
        Implements a stochastic transition model for localization
        :param particle_states: particle states before motion update (batch, k, 3)
        :param odometry: odometry reading - relative motion in robot coordinate frame (batch, 3)
        :return (batch, k, 3): particle states updated with the odometry and optionally transition noise
        """

        translation_std = self.transition_std[0] / self.map_pixel_in_meters   # in pixels
        rotation_std = self.transition_std[1]

        part_x, part_y, part_th = tf.unstack(particle_states, axis=-1, num=3)   # (batch_size, num_particles, 3)

        odometry = tf.expand_dims(odometry, axis=1) # (batch_size, 1, 3)
        odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

        # sample noisy orientation
        noise_th = tf.random.normal(part_th.get_shape(), mean=0.0, stddev=1.0) * rotation_std

        # add orientation noise before translation
        part_th = part_th + noise_th

        cos_th = tf.cos(part_th)
        sin_th = tf.sin(part_th)
        delta_x = cos_th * odom_x - sin_th * odom_y
        delta_y = sin_th * odom_x + cos_th * odom_y
        delta_th = odom_th

        # sample noisy translation
        delta_x = delta_x + tf.random.normal(part_th.get_shape(), mean=0.0, stddev=1.0) * translation_std
        delta_y = delta_y + tf.random.normal(part_th.get_shape(), mean=0.0, stddev=1.0) * translation_std

        return tf.stack([part_x + delta_x , part_y + delta_y, part_th + delta_th], axis=-1)   # (batch_size, num_particles, 3)

    def transform_maps(self, global_map, particle_states, local_map_size):
        """
        Implements global to local map transformation
        :param global_map: global map input (batch, None, None, ch)
        :param particle_states: particle states that define local view for transformation (batch, k, 3)
        :param local_map_size: size of output local maps (height, width)
        :return (batch, k, local_map_size[0], local_map_size[1], ch): each local map shows different transformation
            of global map corresponding to particle state
        """

        # flatten batch and particle
        batch_size, num_particles = particle_states.shape.as_list()[:2]
        total_samples = batch_size * num_particles
        flat_states = tf.reshape(particle_states, [total_samples, 3])

        # define variables
        input_shape = tf.shape(global_map)
        global_height = tf.cast(input_shape[1], tf.float32)
        global_width = tf.cast(input_shape[2], tf.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width
        zero = tf.constant(0, dtype=tf.float32, shape=(total_samples, ))
        one = tf.constant(1, dtype=tf.float32, shape=(total_samples, ))

        window_scaler = 8.0 # global map will be down-scaled by some factor

        # normalize orientations and precompute cos and sin functions
        theta = -flat_states[:, 2] - 0.5 * np.pi
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

        # construct affine transformation matrix step-by-step

        # 1: translate the global map s.t. the center is at the particle state
        translate_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0

        transm1 = tf.stack((one, zero, translate_x, zero, one, translate_y, zero, zero, one), axis=1)
        transm1 = tf.reshape(transm1, [total_samples, 3, 3])

        # 2: rotate map s.t the orientation matches that of the particles
        rotm = tf.stack((costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one), axis=1)
        rotm = tf.reshape(rotm, [total_samples, 3, 3])

        # 3: scale down the map
        scale_x = tf.fill((total_samples, ), float(local_map_size[1] * window_scaler) * width_inverse)
        scale_y = tf.fill((total_samples, ), float(local_map_size[0] * window_scaler) * height_inverse)

        scalem = tf.stack((scale_x, zero, zero, zero, scale_y, zero, zero, zero, one), axis=1)
        scalem = tf.reshape(scalem, [total_samples, 3, 3])

        # 4: translate the local map s.t. the particle defines the bottom mid_point instead of the center
        translate_y2 = tf.constant(-1.0, dtype=tf.float32, shape=(total_samples, ))

        transm2 = tf.stack((one, zero, zero, zero, one, translate_y2, zero, zero, one), axis=1)
        transm2 = tf.reshape(transm2, [total_samples, 3, 3])

        # finally chain all traformation matrices into single one
        transform_m = tf.matmul(tf.matmul(tf.matmul(transm1, rotm), scalem), transm2)

        # reshape to format expected by spatial transform network
        transform_m = tf.reshape(transform_m[:, :2], [batch_size, num_particles, 6])

        # iterate over num_particles to tranform image using spatial transform network
        list = []
        for i in range(num_particles):
            list.append(transformer(global_map, transform_m[:, i], local_map_size))
        local_maps = tf.stack(list, axis=1)

        # reshape if any information has lost in spatial transform network
        local_maps = tf.reshape(local_maps,
            [batch_size, num_particles, local_map_size[0], local_map_size[1], global_map.shape.as_list()[-1]])

        return local_maps   # (batch_size, num_particles, 28, 28, 1)

def pfnet_model(params):

    batch_size = params.batch_size
    num_particles = params.num_particles
    observation = keras.Input(shape=[None, 56, 56, 3], batch_size=batch_size)   # (bs, T, 56, 56, 3)
    odometry = keras.Input(shape=[None, 3], batch_size=batch_size)    # (bs, T, 3)
    global_map = keras.Input(shape=[None, None, None, 1], batch_size=batch_size)   # (bs, T, H, W, 1)

    particle_states = keras.Input(shape=[num_particles, 3], batch_size=batch_size)   # (bs, k, 3)
    particle_weights = keras.Input(shape=[num_particles], batch_size=batch_size)    # (bs, k)

    cell = PFCell(params)
    rnn = keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=False)

    state = [particle_states, particle_weights]
    input = (observation, odometry, global_map)
    x = rnn(inputs=input, initial_state=state)
    output, state = x[:2], x[2:]

    return keras.Model(
        inputs=([observation, odometry, global_map], [particle_states, particle_weights]),
        outputs=([output, state])
    )

if __name__ == '__main__':
    # obs_model = observation_model()
    # keras.utils.plot_model(obs_model, to_file='obs_model.png', show_shapes=True, dpi=64)
    #
    # observations = np.random.random((8*10, 56, 56, 3))
    # obs_out = obs_model(observations)
    # print(obs_out.shape)
    #
    # map_model = map_model()
    # keras.utils.plot_model(map_model, to_file='map_model.png', show_shapes=True, dpi=64)
    #
    # local_maps = np.random.random((8*10, 28, 28, 1))
    # map_out = map_model(local_maps)
    # print(map_out.shape)
    #
    # joint_matrix_model = joint_matrix_model()
    # keras.utils.plot_model(joint_matrix_model, to_file='joint_matrix_model.png', show_shapes=True, dpi=64)
    #
    # joint_features = tf.concat([map_out, obs_out], axis=-1)
    # joint_matrix_out = joint_matrix_model(joint_features)
    # print(joint_matrix_out.shape)
    #
    # joint_vector_model = joint_vector_model()
    # keras.utils.plot_model(joint_vector_model, to_file='joint_vector_model.png', show_shapes=True, dpi=64)
    #
    # joint_matrix_out = tf.reshape(joint_matrix_out, (8 * 10, -1))
    # joint_vector_out = joint_vector_model(joint_matrix_out)
    # joint_vector_out = tf.reshape(joint_vector_out, [8, 10])
    # print(joint_vector_out.shape)
    #
    # particle_states = tf.random.uniform((8, 10, 3))
    # odometry = tf.random.uniform((8, 3))
    # transition_std = np.array([0.0, 0.0])
    # map_pixel_in_meters = 0.02
    # transition_out = transition_model(particle_states, odometry, (8, 10), transition_std, map_pixel_in_meters)
    # print(transition_out.shape)
    #
    # global_maps = tf.random.uniform((8, 300, 300, 1))
    # transform_out = transform_maps(global_maps, particle_states, (8, 10), (28, 28))
    # print(transform_out.shape)

    argparser = argparse.ArgumentParser()
    params = argparser.parse_args()

    params.transition_std = np.array([0.0, 0.0])
    params.map_pixel_in_meters = 0.02
    params.batch_size = 8
    params.num_particles = 30
    params.time_steps = 5

    model = pfnet_model(params)
    keras.utils.plot_model(model, to_file='pfnet_model.png', show_shapes=True, dpi=64)

    particle_states = tf.random.uniform((params.batch_size, params.num_particles, 3))
    particle_weights = tf.random.uniform((params.batch_size, params.num_particles))
    observation = tf.random.uniform((params.batch_size, params.time_steps, 56, 56, 3))
    odometry = tf.random.uniform((params.batch_size, params.time_steps, 3))
    global_map = tf.random.uniform((params.batch_size, params.time_steps, 100, 100, 1))
    inputs = ([observation, odometry, global_map], [particle_states, particle_weights])
    output, state = model(inputs)
    print(output[0].shape, output[1].shape, state[0].shape, state[1].shape)

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')

    # Create a new model instance
    new_model = pfnet_model(params)

    # Restore the weights
    new_model.load_weights('./checkpoints/my_checkpoint')
