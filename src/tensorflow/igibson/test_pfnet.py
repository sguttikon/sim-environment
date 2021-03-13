#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

from utils import render, datautils, arguments, pfnet_loss
from utils.iGibson_env import iGibsonEnv
import tensorflow as tf
import numpy as np
import glob

# set programatically the path to 'pfnet' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/pfnet')
import pfnet

def testing(params):
    """
    testing pfnet-igibson with the parsed arguments
    """

    num_test_batches = 1
    trajlen = params.trajlen
    bptt_steps = params.bptt_steps
    batch_size = params.batch_size
    num_particles = params.num_particles

    # evaluation data
    filenames = list(glob.glob(params.testfiles[0]))
    test_ds = datautils.get_dataflow(filenames, params.batch_size, is_training=True)
    print(f'test data: {filenames}')

    # create gym env
    env = iGibsonEnv(config_file=params.config_filename, mode=params.mode,
                action_timestep=1 / 10.0, physics_timestep=1 / 240.0,
                device_idx=params.gpu_num, max_step=params.max_step)
    env.reset()
    print("=====> iGibsonEnv initialized")

    # create pf model
    pfnet_model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.pfnet_load:
        pfnet_model.load_weights(params.pfnet_load)
        print("=====> Loaded pf model from " + params.pfnet_load)

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=params.learningrate)

    itr = test_ds.as_numpy_iterator()
    # run over all evaluation samples in an epoch
    for idx in range(num_test_batches):
        parsed_record = next(itr)
        batch_sample = datautils.transform_raw_record(env, parsed_record, params)

        observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
        odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
        true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
        floor_map = tf.convert_to_tensor(batch_sample['floor_map'], dtype=tf.float32)
        obstacle_map = tf.convert_to_tensor(batch_sample['obstacle_map'], dtype=tf.float32)
        init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
        init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                    shape=(batch_size, num_particles), dtype=tf.float32)#

        # enable auto-differentiation
        with tf.GradientTape() as tape:
            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, obstacle_map]

            particle_states = []
            particle_weights = []
            for idx in np.arange(0, trajlen, bptt_steps):
                # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                # if non-stateful: pass the state explicity every step
                if params.stateful:
                    pfnet_model.layers[-1].reset_states(state)    # RNN layer

                obs = observation[:, idx:idx+bptt_steps]
                odom = odometry[:, idx:idx+bptt_steps]
                # sanity check
                assert list(obs.shape) == [batch_size, bptt_steps, 56, 56, 3]
                assert list(odom.shape) == [batch_size, bptt_steps, 3]

                input = [obs, odom]
                model_input = (input, state)

                # forward pass
                output, state = pfnet_model(model_input, training=False)

                particle_states.append(output[0])
                particle_weights.append(output[1])

            # print(state[0], state[1])
            # print(particle_states, particle_weights)

            # compute loss
            particle_states = tf.concat(particle_states, axis=1)    # [batch_size, trajlen, num_particles, 3]
            particle_weights = tf.concat(particle_weights, axis=1)  # [batch_size, trajlen, num_particles]

            # sanity check
            assert list(particle_states.shape) == [batch_size, trajlen, num_particles, 3]
            assert list(particle_weights.shape) == [batch_size, trajlen, num_particles]
            assert list(true_states.shape) == [batch_size, trajlen, 3]

            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
            loss_pred = loss_dict['pred']
            print(loss_dict)

        # # compute gradients of the trainable variables with respect to the loss
        # gradients = tape.gradient(loss_pred, pfnet_model.trainable_weights)
        # gradients = list(zip(gradients, pfnet_model.trainable_weights))
        #
        # # run one step of gradient descent
        # optimizer.apply_gradients(gradients)

if __name__ == '__main__':
    params = arguments.parse_args()
    testing(params)
