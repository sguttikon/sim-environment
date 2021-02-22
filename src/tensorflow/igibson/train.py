#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
from utils import datautils, arguments, pfnet_loss
from utils.iGibson_env import iGibsonEnv

# set programatically the path to 'pfnet' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/pfnet')
# set_path('/home/guttikon/awesome_robotics/sim-environment/src/tensorflow/pfnet')

import os
import pfnet
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

def train_dataset_size():
    return 800

def valid_dataset_size():
    return 800

def run_training(params):
    """
    run training with the parsed arguments
    """

    old_stdout = sys.stdout
    log_file = open(params.output,'w')
    sys.stdout = log_file

    trajlen = params.trajlen
    batch_size = params.batch_size
    num_particles = params.num_particles
    num_train_batches = train_dataset_size() // batch_size
    num_valid_batches = valid_dataset_size() // batch_size

    # create gym env
    env = iGibsonEnv(config_file=params.config_filename, mode=params.mode,
                action_timestep=1 / 10.0, physics_timestep=1 / 240.0,
                device_idx=params.gpu_num, max_step=params.max_step)

    # create pf model
    pfnet_model = pfnet.pfnet_model(params)

    # get pretrained action model
    if params.agent == 'pretrained' and params.action_load:
        print("=====> Loading action sampler from " + params.action_load)
        action_model = datautils.load_action_model(env, params.gpu_num, params.action_load)
    else:
        action_model = None

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=params.learningrate)

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = keras.metrics.Mean('test_loss', dtype=tf.float32)

    train_summary_writer = tf.summary.create_file_writer(params.train_log_dir + 'gradient_tape/')
    test_summary_writer = tf.summary.create_file_writer(params.test_log_dir + 'gradient_tape/')

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        # run training over all training samples in an epoch
        for idx in tqdm(range(num_train_batches)):
            batch_sample = datautils.get_batch_data(env, params, action_model)

            odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
            global_map = tf.convert_to_tensor(batch_sample['global_map'], dtype=tf.float32)
            observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
            init_particles_weights = tf.convert_to_tensor(batch_sample['init_particles_weights'], dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particles_weights, global_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if params.stateful:
                pfnet_model.layers[-1].reset_states(state)    # RNN layer

            # run training over trajectory
            input = [observation, odometry]
            model_input = (input, state)

            # enable auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                output, state = pfnet_model(model_input, training=True)

                # compute loss
                particle_states, particle_weights = output
                loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, 1)
                loss_pred = loss_dict['pred']

            # compute gradients of the trainable variables with respect to the loss
            gradients = tape.gradient(loss_pred, pfnet_model.trainable_weights)
            gradients = list(zip(gradients, pfnet_model.trainable_weights))

            # run one step of gradient descent
            optimizer.apply_gradients(gradients)
            train_loss(loss_pred)  # overall trajectory loss

        # log epoch training stats
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        # Save the weights
        print("=====> saving trained model ")
        pfnet_model.save_weights(params.train_log_dir + f'/chks/checkpoint_{epoch}_{train_loss.result():03.3f}/pfnet_checkpoint')

        if params.run_validation:
            # run validation over all validation samples in an epoch
            for idx in tqdm(range(num_valid_batches)):
                batch_sample = datautils.get_batch_data(env, params, action_model)

                odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
                global_map = tf.convert_to_tensor(batch_sample['global_map'], dtype=tf.float32)
                observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
                true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
                init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
                init_particles_weights = tf.convert_to_tensor(batch_sample['init_particles_weights'], dtype=tf.float32)

                # start trajectory with initial particles and weights
                state = [init_particles, init_particles_weights, global_map]

                # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                # if non-stateful: pass the state explicity every step
                if params.stateful:
                    pfnet_model.layers[-1].reset_states(state)    # RNN layer

                # run validation over trajectory
                input = [observation, odometry]
                model_input = (input, state)

                # forward pass
                output, state = pfnet_model(model_input, training=False)

                # compute loss
                particle_states, particle_weights = output
                loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
                loss_pred = loss_dict['pred']

                test_loss(loss_pred)  # overall trajectory loss

            # log epoch validation stats
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)

            # Save the weights
            pfnet_model.save_weights(params.test_log_dir + f'/chks/checkpoint_{epoch}_{test_loss.result():03.3f}/pfnet_checkpoint')

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, test loss: {test_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

    # close gym env
    env.close()

    sys.stdout = old_stdout
    log_file.close()
    print('training finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    params.train_log_dir = 'logs/' + current_time + '/train/'
    params.test_log_dir = 'logs/' + current_time + '/test/'

    params.run_validation = True

    params.output = 'training_results.log'

    run_training(params)
