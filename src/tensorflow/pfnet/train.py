#!/usr/bin/env python3

import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from utils import datautils, arguments, pfnet_loss

def train_dataset_size():
    return 800

def valid_dataset_size():
    return 800

def run_training(params):
    """
    run training with the parsed arguments
    """

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen
    num_train_batches = train_dataset_size() // batch_size
    num_valid_batches = valid_dataset_size() // batch_size

    # training data
    train_ds = datautils.get_dataflow(params.trainfiles, params.batch_size, params.s_buffer_size, is_training=True)

    # validation data
    test_ds = datautils.get_dataflow(params.testfiles, params.batch_size, params.s_buffer_size, is_training=True)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("=====> Loading model from " + params.load)
        model.load_weights(params.load)

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=params.learningrate)

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = keras.metrics.Mean('test_loss', dtype=tf.float32)

    train_summary_writer = tf.summary.create_file_writer(params.train_log_dir + 'gradient_tape/')
    test_summary_writer = tf.summary.create_file_writer(params.test_log_dir + 'gradient_tape/')

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        itr = train_ds.as_numpy_iterator()
        # run training over all training samples in an epoch
        for idx in tqdm(range(num_train_batches)):
            raw_record = next(itr)
            data_sample = datautils.transform_raw_record(raw_record, params)

            observation = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
            global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                        shape=(batch_size, num_particles), dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, global_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if params.stateful:
                model.layers[-1].reset_states(state)    # RNN layer

            # run training over trajectory
            input = [observation, odometry]
            model_input = (input, state)

            # enable auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                output, state = model(model_input, training=True)

                # compute loss
                particle_states, particle_weights = output
                loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
                loss_pred = loss_dict['pred']

            # compute gradients of the trainable variables with respect to the loss
            gradients = tape.gradient(loss_pred, model.trainable_weights)
            gradients = list(zip(gradients, model.trainable_weights))

            # run one step of gradient descent
            optimizer.apply_gradients(gradients)
            train_loss(loss_pred)  # overall trajectory loss

        # log epoch training stats
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        # Save the weights
        print("=====> saving trained model ")
        model.save_weights(params.train_log_dir + f'/chks/checkpoint_{epoch}_{train_loss.result():03.3f}/pfnet_checkpoint')

        if params.run_validation:
            itr = train_ds.as_numpy_iterator()
            # run validation over all validation samples in an epoch
            for idx in tqdm(range(num_valid_batches)):
                raw_record = next(itr)
                data_sample = datautils.transform_raw_record(raw_record, params)

                observation = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
                odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
                true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
                global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
                init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
                init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                            shape=(batch_size, num_particles), dtype=tf.float32)

                # start trajectory with initial particles and weights
                state = [init_particles, init_particle_weights, global_map]

                # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                # if non-stateful: pass the state explicity every step
                if params.stateful:
                    model.layers[-1].reset_states(state)    # RNN layer

                # run validation over trajectory
                input = [observation, odometry]
                model_input = (input, state)

                # forward pass
                output, state = model(model_input, training=False)

                # compute loss
                particle_states, particle_weights = output
                loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
                loss_pred = loss_dict['pred']

                test_loss(loss_pred)  # overall trajectory loss

            # log epoch validation stats
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)

            # Save the weights
            model.save_weights(params.test_log_dir + f'/chks/checkpoint_{epoch}_{test_loss.result():03.3f}/pfnet_checkpoint')

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, test loss: {test_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

    print('training finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    params.train_log_dir = 'logs/' + current_time + '/train/'
    params.test_log_dir = 'logs/' + current_time + '/test/'

    params.run_validation = True
    params.s_buffer_size = 500

    run_training(params)
