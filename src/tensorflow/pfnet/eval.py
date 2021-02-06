#!/usr/bin/env python3

import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import datautils, arguments, pfnet_loss

def run_evaluation(params):
    """
    run training with the parsed arguments
    """

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen

    # evaluation data
    test_ds = datautils.get_dataflow(params.testfiles, params.batch_size, is_training=False)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("Loading model from " + params.load)
        model.load_weights(params.load)

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        mse_list = []
        success_list = []

        # run evaluation over all evaluation samples in an epoch
        for raw_record in tqdm(test_ds.as_numpy_iterator()):
            data_sample = datautils.transform_raw_record(raw_record, params)

            observations = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
            global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                        shape=(batch_size, num_particles), dtype=tf.float32)

            # HACK: tile global map since RNN accepts [batch, time_steps, ...]
            global_map = tf.tile(tf.expand_dims(global_map, axis=1), [1, trajlen, 1, 1, 1])

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if params.stateful:
                model.layers[-1].reset_states(state)    # RNN layer

            input = [observations, odometry, global_map]
            model_input = (input, state)

            # forward pass
            output, state = model(model_input, training=False)

            # compute loss
            particle_states, particle_weights = output
            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)

            # we have squared differences along the trajectory
            mse = np.mean(loss_dict['coords'])
            mse_list.append(mse)

            # localization is successfull if the rmse error is below 1m for the last 25% of the trajectory
            successful = np.all(loss_dict['coords'][-trajlen//4:] < 1.0 ** 2)  # below 1 meter
            success_list.append(successful)

        # report results
        mean_rmse = np.mean(np.sqrt(mse_list))
        total_rmse = np.sqrt(np.mean(mse_list))
        mean_success = np.mean(np.array(success_list, 'i'))
        print(f'Mean RMSE (average RMSE per trajectory) = {mean_rmse*100} cm')
        print(f'Overall RMSE (reported value) = {total_rmse*100} cm')
        print(f'Success rate = {mean_success*100} %')

    print('evaluation finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    run_evaluation(params)
