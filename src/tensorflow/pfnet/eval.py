#!/usr/bin/env python3

import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import datautils, arguments, pfnet_loss

def dataset_size():
    return 800

def run_evaluation(params):
    """
    run training with the parsed arguments
    """

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen
    num_batches = dataset_size() // batch_size

    # evaluation data
    test_ds = datautils.get_dataflow(params.testfiles, params.batch_size, is_training=False)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("=====> Loading model from " + params.load)
        model.load_weights(params.load)

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        mse_list = []
        success_list = []
        itr = test_ds.as_numpy_iterator()
        # run evaluation over all evaluation samples in an epoch
        for idx in tqdm(range(num_batches)):
            raw_record = next(itr)
            data_sample = datautils.transform_raw_record(raw_record, params)

            observations = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
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

            input = [observations, odometry]
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
        mean_rmse = np.mean(np.sqrt(mse_list)) * 100
        total_rmse = np.sqrt(np.mean(mse_list)) * 100
        mean_success = np.mean(np.array(success_list, 'i')) * 100
        print(f'Mean RMSE (average RMSE per trajectory) = {mean_rmse:03.3f} cm')
        print(f'Overall RMSE (reported value) = {total_rmse:03.3f} cm')
        print(f'Success rate = {mean_success:03.3f} %')

    print('evaluation finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    run_evaluation(params)
