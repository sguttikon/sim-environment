#!/usr/bin/env python3

from utils import render, datautils, arguments, pfnet_loss
from utils.iGibson_env import iGibsonEnv
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import glob


import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
# set programatically the path to 'pfnet' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/pfnet')
import pfnet

def dataset_size():
    return 50

def run_evaluation(params):
    """
    run evaluation with the parsed arguments
    """

    old_stdout = sys.stdout
    log_file = open(params.output,'w')
    sys.stdout = log_file

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen
    num_batches = dataset_size() // batch_size

    # evaluation data
    # filenames = list(glob.glob(params.testfiles[0]))
    filenames = params.testfiles
    test_ds = datautils.get_dataflow(filenames, params.batch_size, is_training=False)

    # create gym env
    env = iGibsonEnv(config_file=params.config_filename, mode=params.mode,
                action_timestep=1 / 10.0, physics_timestep=1 / 240.0,
                device_idx=params.gpu_num, max_step=params.max_step)
    env.reset()

    # pf model
    pfnet_model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.pfnet_load:
        print("=====> Loading model from " + params.pfnet_load)
        pfnet_model.load_weights(params.pfnet_load)

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        mse_list = []
        success_list = []
        itr = test_ds.as_numpy_iterator()
        # run evaluation over all evaluation samples in an epoch
        for idx in tqdm(range(num_batches)):
            parsed_record = next(itr)
            data_sample = datautils.transform_raw_record(env, parsed_record, params)

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
                pfnet_model.layers[-1].reset_states(state)    # RNN layer

            input = [observations, odometry]
            model_input = (input, state)

            # forward pass
            output, state = pfnet_model(model_input, training=False)

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

    sys.stdout = old_stdout
    log_file.close()
    print('evaluation finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    params.output = 'evaluation_results.log'

    run_evaluation(params)
