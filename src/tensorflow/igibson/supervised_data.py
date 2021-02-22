#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from utils import arguments, datautils
from utils.iGibson_env import iGibsonEnv

def collect_data(env, params, action_model, filename='./test.tfrecord', num_records=10):
    """
    Run the gym environment and collect the required stats
    :param params: parsed parameters
    :param action_model: pretrained action sampler model
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(num_records):
            print(f'episode: {i}')
            episode_data = datautils.gather_episode_stats(env, params, action_model)
            record = datautils.serialize_tf_record(episode_data)
            writer.write(record)

    print(f'Collected successfully in {filename}')

if __name__ == '__main__':
    params = arguments.parse_args()

    # create gym env
    env = iGibsonEnv(config_file=params.config_filename, mode=params.mode,
                action_timestep=1 / 10.0, physics_timestep=1 / 240.0,
                device_idx=params.gpu_num, max_step=params.max_step)
    env.reset()

    collect_data(env, params, None, './test.tfrecord', num_records=50)
