#!/usr/bin/env python3

import argparse
import datetime
from gibson2.envs.igibson_env import iGibsonEnv
import gym
import os
import tensorflow as tf

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
# path to custom tf_agents
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/stanford/agents')

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks.utils import mlp_layers
import suite_gibson

def train_eval(params):

    root_dir = os.path.expanduser(params.rootdir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')


    # configure summary writer
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        logdir=train_dir,
        flush_millis=params.summaries_flush_secs*1000
    )
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        logdir=eval_dir,
        flush_millis=params.summaries_flush_secs*1000
    )

    # configure eval metrics
    eval_metrics = [
        tf_metrics.AverageReturnMetric(
            buffer_size=params.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=params.num_eval_episodes),
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step%params.summary_interval, 0)
    ):

        # wrap train iGibsonEnv to PyEnvironment
        config_file = os.path.join('../configs/', 'turtlebot_navigate.yaml')
        tf_py_env = suite_gibson.load(
            config_file=config_file,
            env_mode='headless',
            device_idx=params.gpu_num,
        )
        assert isinstance(tf_py_env, py_environment.PyEnvironment)
        tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)

        # wrap eval iGibsonEnv to PyEnvironment
        ## HACK: use same train environment
        eval_tf_env = tf_env

        time_step_spec = tf_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = tf_env.action_spec()
        print('observation_spec', observation_spec)
        print('action_spec', action_spec)

        # configure preprocessing layer and combiner
        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        preprocessing_layers = {}
        if 'rgb' in observation_spec:
            preprocessing_layers['rgb'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=params.conv_2d_layer_params,
                fc_layer_params=params.encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'depth' in observation_spec:
            preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=params.conv_2d_layer_params,
                fc_layer_params=params.encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'scan' in observation_spec:
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=params.conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=params.encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_spec:
            preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=params.encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if len(preprocessing_layers) <= 1:
            preprocessing_combiner = None
        else:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        # actor network
        actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=params.actor_fc_layers,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
                kernel_initializer=glorot_uniform_initializer
            )
        # critic network
        critic_net = critic_network.CriticNetwork(
                input_tensor_spec=(observation_spec, action_spec),
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                observation_fc_layer_params=params.critic_obs_fc_layers,
                action_fc_layer_params=params.critic_action_fc_layers,
                joint_fc_layer_params=params.critic_joint_fc_layers,
                kernel_initializer=glorot_uniform_initializer
            )

def parse_args():
    """
    parse command line arguments

    :return dict: dictionary of parameters
    """

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--rootdir', type=str, default='./runs')
    argparser.add_argument('--gpu_num', type=int, default='0', help='use gpu no. to train')
    argparser.add_argument('--summaries_flush_secs', type=int, default=10)
    argparser.add_argument('--summary_interval', type=int, default=1000)

    # params for train
    argparser.add_argument('--num_eval_episodes', type=int, default=30)

    params = argparser.parse_args()

    params.rootdir = os.path.join(params.rootdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    params.conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    params.conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    params.encoder_fc_layers = [256]
    params.actor_fc_layers = [256]
    params.critic_obs_fc_layers = [256]
    params.critic_action_fc_layers = [256]
    params.critic_joint_fc_layers = [256]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert params.gpu_num < len(gpus)
    if gpus:
        # restrict TF to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[params.gpu_num], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            print(e)

    return params

if __name__ == '__main__':
    params = parse_args()

    train_eval(params)
