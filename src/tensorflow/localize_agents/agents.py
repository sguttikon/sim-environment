#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
# path to custom tf_agents
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/stanford/agents')
# set_path('/home/guttikon/awesome_robotics/sim-environment/src/tensorflow/stanford/agents')


from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

class SACAgent(object):

    def __init__(
        self,
        # env specs
        observation_spec,
        action_spec,
        time_step_spec,
        # for network construction
        conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)],
        encoder_fc_layers = [256],
        actor_fc_layers = [256],
        critic_obs_fc_layers = [256],
        critic_action_fc_layers = [256],
        critic_joint_fc_layers = [256],
        # for training
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_update_tau=0.005,
        target_update_period=1,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=1.0,
    ):

        strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

        # initialize preprocessing_layers and preprocessing_combiner for critic/actor networks
        with strategy.scope():
            preprocessing_layers = {}
            if 'rgb' in observation_spec:
                preprocessing_layers['rgb'] = tf.keras.Sequential(mlp_layers(
                    conv_1d_layer_params=None,
                    conv_2d_layer_params=conv_2d_layer_params,
                    fc_layer_params=encoder_fc_layers,
                    kernel_initializer='glorot_uniform',
                ))

            if 'depth' in observation_spec:
                preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
                    conv_1d_layer_params=None,
                    conv_2d_layer_params=conv_2d_layer_params,
                    fc_layer_params=encoder_fc_layers,
                    kernel_initializer='glorot_uniform',
                ))

            if 'scan' in observation_spec:
                preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                    conv_1d_layer_params=conv_1d_layer_params,
                    conv_2d_layer_params=None,
                    fc_layer_params=encoder_fc_layers,
                    kernel_initializer='glorot_uniform',
                ))

            if 'task_obs' in observation_spec:
                preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
                    conv_1d_layer_params=None,
                    conv_2d_layer_params=None,
                    fc_layer_params=encoder_fc_layers,
                    kernel_initializer='glorot_uniform',
                ))

            if len(preprocessing_layers) <= 1:
                preprocessing_combiner = None
            else:
                preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        # Critic Network to estimate Q(s, a)
        with strategy.scope():
            critic_net = critic_network.CriticNetwork(
                input_tensor_spec=(observation_spec, action_spec),
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                observation_fc_layer_params=critic_obs_fc_layers,
                action_fc_layer_params=critic_action_fc_layers,
                joint_fc_layer_params=critic_joint_fc_layers,
                kernel_initializer='glorot_uniform',
            )

        # Actor Network to p(a|s) distribution
        with strategy.scope():
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
                kernel_initializer='glorot_uniform',
            )

        # SAC Agent
        with strategy.scope():
            self.train_step = train_utils.create_train_step()
            self.tf_agent = sac_agent.SacAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                critic_network=critic_net,
                actor_network=actor_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate
                ),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate
                ),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate
                ),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=self.train_step,
            )
            self.tf_agent.initialize()

        # Policies
        self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                policy=self.tf_agent.policy,
                use_tf_function=True
        )
        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
                policy=self.tf_agent.collect_policy,
                use_tf_function=True
        )
