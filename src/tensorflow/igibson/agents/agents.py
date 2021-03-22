#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec
from tensorflow.python.framework.tensor_spec import TensorSpec
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import train_utils
from tf_agents.trajectories.time_step import TimeStep

IMG_WIDTH = 56
IMG_HEIGHT = 56

def normal_projection_net(
    action_spec,
    init_action_stddev=0.35,
    init_means_output_factor=0.1
):
    return normal_projection_network.NormalProjectionNetwork(
        sample_spec=action_spec,
        init_means_output_factor=init_means_output_factor,
        mean_transform=None,
        std_transform=sac_agent.std_clip_transform,
        state_dependent_std=True,
        scale_distribution=True
    )

class SACAgent(object):

    def __init__(
        self,
        # for network construction
        actor_fc_layers = [256, 256],
        critic_joint_fc_layers = [256, 256],
        critic_obs_fc_layers = [256, 256],
        critic_action_fc_layers = [256, 256],
        conv_layers=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)],
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

        # initialize time_step_spec, observation_spec, action_spec
        time_step_spec = TimeStep(
            step_type=TensorSpec(
                shape=(),
                dtype=tf.int32,
                name='step_type'
            ),
            reward=TensorSpec(
                shape=(),
                dtype=tf.float32,
                name='reward'
            ),
            discount=BoundedTensorSpec(
                shape=(),
                dtype=tf.float32,
                name='discount',
                minimum=np.array(0., dtype=np.float32),
                maximum=np.array(1., dtype=np.float32)
            ),
            observation=BoundedTensorSpec(
                shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                dtype=tf.float32,
                name='observation',
                minimum=np.array(-1., dtype=np.float32),
                maximum=np.array(+1., dtype=np.float32)
            ),
        )
        observation_spec = time_step_spec.observation
        action_spec = BoundedTensorSpec(
            shape=(2,),
            dtype=tf.float32,
            name='action',
            minimum=np.array(-1., dtype=np.float32),
            maximum=np.array(+1., dtype=np.float32)
        )

        # Critic Network to estimate Q(s, a)
        critic_net = critic_network.CriticNetwork(
            input_tensor_spec=(observation_spec, action_spec),
            observation_conv_layer_params=conv_layers,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer='glorot_uniform',
        )

        # Actor Network to p(a|s) distribution
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            # preprocessing_layers=None,
            # preprocessing_combiner=None,
            conv_layer_params=conv_layers,
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=normal_projection_net,
            kernel_initializer='glorot_uniform',
        )

        # SAC Agent
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
