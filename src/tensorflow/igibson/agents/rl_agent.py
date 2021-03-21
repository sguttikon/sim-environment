#!/usr/bin/env python3

import os
import gym
import reverb
import imageio
import argparse
import datetime
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gibson2.envs.igibson_env import iGibsonEnv

import tensorflow as tf
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec
from tensorflow.python.framework.tensor_spec import TensorSpec
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import gym_wrapper
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories.time_step import TimeStep

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson')
from utils import datautils

IMG_WIDTH = 56
IMG_HEIGHT = 56

class NavigateGibsonEnv(iGibsonEnv):

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):

        super(NavigateGibsonEnv, self).__init__(config_file=config_file,
                        scene_id=scene_id,
                        mode=mode,
                        action_timestep=action_timestep,
                        physics_timestep=physics_timestep,
                        device_idx=device_idx,
                        render_to_tensor=render_to_tensor,
                        automatic_reset=automatic_reset)

        self.observation_space = gym.spaces.Box(
                low=-1., high=+1.,
                shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                dtype=np.float32)

    def step(self, action):
        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])
        assert -1. <= np.min(rgb) and  np.max(rgb) <= +1.

        return rgb, reward, done, info

    def reset(self):
        state = super(NavigateGibsonEnv, self).reset()

        # process image for training
        rgb = datautils.process_raw_image(state['rgb'])
        assert -1. <= np.min(rgb) and  np.max(rgb) <= +1.

        return rgb

# metrics
def get_eval_metrics(actor):
    actor.run()
    results = {}
    for metric in actor.metrics:
        results[metric.name] = metric.result()
    return results

def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print(f'step = {step}: {eval_results}')

def train_sac(params):

    config_filename = os.path.join('../configs/', 'turtlebot_navigate.yaml')
    collect_env = NavigateGibsonEnv(
                    config_file=config_filename,
                    mode='headless',
                    action_timestep=1.0 / 120.0,
                    physics_timestep=1.0 / 120.0)
    # eval_env = NavigateGibsonEnv(
    #                 config_file=config_filename,
    #                 mode='headless',
    #                 action_timestep=1.0 / 120.0,
    #                 physics_timestep=1.0 / 120.0)

    # wrap iGibsonEnv to PyEnvironment
    collect_env = gym_wrapper.GymWrapper(collect_env)
    eval_env = collect_env # gym_wrapper.GymWrapper(eval_env)
    assert isinstance(collect_env, py_environment.PyEnvironment)
    assert isinstance(eval_env, py_environment.PyEnvironment)

    # strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    time_step_spec = TimeStep(
        TensorSpec(
            shape=(),
            dtype=tf.int32,
            name='step_type'
        ),
        TensorSpec(
            shape=(),
            dtype=tf.float32,
            name='reward'
        ),
        BoundedTensorSpec(
            shape=(),
            dtype=tf.float32,
            name='discount',
            minimum=np.array(0., dtype=np.float32),
            maximum=np.array(1., dtype=np.float32)
        ),
        BoundedTensorSpec(
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

    actor_fc_layers = (256, 256)
    critic_joint_fc_layers = (256, 256)

    # Critic Network to estimate Q(s, a)
    critic_net = critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform'
    )

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork
        ),
    )

    # SAC Agent
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        critic_network=critic_net,
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=params.actor_learning_rate
        ),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=params.critic_learning_rate
        ),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=params.alpha_learning_rate
        ),
        target_update_tau=params.target_update_tau,
        target_update_period=params.target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=params.gamma,
        reward_scale_factor=params.reward_scale_factor,
        train_step_counter=train_step,
    )

    tf_agent.initialize()

    # Replay Buffer
    table_name = 'uniform_table'
    table = reverb.Table(
        name=table_name,
        max_size=int(params.replay_buffer_capacity),
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )
    reverb_server = reverb.Server(tables=[table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=params.sequence_length,
        local_server=reverb_server,
    )

    dataset = reverb_replay.as_dataset(
        sample_batch_size=params.batch_size,
        num_steps=params.sequence_length
    ).prefetch(50)
    experience_dataset_fn = lambda: dataset

    # Policies
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy=tf_agent.policy,
            use_tf_function=True
    )
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy=tf_agent.collect_policy,
            use_tf_function=True
    )
    random_policy = random_py_policy.RandomPyPolicy(
            time_step_spec=collect_env.time_step_spec(),
            action_spec=collect_env.action_spec())

    # Actor
    # trajectories as [t0, t1, t2, t3], [t1, t2, t3, t4], ....
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                    py_client=reverb_replay.py_client,
                    table_name=table_name,
                    sequence_length=params.sequence_length,
                    stride_length=1)

    # use random policy to collect experiences to seed replay buffer
    initial_collect_actor = actor.Actor(
        env=collect_env,
        policy=random_policy,
        train_step=train_step,
        steps_per_run=params.initial_collect_steps,
        observers=[rb_observer]
    )
    initial_collect_actor.run()

    # collect actor for training
    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
                        env=collect_env,
                        policy=collect_policy,
                        train_step=train_step,
                        steps_per_run=1,
                        metrics=actor.collect_metrics(10),
                        summary_dir=os.path.join(params.rootdir, learner.TRAIN_DIR),
                        observers=[rb_observer, env_step_metric])

    # eval actor to evaluate the policy during training
    eval_actor = actor.Actor(env=eval_env,
                        policy=eval_policy,
                        train_step=train_step,
                        episodes_per_run=params.num_eval_episodes,
                        metrics=actor.eval_metrics(params.num_eval_episodes),
                        summary_dir=os.path.join(params.rootdir, 'eval'))

    # Learner performs gradient step updates using experience data from replay buffer
    saved_model_dir = os.path.join(params.rootdir, learner.POLICY_SAVED_MODEL_DIR)

    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir=saved_model_dir,
            agent=tf_agent,
            train_step=train_step,
            interval=params.policy_save_interval),
        triggers.StepPerSecondLogTrigger(
            train_step=train_step,
            interval=1000)
    ]

    agent_learner = learner.Learner(
                        root_dir=params.rootdir,
                        train_step=train_step,
                        agent=tf_agent,
                        experience_dataset_fn=experience_dataset_fn,
                        triggers=learning_triggers)

    # train SAC agent
    tf_agent.train_step_counter.assign(0)

    avg_return = get_eval_metrics(eval_actor)['AverageReturn']
    returns = [avg_return]

    for _ in tqdm(range(params.num_iterations)):
        # train
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # eval
        step = agent_learner.train_step_numpy

        if params.eval_interval and step % params.eval_interval == 0:
            metrics = get_eval_metrics(eval_actor)
            log_eval_metrics(step, metrics)
            returns.append(metrics['AverageReturn'])

        if params.log_interval and step % params.log_interval == 0:
            print(f'step = {step}: loss = {loss_info.loss.numpy()}')

    rb_observer.close()
    reverb_server.stop()

    steps = range(0, params.num_iterations + 1, params.eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
    plt.savefig('./average_return.png')

    num_episodes = 3
    video_filename = './sac_navigate.mp4'
    with imageio.get_writer(video_filename, fps=60) as video:
      for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(eval_env.render())
        while not time_step.is_last():
          action_step = eval_actor.policy.action(time_step)
          time_step = eval_env.step(action_step.action)
          video.append_data(eval_env.render())

    print('training finished')

def parse_args():
    """
    parse command line arguments

    :return dict: dictionary of parameters
    """

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num_iterations', type=int, default=1e6)

    argparser.add_argument('--initial_collect_steps', type=int, default=1e4)
    argparser.add_argument('--replay_buffer_capacity', type=int, default=1e4)
    argparser.add_argument('--sequence_length', type=int, default=2)
    argparser.add_argument('--collect_steps_per_iteration', type=int, default=1)

    argparser.add_argument('--batch_size', type=int, default=256)

    argparser.add_argument('--critic_learning_rate', type=float, default=3e-4)
    argparser.add_argument('--actor_learning_rate', type=float, default=3e-4)
    argparser.add_argument('--alpha_learning_rate', type=float, default=3e-4)
    argparser.add_argument('--target_update_tau', type=float, default=0.005)
    argparser.add_argument('--target_update_period', type=float, default=1)
    argparser.add_argument('--gamma', type=float, default=0.994)
    argparser.add_argument('--reward_scale_factor', type=float, default=1.0)

    argparser.add_argument('--log_interval', type=int, default=5e3)
    argparser.add_argument('--num_eval_episodes', type=int, default=20)
    argparser.add_argument('--eval_interval', type=int, default=1e4)
    argparser.add_argument('--policy_save_interval', type=int, default=5e3)

    params = argparser.parse_args()

    params.rootdir = './runs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    return params

if __name__ == '__main__':
    params = parse_args()

    train_sac(params)
