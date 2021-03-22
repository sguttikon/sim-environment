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
from agents import SACAgent
import matplotlib.pyplot as plt
from gibson2.envs.igibson_env import iGibsonEnv

import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import gym_wrapper
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import policy_saver
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson')
# set_path('/home/guttikon/awesome_robotics/sim-environment/src/tensorflow/igibson')
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
                    physics_timestep=1.0 / 120.0
    )
    # eval_env = NavigateGibsonEnv(
    #                 config_file=config_filename,
    #                 mode='headless',
    #                 action_timestep=1.0 / 120.0,
    #                 physics_timestep=1.0 / 120.0)

    # wrap iGibsonEnv to PyEnvironment
    collect_py_env = gym_wrapper.GymWrapper(collect_env)
    eval_py_env = collect_py_env # gym_wrapper.GymWrapper(eval_env)
    assert isinstance(collect_py_env, py_environment.PyEnvironment)
    assert isinstance(eval_py_env, py_environment.PyEnvironment)

    sac = SACAgent()
    tf_agent = sac.tf_agent
    train_step = sac.train_step

    # Replay Buffer
    table_name = 'uniform_table'
    table = reverb.Table(
        name=table_name,
        max_size=params.replay_buffer_capacity,
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
    eval_policy = sac.eval_policy
    collect_policy = sac.collect_policy
    random_policy = random_py_policy.RandomPyPolicy(
            time_step_spec=collect_py_env.time_step_spec(),
            action_spec=collect_py_env.action_spec()
    )
    tf_policy_saver = policy_saver.PolicySaver(sac.tf_agent.policy)

    # Actor
    # trajectories as [t0, t1, t2, t3], [t1, t2, t3, t4], ....
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                    py_client=reverb_replay.py_client,
                    table_name=table_name,
                    sequence_length=params.sequence_length,
                    stride_length=1
    )

    # use random policy to collect experiences to seed replay buffer
    initial_collect_actor = actor.Actor(
        env=collect_py_env,
        policy=random_policy,
        train_step=train_step,
        steps_per_run=params.initial_collect_steps,
        observers=[rb_observer]
    )
    initial_collect_actor.run()

    # collect actor for training
    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
                        env=collect_py_env,
                        policy=collect_policy,
                        train_step=train_step,
                        steps_per_run=1,
                        metrics=actor.collect_metrics(10),
                        summary_dir=os.path.join(params.rootdir, learner.TRAIN_DIR),
                        observers=[rb_observer, env_step_metric]
    )

    # eval actor to evaluate the policy during training
    eval_actor = actor.Actor(env=eval_py_env,
                        policy=eval_policy,
                        train_step=train_step,
                        episodes_per_run=params.num_eval_episodes,
                        metrics=actor.eval_metrics(params.num_eval_episodes),
                        summary_dir=os.path.join(params.rootdir, 'eval')
    )

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
                        triggers=learning_triggers
    )

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

    policy_dir = os.path.join(params.rootdir, 'output')
    tf_policy_saver.save(policy_dir)

    steps = range(0, params.num_iterations + 1, params.eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
    plt.savefig(policy_dir + '/average_return.png')

    print('training finished')

def test_sac(params):

    # wrap iGibsonEnv to PyEnvironment
    config_filename = os.path.join('../configs/', 'turtlebot_navigate.yaml')
    eval_env = NavigateGibsonEnv(
                    config_file=config_filename,
                    mode='headless',
                    action_timestep=1.0 / 120.0,
                    physics_timestep=1.0 / 120.0
    )
    eval_py_env = gym_wrapper.GymWrapper(eval_env)
    assert isinstance(eval_py_env, py_environment.PyEnvironment)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    policy_dir = './runs/20210322-162208/output'
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)

    num_episodes = 3
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

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
    argparser.add_argument('--seed', type=int, default='42', help='Fix the random seed of numpy and tensorflow.')

    params = argparser.parse_args()

    params.num_iterations = int(params.num_iterations)
    params.replay_buffer_capacity = int(params.replay_buffer_capacity)

    # fix seed
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    params.rootdir = './runs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    return params

if __name__ == '__main__':
    params = parse_args()

    train_sac(params)

    # test_sac(params)
