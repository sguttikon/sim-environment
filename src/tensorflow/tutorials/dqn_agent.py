#!/usr/bin/env python3

import imageio
import PIL.Image
import numpy as np
import tensorflow as tf

from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym, tf_py_environment

# hyper parameters
lr = 1e-3
batch_size = 64
log_interval = 200
num_iterations = 20000

collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

eval_interval = 1000
num_eval_episodes = 10

def compute_avg_return(env, policy, num_episodes):
    total_return = 0.
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_dense_layer(
        num_units,
        activation_fn,
        k_initializer,
        b_initializer):

    return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=k_initializer,
                bias_initializer=b_initializer)

# load cartpole environment from OpenAI gym suite
env_name = 'CartPole-v0'

train_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# initialize QNetwork
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
num_actions = action_tensor_spec.maximum - \
                action_tensor_spec.minimum + 1

dense_layers = [
                create_dense_layer(
                    fc_layer_params[0],
                    tf.keras.activations.relu,
                    tf.keras.initializers.VarianceScaling(
                            scale=2.0,
                            mode='fan_in',
                            distribution='truncated_normal'),
                    None),
                create_dense_layer(
                    fc_layer_params[1],
                    tf.keras.activations.relu,
                    tf.keras.initializers.VarianceScaling(
                            scale=2.0,
                            mode='fan_in',
                            distribution='truncated_normal'),
                    None),
                create_dense_layer(
                    num_actions,
                    None,
                    tf.keras.initializers.RandomUniform(
                            minval=-0.03,
                            maxval=+0.03),
                    tf.keras.initializers.Constant(-0.2)),
                ]
q_net = sequential.Sequential(dense_layers)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
train_step_counter = tf.Variable(0)

# initialize DQNAgent
agent = dqn_agent.DqnAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

# initialize replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    data_spec=agent.collect_data_spec,
                    batch_size=train_env.batch_size,
                    max_length=replay_buffer_max_length)

def collect_step(env, policy, buffer):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(
                        time_step,
                        action_step,
                        next_time_step)

    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

random_policy = random_tf_policy.RandomTFPolicy(
                        train_env.time_step_spec(),
                        train_env.action_spec())
collect_data(
        train_env,
        random_policy,
        replay_buffer,
        100)

# replay buffer iterator
dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=2).prefetch(3)
iterator = iter(dataset)

# train agent
agent.train = common.function(agent.train) # wrap code in a tf graph

agent.train_step_counter.assign(0)

avg_return = compute_avg_return(
                    eval_env,
                    agent.policy,
                    num_eval_episodes)
returns = [avg_return]

tf_policy_saver = policy_saver.PolicySaver(agent.policy)
for _ in range(10):

    # collect data using collect_policy
    collect_data(
            train_env,
            agent.collect_policy,
            replay_buffer,
            collect_steps_per_iteration)

    # sample a batch of data from buffer
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f'step = {step}: loss = {train_loss}')

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
                            eval_env,
                            agent.policy,
                            num_eval_episodes)
        returns.append(avg_return)

policy_dir = './dqn_policy'
tf_policy_saver.save(policy_dir)

saved_policy = tf.compat.v2.saved_model.load(policy_dir)

def create_policy_eval_video(
        policy,
        filename,
        num_episodes=5,
        fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())

create_policy_eval_video(saved_policy, "trained-agent")
# create_policy_eval_video(random_policy, "random-agent")
