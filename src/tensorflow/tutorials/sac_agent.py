#!/usr/bin/env python3

import os
import reverb
import tempfile
import tensorflow as tf

from tf_agents.metrics import py_metrics
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.environments import suite_pybullet
from tf_agents.train import actor, learner, triggers
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train.utils import strategy_utils, spec_utils, train_utils

env_name = 'MinitaurBulletEnv-v0'
collect_env = suite_pybullet.load(env_name)
eval_env = suite_pybullet.load(env_name)

# distribution across multiple GPU
# all variable and agents need to be under strategy.scope()
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))

# Critic Network which extimates Q(s, a)
critic_joint_fc_layer_params = (256, 256)
with strategy.scope():
    critic_net = critic_network.CriticNetwork(
                    input_tensor_spec=(
                        observation_spec,
                        action_spec),
                    observation_fc_layer_params=None,
                    action_fc_layer_params=None,
                    joint_fc_layer_params=critic_joint_fc_layer_params,
                    kernel_initializer='glorot_uniform',
                    last_kernel_initializer='glorot_uniform')

# Actor Network used Critic Netowkr to predict parameters of MultiVariateNormalDiag distribution.
# This distribution will be conditioned on current observation to generate actions.
actor_fc_layer_params = (256, 256)
with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                fc_layer_params=actor_fc_layer_params,
                continuous_projection_net=(
                    tanh_normal_projection_network.TanhNormalProjectionNetwork)
                )

# SAC Agent
critic_lr = 3e-4
actor_lr = 3e-4
alpha_lr = 3e-4
update_tau = 0.005
update_period = 1
gamma = 0.99
reward_scale = 1.0
with strategy.scope():
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                                    learning_rate=actor_lr),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                                    learning_rate=critic_lr),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                                    learning_rate=alpha_lr),
                target_update_tau=update_tau,
                target_update_period=update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=reward_scale,
                train_step_counter=train_step)
    tf_agent.initialize()

# replay buffer
rb_capacity = 10000
tb_name = 'uniform_table'
# rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
#                     samples_per_insert=3.0,
#                     min_size_to_sample=3,
#                     error_buffer=3.0)
table = reverb.Table(
            name=tb_name,
            max_size=rb_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1))
reverb_server = reverb.Server([table])

# SAC agent need both current and next observation to compute loss
reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
                    data_spec=tf_agent.collect_data_spec,
                    sequence_length=2,
                    table_name=tb_name,
                    local_server=reverb_server)

# generate tf dataset from reverb buffer
batch_size = 356
dataset = reverb_replay.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

# policies
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                policy=tf_agent.policy,
                use_tf_function=True)
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
                policy=tf_agent.collect_policy,
                use_tf_function=True)
random_policy = random_py_policy.RandomPyPolicy(
                time_step_spec=collect_env.time_step_spec(),
                action_spec=collect_env.action_spec())

# to store trajectories as [t0, t1], [t1, t2]
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                py_client=reverb_replay.py_client,
                table_name=tb_name,
                sequence_length=2,
                stride_length=1)

# Actor to manage interactions between a policy and env
initial_collect_steps = 10000
initial_collect_actor = actor.Actor(
                            env=collect_env,
                            policy=random_policy,
                            train_step=train_step,
                            steps_per_run=initial_collect_steps,
                            observers=[rb_observer])
initial_collect_actor.run()

tempdir = tempfile.gettempdir()
env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
                    env=collect_env,
                    policy=collect_policy,
                    train_step=train_step,
                    steps_per_run=1,
                    metrics=actor.collect_metrics(10),
                    summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
                    observers=[rb_observer, env_step_metric])

num_eval_episodes = 20
eval_actor = actor.Actor(env=eval_env,
                    policy=eval_policy,
                    train_step=train_step,
                    episodes_per_run=num_eval_episodes,
                    metrics=actor.eval_metrics(num_eval_episodes),
                    summary_dir=os.path.join(tempdir, 'eval'))

# Learner performs gradient step updates
saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

policy_save_interval = 5000
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir=saved_model_dir,
        agent=tf_agent,
        train_step=train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(
        train_step=train_step,
        interval=1000)
]

agent_learner = learner.Learner(
                    root_dir=tempdir,
                    train_step=train_step,
                    agent=tf_agent,
                    experience_dataset_fn=experience_dataset_fn,
                    triggers=learning_triggers)

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

# train SAC agent
tf_agent.train_step_counter.assign(0)

avg_return = get_eval_metrics(eval_actor)['AverageReturn']
returns = [avg_return]

log_interval = 5000
eval_interval = 10000
num_iterations = 100000
for _ in range(num_iterations):
    # train
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # eval
    step = agent_learner.train_step_numpy

    if eval_interval and step % eval_interval == 0:
        metrics = get_eval_metrics(eval_actor)
        log_eval_metrics(step, metrics)
        returns.append(metrics['AverageReturn'])

    if log_interval and step % log_interval == 0:
        print(f'step = {step}: loss = {loss_info.loss.numpy()}')

rb_observer.close()
reverb_server.stop()
