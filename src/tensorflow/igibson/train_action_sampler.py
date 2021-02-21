#!/usr/bin/env python3

import os
from stable_baselines3 import PPO
from iGibson_env import iGibsonEnv
from stable_baselines3.ppo import MlpPolicy

def train_action_sampler(timesteps, max_step):
    config_filename = os.path.join('./configs/', 'turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless', max_step=max_step)

    # Train the agent
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("ppo_igibson")

    del model # remove to demonstrate saving and loading
    env.close()

    print('training finished')

def test_action_sampler(max_step):
    config_filename = os.path.join('./configs/', 'turtlebot_demo.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless', show_plot=True, max_step=max_step)

    # Test the agent
    model = PPO.load('ppo_igibson')

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()

    print('training finished')

if __name__ == '__main__':
    train_action_sampler(timesteps=50000, max_step=100)
    test_action_sampler(max_step=100)
