from tuner import Tuner
from env import Environment
import gymnasium as gym
import os

if __name__ == "__main__":
    environment_name = 'FetchReach-v2'
    environment = Environment(environment_name, max_episode_steps=100, render_mode='human')
    env = environment.env
    log_path = environment.log_path
    tuner = Tuner(env, log_path)

    #tuner.tunePPO(max_timesteps=500000, max_iterations=10, max_time=3600, save_model=True)

    environment.train_model(timesteps=1000)