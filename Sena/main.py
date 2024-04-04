from tuner import Tuner
from env import Environment

if __name__ == "__main__":
    environment_name = 'FetchReach-v2'
    environment = Environment(environment_name, max_episode_steps=100, render_mode='rgb_array')
    tuner = Tuner(environment)

    tuner.tune(model_type="PPO", max_timesteps=500000, max_iterations=10, max_time=3600, save_model=True)

    #environment.train_model(timesteps=1000)