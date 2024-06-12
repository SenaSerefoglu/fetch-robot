from env import Environment
from tuner import Tuner
from stable_baselines3 import HerReplayBuffer, SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import os

if __name__ == "__main__":
    pass
    """env_name = "FetchSlide-v2"

    env = gym.make(env_name, max_episode_steps=100, render_mode='human')
    env = DummyVecEnv([lambda: env])

    log_path = os.path.join('Logs', env_name)
    model = SAC(
        'MultiInputPolicy', 
        env, 
        verbose=1, 
        learning_rate=0.001, 
        batch_size=1024, 
        gamma=0.95,
        tau=0.05,
        policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
        tensorboard_log=log_path
    )

    model.learn(total_timesteps=6000000)

    # Save the model
    model.save("SAC_fetch_slide")"""



    """env = Environment("FetchSlide-v2", 100, 'human')
    env.define_model("DDPG")
    env.model.learn(total_timesteps=6000000)
    env.model.save()
"""

    """tuner = Tuner(Environment("FetchReach-v2", 100, 'human'))
    tuner.tune(model_type="PPO", max_timesteps=1000000, max_iterations=10, max_time=3600)"""