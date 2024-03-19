import os
import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC

environment_name = 'FetchReach-v2'
env = gym.make(environment_name, max_episode_steps=100, render_mode='rgb_array')

#Eğitim kısmı
log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)

PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Reach')
model.save(PPO_Path)

#Eğitim sonrasında değerlendirme için ayrılan kısım
"""del model"""
"""model = TQC.load(PPO_Path, env=env, tensorboard_log=log_path)"""
env = gym.make(environment_name, render_mode='human', max_episode_steps=100)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
env.close()