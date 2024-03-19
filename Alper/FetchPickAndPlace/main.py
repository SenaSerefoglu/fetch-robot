import os
import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC

environment_name = 'FetchPickAndPlace-v2'
env = gym.make(environment_name, max_episode_steps=100, render_mode='rgb_array')

#Eğitim kısmı
log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
#env = sb3_contrib.common.wrappers.TimeFeatureWrapper(env, max_steps=100)
model = TQC('MultiInputPolicy', env, learning_rate=0.001, batch_size=512, gamma=0.98, tau=0.005, train_freq=1,
            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)
"""model.learn(total_timesteps=1000000)"""

PPO_Path = os.path.join('Training', 'Saved Models', 'TQC_PickAndPlace')
"""model.save(PPO_Path)"""

#Eğitim sonrasında değerlendirme için ayrılan kısım
"""del model"""
model = TQC.load(PPO_Path, env=env, tensorboard_log=log_path)
env = gym.make(environment_name, render_mode='human', max_episode_steps=100)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
env.close()