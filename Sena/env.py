import os
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class Environment:
    def __init__(self, environment_name, max_episode_steps, render_mode):
        self.environment_name = environment_name
        self.env = gym.make(self.environment_name, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self.env = DummyVecEnv([lambda: self.env])
        self.log_path = os.path.join('Logs', environment_name)
        
    """def define_model(self, modelType=TQC, batch_size=1024, gamma=0.95, tau=0.005, train_freq=1, target_entropy=0.01, learning_starts=1000,
                     policy_kwargs=dict(net_arch=[512, 512, 256, 128], n_critics=2), replay_buffer_class=HerReplayBuffer, buffer_size=1000000, 
                     replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1):
        
        self.model = modelType('MultiInputPolicy', self.env, batch_size=batch_size, gamma=gamma, tau=tau, train_freq=train_freq, target_entropy=target_entropy,
                    learning_starts=learning_starts, policy_kwargs=policy_kwargs, replay_buffer_class=replay_buffer_class, buffer_size=buffer_size,
                    replay_buffer_kwargs=replay_buffer_kwargs, verbose=verbose, tensorboard_log=self.log_path)"""
    

    def train_model(self, timesteps=1000000):
        self.model.learn(total_timesteps=timesteps)
        
    def save_params(self): #Bitir
        params = self.model.get_parameters()
    
    def save_model(self, modelPath): # yolu degistir
        self.model.save(modelPath)
        
    def evaluate_model(self, modelPath):
        model = TQC.load(modelPath, env=self.env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=100)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()