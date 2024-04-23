import os
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback


class Environment:
    def __init__(self, environment_name, max_episode_steps, render_mode):
        self.environment_name = environment_name
        self.env = gym.make(self.environment_name, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self.env = DummyVecEnv([lambda: self.env])
        self.log_path = os.path.join('Logs', environment_name)
        
    def define_model(self, model_type):
        if self.environment_name == "FetchReach-v2" or self.environment_name == "FetchReachDense-v2":
            if model_type == "PPO":
                self.model = PPO('MultiInputPolicy', self.env, verbose=1, tensorboard_log=self.log_path)
                self.callback_list = CallbackList([])


        elif self.environment_name == "FetchPush-v2" or self.environment_name == "FetchPushDense-v2":
            if model_type == "TQC":
                self.model = TQC('MultiInputPolicy',self.env, batch_size=1024, gamma=0.95, tau=0.005, train_freq=1, target_entropy=0.01,
                                learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 256, 128], n_critics=2), replay_buffer_class=HerReplayBuffer, buffer_size=1000000,
                                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=self.log_path)
                self.callback_list = CallbackList([])


        elif self.environment_name == "FetchSlide-v2" or self.environment_name == "FetchSlideDense-v2":
            if model_type == "TQC":
                self.model = TQC('MultiInputPolicy',self.env, learning_rate=0.001, buffer_size=1000000, learning_starts=1000, batch_size=2048,
                                tau=0.05, gamma=0.95, ent_coef=0.01, policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
                                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), target_entropy='auto',
                                replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=self.log_path)
                
                logPathForBestmModel = os.path.join(self.log_path, "bestModel")
                eval_callback = EvalCallback(self.env, best_model_save_path=f'./{logPathForBestmModel}/',
                             log_path=f'./{logPathForBestmModel}/', eval_freq=5000,
                             deterministic=True, render=False)
                self.callback_list = CallbackList([eval_callback])
        

        elif self.environment_name == "FetchPickAndPlace-v2" or self.environment_name == "FetchPickAndPlaceDense-v2":
            if model_type == "TQC":
                self.model = TQC('MultiInputPolicy', self.env, learning_rate=0.001, batch_size=1024, gamma=0.98, tau=0.005, train_freq=1,
                                learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,
                                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=self.log_path)
                self.callback_list = CallbackList([])

    def train_model(self, timesteps=1000000):        
        self.model.learn(total_timesteps=timesteps, callback=self.callback_list)
    
    def save_model(self): 
        modelPath = os.path.join(self.log_path, 'Saved Models')
        self.model.save(modelPath)
        
    def evaluate_model(self, modelPath):
        model = TQC.load(modelPath, env=self.env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=100)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()