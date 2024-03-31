import os
import gymnasium as gym
from sb3_contrib import TQC, PPO
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class FetchPush:
    def __init__(self, isDense):
        if isDense:
            self.environment_name = 'FetchPushDense-v2'
        else:
            self.environment_name = 'FetchPush-v2'
        # Create the environment
        self.env = gym.make(self.environment_name, max_episode_steps=100, render_mode='human')
        self.env = DummyVecEnv([lambda: self.env])
        # Path to save the model and logs
        self.log_path = os.path.join('Training', 'Logs')

        
    def define_TQCmodel(self):
        # Create the model
        self.model = TQC('MultiInputPolicy',self.env, batch_size=1024, gamma=0.95, tau=0.005, train_freq=1, target_entropy=0.01,
                    learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 256, 128], n_critics=2), replay_buffer_class=HerReplayBuffer, buffer_size=1000000,
                    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=self.log_path)

    def train_model(self, timesteps=1000000):
        self.model.learn(total_timesteps=timesteps)
        
    def save_model(self):
        self.modelPath = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchPushDense-v2')
        self.model.save(self.modelPath)
        
    def evaluate_model(self):
        model = TQC.load(self.modelPath, env=env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=100)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()



class FetchSlide:
    def __init__(self, isDense):
        if isDense:
            self.environment_name = 'FetchSlideDense-v2'
        else:
            self.environment_name = 'FetchSlide-v2'
        # Create the environment
        self.env = gym.make(self.environment_name, max_episode_steps=100, render_mode='rgb_array')
        self.env = DummyVecEnv([lambda: self.env])
        # Path to save the model and logs
        self.log_path = os.path.join('Training', 'Logs')


    def define_TQCmodel(self):
        # Create the model
        self.model = TQC('MultiInputPolicy',self.env, learning_rate=0.001, buffer_size=1000000, learning_starts=1000, batch_size=2048,
                        tau=0.05, gamma=0.95, ent_coef=0.01, policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
                        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), target_entropy='auto',
                        replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=self.log_path)
    
    def train_model(self, timesteps=6000000):
        self.model.learn(total_timesteps=timesteps)

    def save_model(self):
        self.modelPath = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchSlide-v2_tuned')
        self.model.save(self.modelPath)

    def evaluate_model(self):
        model = TQC.load(self.modelPath, env=env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=100)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()



class FetchPickAndPlace:
    def __init__(self, isDense):
        if isDense:
            self.environment_name = 'FetchPickAndPlaceDense-v2'
        else:
            self.environment_name = 'FetchPickAndPlace-v2'
        # Create the environment
        self.env = gym.make(self.environment_name, max_episode_steps=100, render_mode='human')
        self.env = DummyVecEnv([lambda: self.env])
        # Path to save the model and logs
        self.log_path = os.path.join('Training', 'Logs')


    def define_TQCmodel(self):
        # Create the model
        self.model = TQC('MultiInputPolicy', self.env, learning_rate=0.001, batch_size=512, gamma=0.98, tau=0.005, train_freq=1,
        learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=self.log_path)
    
    def train_model(self, timesteps=1000000):
        self.model.learn(total_timesteps=timesteps)

    def save_model(self):
        self.modelPath = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchPickAndPlaceDense-v2')
        self.model.save(self.modelPath)

    def evaluate_model(self):
        model = TQC.load(self.modelPath, env=env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=100)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()



class FetchReach:
    def __init__(self, isDense):
        if isDense:
            self.environment_name = 'FetchReachDense-v2'
        else:
            self.environment_name = 'FetchReach-v2'
        # Create the environment
        self.env = gym.make(self.environment_name, max_episode_steps=50, render_mode='human')
        self.env = DummyVecEnv([lambda: self.env])
        # Path to save the model and logs
        self.log_path = os.path.join('Training', 'Logs')


    def define_PPOmodel(self):
        # Create the model
        self.model = PPO('MultiInputPolicy', self.env, verbose=1, tensorboard_log=self.log_path)

    def train_model(self, timesteps=1000000):
        self.model.learn(total_timesteps=timesteps)

    def save_model(self):
        self.modelPath = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchReachDense-v2')
        self.model.save(self.modelPath)
        
    def evaluate_model(self):
        model = PPO.load(self.modelPath, env=env, tensorboard_log=self.log_path)
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=50)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=True,)
        print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')
        env.close()