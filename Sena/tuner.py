import os
import gymnasium as gym
from sb3_contrib import TQC, ARS, TRPO, RecurrentPPO
from stable_baselines3 import HerReplayBuffer, PPO, DDPG, A2C, SAC, HER, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
import random
import time

class TimeLimitCallback(BaseCallback):
    def __init__(self, time_limit, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.time_limit = time_limit
        self.time_passed = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.time_passed = 0

    def _on_step(self) -> bool:
        if self.start_time is not None and time.time() - self.start_time > self.time_limit:
            print(f"Training stopped after {self.time_limit} seconds.")
            return False
        return True
    
    def _on_training_end(self) -> None:
        self.time_passed = time.time() - self.start_time

class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.mean_reward = 0
        self.episode_rewards = []

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.mean_reward = 0
        self.episode_rewards = []

    def _on_episode_end(self) -> None:
        self.episode_rewards.append(self.model.ep_info_buffer['r'])

    def get_mean_reward(self):
        if len(self.episode_rewards) > 0:
            self.mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        return self.mean_reward


class ParameterDicts:
    def __init__(self) -> None:
        self.PPO = {
                'learning_rate': [3e-4, 1e-4, 5e-5],
                'n_steps': [128, 256, 512],
                'batch_size': [64, 128, 256],
                'n_epochs': [4, 8, 16],
                'gamma': [0.99, 0.95],
                'gae_lambda': [0.95, 0.98],
                'clip_range': [0.2, 0.1],
                'clip_range_vf': [None, 0.2],
                'normalize_advantage': [True, False],
                'ent_coef': [0.01, 0.001],
                'vf_coef': [0.5, 0.75],
                'max_grad_norm': [0.5, 0.8],
                'use_sde': [True, False],
                'sde_sample_freq': [4, 8],
                'target_kl': [0.05],
                'policy_kwargs': [dict(net_arch=[512, 512, 256, 128])]
            }

class Tuner:
    
    def __init__(self, env, log_path) -> None:
        self.env = env
        self.log_path = os.path.join(log_path, 'Tuning')
        self.best_mean_reward = -float('inf')
        self.best_params = {}
        self.best_model = None
        self.parameters = ParameterDicts()


    def tunePPO(self, max_timesteps=1000000, max_iterations=10, max_time=3600, training_device="cuda", save_model=False):
        reward_tracker = RewardTrackerCallback()
        time_callback = TimeLimitCallback(time_limit=max_time)
        
        for iteration in range(max_iterations):
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=12, min_evals=20, verbose=1)
            eval_callback = EvalCallback(self.env, eval_freq=5000, callback_after_eval=stop_train_callback, verbose=0)
            callback_list = CallbackList([eval_callback, time_callback, reward_tracker])

            model = PPO('MultiInputPolicy', self.env, device=training_device, verbose=1, tensorboard_log=self.log_path)
        
            model.learning_rate = random.choice(self.parameters.PPO['learning_rate'])
            model.batch_size = random.choice(self.parameters.PPO['batch_size'])
            model.n_epochs = random.choice(self.parameters.PPO['n_epochs'])
            model.gamma = random.choice(self.parameters.PPO['gamma'])
            model.gae_lambda = random.choice(self.parameters.PPO['gae_lambda'])
            model.normalize_advantage = random.choice(self.parameters.PPO['normalize_advantage'])
            model.ent_coef = random.choice(self.parameters.PPO['ent_coef'])
            model.vf_coef = random.choice(self.parameters.PPO['vf_coef'])
            model.max_grad_norm = random.choice(self.parameters.PPO['max_grad_norm'])
            model.sde_sample_freq = random.choice(self.parameters.PPO['sde_sample_freq'])
            model.target_kl = random.choice(self.parameters.PPO['target_kl'])
            model.policy_kwargs = random.choice(self.parameters.PPO['policy_kwargs'])


            model.learn(total_timesteps=max_timesteps, callback=callback_list)

            mean_reward = reward_tracker.get_mean_reward()
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_params = model.get_parameters()
                self.best_model = model

            with open(os.path.join(self.log_path, f'PPO_{iteration+1}', 'params.txt'), 'a') as f:
                f.write(f'{model.get_parameters()}') # değişecek
                f.write(f'{mean_reward}')
                f.write(f'{time_callback.time_passed}')

            if save_model:
                model.save(os.path.join(self.log_path, f'PPO_{iteration+1}', 'model'))
        
        print(f'Best mean reward: {self.best_mean_reward}')
        print(f'Best parameters: {self.best_params}')
        