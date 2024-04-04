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
            print(f"Training stopped after {self.time_limit} seconds.\n")
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
        pass

    class FetchReach:
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
                    'policy_kwargs': [dict(net_arch=[512, 512, 256, 128])]
                }
            self.TQC = {
                    'learning_rate': [5e-3, 1e-4, 5e-5],
                    'n_steps': [128, 256, 512, 1024],
                    'batch_size': [128, 256, 512],
                    'buffer_size': [int(1e4), int(1e5), int(1e6)],
                    'learning_starts': [1e3, 1e4],
                    'gamma': [0.99, 0.95, 0.9],
                    'policy_kwargs': [dict(net_arch=[64, 64]), dict(net_arch=[128, 128])],
                    'replay_buffer_class': [HerReplayBuffer],
                    'replay_buffer_kwargs': {'n_sampled_goal': 4, 'goal_selection_strategy': ['future']}
            }
        

class Tuner:
    
    def __init__(self, environment) -> None:
        self.env = environment.env
        log_path = environment.log_path
        env_name = environment.environment_name
        self.log_path = os.path.join(log_path, 'Tuning')
        self.best_mean_reward = -float('inf')
        self.best_params = {}
        self.__temp_params = {}
        self.best_model = None
        
        if env_name == 'FetchReach-v2':
            self.__parameters = ParameterDicts().FetchReach()


    def tune(self, model_type, max_timesteps=1000000, max_iterations=10, max_time=3600, training_device="cuda", save_model=False):
        reward_tracker = RewardTrackerCallback()
        time_callback = TimeLimitCallback(time_limit=max_time)
        
        for iteration in range(max_iterations):
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=12, min_evals=20, verbose=1)
            eval_callback = EvalCallback(self.env, eval_freq=5000, callback_after_eval=stop_train_callback, verbose=0)
            callback_list = CallbackList([eval_callback, time_callback, reward_tracker])

            if model_type == 'PPO':
                model = PPO('MultiInputPolicy', self.env, device=training_device, verbose=1, tensorboard_log=self.log_path)

                model.learning_rate = random.choice(self.__parameters.PPO['learning_rate'])
                model.batch_size = random.choice(self.__parameters.PPO['batch_size'])
                model.n_epochs = random.choice(self.__parameters.PPO['n_epochs'])
                model.gamma = random.choice(self.__parameters.PPO['gamma'])
                model.gae_lambda = random.choice(self.__parameters.PPO['gae_lambda'])
                model.normalize_advantage = random.choice(self.__parameters.PPO['normalize_advantage'])
                model.ent_coef = random.choice(self.__parameters.PPO['ent_coef'])
                model.vf_coef = random.choice(self.__parameters.PPO['vf_coef'])
                model.policy_kwargs = random.choice(self.__parameters.PPO['policy_kwargs'])


                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
                    'learning_rate': model.learning_rate,
                    'batch_size': model.batch_size,
                    'n_epochs': model.n_epochs,
                    'gamma': model.gamma,
                    'gae_lambda': model.gae_lambda,
                    'normalize_advantage': model.normalize_advantage,
                    'ent_coef': model.ent_coef,
                    'vf_coef': model.vf_coef,
                    'max_grad_norm': model.max_grad_norm,
                    'sde_sample_freq': model.sde_sample_freq,
                    'target_kl': model.target_kl,
                    'policy_kwargs': model.policy_kwargs
                }

                with open(os.path.join(self.log_path, f'PPO_{iteration+1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'PPO_{iteration+1}', 'model'))

            if model_type == 'TQC':
                model = TQC('MultiInputPolicy', self.env, device=training_device, verbose=1,
                            tensorboard_log=self.log_path)

                model.learning_rate = random.choice(self.__parameters.TQC['learning_rate'])
                model.batch_size = random.choice(self.__parameters.TQC['batch_size'])
                model.buffer_size = random.choice(self.__parameters.TQC['buffer_size'])
                model.learning_starts = random.choice(self.__parameters.TQC['learning_starts'])
                model.gamma = random.choice(self.__parameters.TQC['gamma'])
                model.policy_kwargs = random.choice(self.__parameters.TQC['policy_kwargs'])

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
                    'learning_rate': model.learning_rate,
                    'batch_size': model.batch_size,
                    'buffer_size': model.buffer_size,
                    'learning_starts': model.learning_starts,
                    'gamma': model.gamma,
                    'policy_kwargs': model.policy_kwargs
                }

                with open(os.path.join(self.log_path, f'TQC_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'TQC_{iteration + 1}', 'model'))

            mean_reward = reward_tracker.get_mean_reward()
            if mean_reward > self.best_mean_reward:
                self.best_params = self.__temp_params
                self.best_mean_reward = mean_reward
                self.best_model = model

            print(f'Best Parameters For Now: {self.best_params}\n')     
        
        print(f'Best mean reward: {self.best_mean_reward}\n')
        print(f'Best parameters: {self.best_params}\n')
        