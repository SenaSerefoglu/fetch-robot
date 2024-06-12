from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer, PPO, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import get_latest_run_id
from callback import TimeLimitCallback, RewardTrackerCallback
import time
import os
import random


class ParameterDicts:
    def __init__(self) -> None:
        pass

    class FetchReach:
        def __init__(self) -> None:
            self.PPO = {
                'learning_rate': [3e-4],
                'n_steps': [1024, 2048],
                'batch_size': [56, 128, 256],
                'n_epochs': [10, 8, 12],
                'gamma': [0.99, 0.95],
                'gae_lambda': [0.95, 0.98],
                'clip_range': [0.2, 0.1],
                'clip_range_vf': [None, 0.2],
                'normalize_advantage': [True, False],
                'ent_coef': [0.001, 0.0001],
                'vf_coef': [0.5, 0.75],
                'policy_kwargs': [dict(net_arch=[512, 512, 256, 128]), None]
                }
            self.TQC = {
                'learning_rate': [5e-3, 1e-4, 5e-5],
                'n_steps': [512, 1024],
                'batch_size': [128, 256, 512],
                'buffer_size': [int(1e7), int(1e6), int(1e5)],
                'learning_starts': [int(1e4), int(15e3)],
                'gamma': [0.99, 0.95, 0.9],
                'tau': [0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[64, 64]), dict(net_arch=[128, 128])],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [[dict(n_sampled_goal=4), dict(goal_selection_strategy='future')]]
            }
            self.OTHER = {
                'n_steps': [512, 1024],
                'batch_size': [128, 256, 512],
                'buffer_size': [int(1e7), int(1e6), int(1e5)],
                'learning_starts': [int(15e3), int(2e4)],
                'learning_rate': [1e-3, 1e-4],
                'gamma': [0.99, 0.95, 0.9],
                'tau': [0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[64, 64], n_critics=2), dict(net_arch=[128, 128], n_critics=2)],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [[dict(n_sampled_goal=4), dict(goal_selection_strategy='future')]]
            }

    class FetchPush:
        def __init__(self) -> None:
            self.TQC = {
                'learning_rate': [1e-3, 5e-4, 1e-4],
                'n_steps': [1024, 2048],
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [int(1e3), int(1e4)],
                'gamma': [0.9, 0.95, 0.98, 0.99],
                'tau': [0.05, 0.08],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [dict(goal_selection_strategy='future',n_sampled_goal=4)],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(n_critics=2, net_arch=[512, 512, 512])]
            }
            self.OTHER = {
                'n_steps': [1024, 2048],
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [int(1e4), int(15e3)],
                'learning_rate': [1e-3, 1e-4],
                'gamma': [0.99, 0.95, 0.9],
                'tau': [0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(net_arch=[512, 512, 512], n_critics=2)],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [[dict(n_sampled_goal=4), dict(goal_selection_strategy='future')]]
            }

    class FetchSlide:
        def __init__(self) -> None:
            self.TQC = {
                'learning_rate': [1e-3, 5e-4],
                'n_steps': [2048], 
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e6), int(1e7)],
                'learning_starts': [int(1e3), int(1e4)],
                'gamma': [0.98, 0.95],
                'tau': [0.08, 0.05],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(n_critics=2, net_arch=[512, 512, 512])],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [dict(n_sampled_goal=4, goal_selection_strategy='future')]
            }
            self.OTHER = {
                'n_steps': [1024, 2048],
                'batch_size': [2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [int(1e4), int(15e3)],
                'learning_rate': [1e-3, 1e-4],
                'gamma': [0.99, 0.95, 0.9],
                'tau': [0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(net_arch=[512, 512, 512], n_critics=2)],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [[dict(n_sampled_goal=4), dict(goal_selection_strategy='future')]]
            }

    class FetchPickAndPlace:
        def __init__(self) -> None:
            self.TQC = {
                'learning_rate': [1e-3, 5e-4, 1e-4],
                'n_steps': [1024, 2048], 
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [1e3, 1e4],
                'gamma': [0.9, 0.95, 0.98, 0.99],
                'tau': [0.02, 0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(n_critics=2, net_arch=[512, 512, 512])],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [dict(n_sampled_goal=4, goal_selection_strategy='future')]
            }
            self.OTHER = {
                'n_steps': [1024, 2048],
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [int(1e4), int(15e3)],
                'learning_rate': [1e-3, 1e-4],
                'gamma': [0.99, 0.95, 0.9],
                'tau': [0.05, 0.08],
                'policy_kwargs': [dict(net_arch=[256, 256, 256], n_critics=2), dict(net_arch=[512, 512, 512], n_critics=2)],
                'replay_buffer_class': [HerReplayBuffer],
                'replay_buffer_kwargs': [[dict(n_sampled_goal=4), dict(goal_selection_strategy='future')]]
            }

class Tuner:
    
    def __init__(self, environment) -> None:
        self.env = environment.env
        log_path = environment.log_path
        env_name = environment.environment_name
        self.log_path = os.path.join(log_path, 'Tuning')
        
        if env_name == 'FetchReach-v2':
            self.__parameters = ParameterDicts().FetchReach()
        if env_name == 'FetchPush-v2':
            self.__parameters = ParameterDicts().FetchPush()
        if env_name == 'FetchSlide-v2':
            self.__parameters = ParameterDicts().FetchSlide()
        if env_name == 'FetchPickAndPlace-v2':
            self.__parameters = ParameterDicts().FetchPickAndPlace()


    def tune(self, model_type, max_timesteps=1000000, max_iterations=10, max_time=3600, training_device="cuda", save_model=False):
        reward_tracker = RewardTrackerCallback(self.env)
        time_callback = TimeLimitCallback(time_limit=max_time)
        best_mean_reward = -float('inf')
        best_params = {}
        __temp_params = {}
        best_model = None
        best_models_iter = 0
        
        for iteration in range(max_iterations):
            if iteration != 0:
                time.sleep(600)
            callback_list = CallbackList([time_callback, reward_tracker])

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


                __temp_params = {
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


            elif model_type == 'TQC':
                model = TQC('MultiInputPolicy', self.env, train_freq=1, device=training_device, verbose=1,
                            tensorboard_log=self.log_path)

                model.learning_rate = random.choice(self.__parameters.TQC['learning_rate'])
                model.n_steps = random.choice(self.__parameters.TQC['n_steps'])
                model.batch_size = random.choice(self.__parameters.TQC['batch_size'])
                model.buffer_size = random.choice(self.__parameters.TQC['buffer_size'])
                model.learning_starts = random.choice(self.__parameters.TQC['learning_starts'])
                model.gamma = random.choice(self.__parameters.TQC['gamma'])
                model.tau = random.choice(self.__parameters.TQC['tau'])
                model.policy_kwargs = random.choice(self.__parameters.TQC['policy_kwargs'])
                model.replay_buffer_class = random.choice(self.__parameters.TQC['replay_buffer_class'])
                model.replay_buffer_kwargs = random.choice(self.__parameters.TQC['replay_buffer_kwargs'])


                __temp_params = {
                    'learning_rate': model.learning_rate,
                    'n_steps': model.n_steps,
                    'batch_size': model.batch_size,
                    'buffer_size': model.buffer_size,
                    'learning_starts': model.learning_starts,
                    'gamma': model.gamma,
                    'tau': model.tau,
                    'policy_kwargs': model.policy_kwargs,
                    'replay_buffer_class': model.replay_buffer_class,
                    'replay_buffer_kwargs': model.replay_buffer_kwargs
                }


            else:
                if model_type == 'DDPG':
                    model = DDPG('MultiInputPolicy', self.env, device=training_device, verbose=1,
                                 tensorboard_log=self.log_path)
                elif model_type == 'SAC':
                    model = SAC('MultiInputPolicy', self.env, device=training_device, verbose=1,
                                tensorboard_log=self.log_path)
                elif model_type == 'TD3':
                    model = TD3('MultiInputPolicy', self.env, device=training_device, verbose=1,
                                tensorboard_log=self.log_path)

                model.n_steps = random.choice(self.__parameters.OTHER['n_steps'])
                model.batch_size = random.choice(self.__parameters.OTHER['batch_size'])
                model.buffer_size = random.choice(self.__parameters.OTHER['buffer_size'])
                model.learning_starts = random.choice(self.__parameters.OTHER['learning_starts'])
                model.learning_rate = random.choice(self.__parameters.OTHER['learning_rate'])
                model.gamma = random.choice(self.__parameters.OTHER['gamma'])
                model.tau = random.choice(self.__parameters.OTHER['tau'])
                model.policy_kwargs = random.choice(self.__parameters.OTHER['policy_kwargs'])
                model.replay_buffer_class = random.choice(self.__parameters.OTHER['replay_buffer_class'])
                model.replay_buffer_kwargs = random.choice(self.__parameters.OTHER['replay_buffer_kwargs'])


                __temp_params = {
                    'learning_rate': model.learning_rate,
                    'n_steps': model.n_steps,
                    'batch_size': model.batch_size,
                    'buffer_size': model.buffer_size,
                    'learning_starts': model.learning_starts,
                    'gamma': model.gamma,
                    'tau': model.tau,
                    'policy_kwargs': model.policy_kwargs,
                    'replay_buffer_class': model.replay_buffer_class,
                    'replay_buffer_kwargs': model.replay_buffer_kwargs
                }

            model.learn(total_timesteps=max_timesteps, callback=callback_list)

            latest_run_id = get_latest_run_id(self.log_path, model.__class__.__name__) + 1
            # Check if the directory exists
            save_path = os.path.join(self.log_path, f'{model.__class__.__name__}_{latest_run_id}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(os.path.join(save_path, 'params.txt'), 'a') as f:
                f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                f.write(f'{time_callback.time_passed}')
                f.write(f'{__temp_params}\n')

            if save_model:
                model.save(os.path.join(save_path, 'model'))

            mean_reward = reward_tracker.get_mean_reward()
            if mean_reward > best_mean_reward or iteration == 0:
                best_params = __temp_params
                best_mean_reward = mean_reward
                best_model = model
                best_models_iter = get_latest_run_id(self.log_path, model.__class__.__name__) + 1
                best_models_iter = best_models_iter

            print(f'Best Parameters For Now: {best_params}\n')
            print(f'Best Mean Reward For Now: {best_mean_reward}\n')
        
        print(f'Best mean reward: {best_mean_reward}\n')
        print(f'Best parameters: {best_params}\n')

        # Check if the directory exists
        best_model_save_path = os.path.join(self.log_path, f'{best_model.__class__.__name__}_best_models')
        if not os.path.exists(best_model_save_path):
            os.makedirs(best_model_save_path)

        if save_model:
            with open(os.path.join(best_model_save_path, f'{best_model.__class__.__name__}_{best_models_iter}'), 'a') as f:
                f.write(f'Best Mean Reward: {best_mean_reward}\n')
                f.write(f'Best Parameters: {best_params}\n')
            best_model.save(os.path.join(best_model_save_path, f'{best_model.__class__.__name__}_{best_models_iter}'))

        return best_model, best_params
