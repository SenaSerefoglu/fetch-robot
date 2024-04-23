from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer, PPO, DDPG, A2C, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
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
                'learning_rate': [3e-4, 1e-4, 5e-5],
                'n_steps': [512, 1024],
                'batch_size': [128, 256],
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
                'n_steps': [2048], #Ortamın karmaşıklığına göre arttırmak daha doğru ki daha çok veri toplayarak öğrensin ama zaman maliyeti oluşturuyor.
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e6), int(1e7)],
                'learning_starts': [int(1e3), int(1e4)],
                #use sde kullanımı
                #'use_sde': [True, False],
                #'sde_sample_freq': [4, 8],
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
                'n_steps': [1024, 2048], #Ortamın karmaşıklığına göre arttırmak daha doğru ki daha çok veri toplayarak öğrensin ama zaman maliyeti oluşturuyor.
                'batch_size': [1024, 2048],
                'buffer_size': [int(1e7), int(1e6)],
                'learning_starts': [1e3, 1e4],
                #use sde kullanımı
                #'use_sde': [True, False],
                #'sde_sample_freq': [4, 8],
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
        self.best_mean_reward = -float('inf')
        self.best_params = {}
        self.__temp_params = {}
        self.best_model = None
        
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
        
        for iteration in range(max_iterations):
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=4, verbose=1)
            eval_callback = EvalCallback(self.env, eval_freq=50000, callback_after_eval=stop_train_callback, verbose=0)
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
                model = TQC('MultiInputPolicy', self.env, train_freq=1, device=training_device, verbose=1,
                            tensorboard_log=self.log_path)

                model.learning_rate = random.choice(self.__parameters.TQC['learning_rate'])
                model.n_steps = random.choice(self.__parameters.TQC['n_steps'])
                model.batch_size = random.choice(self.__parameters.TQC['batch_size'])
                model.buffer_size = random.choice(self.__parameters.TQC['buffer_size'])
                model.learning_starts = random.choice(self.__parameters.TQC['learning_starts'])
                #model.use_sde = random.choice(self.__parameters.TQC['use_sde'])
                #model.sde_sample_freq = random.choice(self.__parameters.TQC['sde_sample_freq'])
                model.gamma = random.choice(self.__parameters.TQC['gamma'])
                model.tau = random.choice(self.__parameters.TQC['tau'])
                model.policy_kwargs = random.choice(self.__parameters.TQC['policy_kwargs'])
                model.replay_buffer_class = random.choice(self.__parameters.TQC['replay_buffer_class'])
                model.replay_buffer_kwargs = random.choice(self.__parameters.TQC['replay_buffer_kwargs'])
                print(f"REPLAY BUFFER KWARGS => {model.policy_kwargs}")

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
                    'learning_rate': model.learning_rate,
                    'n_steps': model.n_steps,
                    'batch_size': model.batch_size,
                    'buffer_size': model.buffer_size,
                    'learning_starts': model.learning_starts,
                    #'use_sde': model.use_sde,
                    #'sde_sample_freq': model.sde_sample_freq,
                    'gamma': model.gamma,
                    'tau': model.tau,
                    'policy_kwargs': model.policy_kwargs,
                    'replay_buffer_class': model.replay_buffer_class,
                    'replay_buffer_kwargs': model.replay_buffer_kwargs
                }

                with open(os.path.join(self.log_path, f'TQC_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'TQC_{iteration + 1}', 'model'))

            if model_type == 'DDPG':
                model = DDPG('MultiInputPolicy', self.env, device=training_device, verbose=1,
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

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
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

                with open(os.path.join(self.log_path, f'DDPG_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'DDPG_{iteration + 1}', 'model'))


            if model_type == 'SAC':
                model = SAC('MultiInputPolicy', self.env, device=training_device, verbose=1, tensorboard_log=self.log_path)

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

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
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

                with open(os.path.join(self.log_path, f'SAC_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'SAC_{iteration + 1}', 'model'))


            if model_type == 'A2C':
                model = A2C('MultiInputPolicy', self.env, device=training_device, verbose=1, tensorboard_log=self.log_path)

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

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
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

                with open(os.path.join(self.log_path, f'A2C_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'A2C_{iteration + 1}', 'model'))

            if model_type == 'TD3':
                model = TD3('MultiInputPolicy', self.env, device=training_device, verbose=1, tensorboard_log=self.log_path)

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

                model.learn(total_timesteps=max_timesteps, callback=callback_list)

                self.__temp_params = {
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

                with open(os.path.join(self.log_path, f'TD3_{iteration + 1}', 'params.txt'), 'a') as f:
                    f.write(f'Mean Reward: {reward_tracker.get_mean_reward()}\n')
                    f.write(f'{time_callback.time_passed}')
                    f.write(f'{self.__temp_params}\n')

                if save_model:
                    model.save(os.path.join(self.log_path, f'TD3_{iteration + 1}', 'model'))

            mean_reward = reward_tracker.get_mean_reward()
            if mean_reward > self.best_mean_reward:
                self.best_params = self.__temp_params
                self.best_mean_reward = mean_reward
                self.best_model = model

            print(f'Best Parameters For Now: {self.best_params}\n')
            print(f'Best Mean Reward For Now: {self.best_mean_reward}\n')
            time.sleep(600)
        
        print(f'Best mean reward: {self.best_mean_reward}\n')
        print(f'Best parameters: {self.best_params}\n')

        if save_model:
            with open(os.path.join(self.log_path, f'{type(self.best_model)}_best_model', 'params.txt'), 'a') as f:
                f.write(f'Best Mean Reward: {self.best_mean_reward}\n')
                f.write(f'Best Parameters: {self.best_params}\n')
            self.best_model.save(os.path.join(self.log_path, f'{type(self.best_model)}_best_model', 'model'))

        return self.best_model, self.best_params 
