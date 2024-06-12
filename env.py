from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_latest_run_id
from models import Reach, Push, PickAndPlace, Slide
import gymnasium as gym
import os


class Environment:
    def __init__(self, environment_name, max_episode_steps, render_mode):
        self.environment_name = environment_name
        self.env = gym.make(self.environment_name, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self.env = DummyVecEnv([lambda: self.env])
        self.log_path = os.path.join('Logs', environment_name)
        self.callback_list = CallbackList([])
        
    def define_model(self, model_type):
        if self.environment_name == "FetchReach-v2" or self.environment_name == "FetchReachDense-v2":
            reach = Reach(self.env, self.log_path)

            if model_type == "PPO":
                self.model = reach.PPOmodel

            elif model_type == "TQC":
                self.model = reach.TQCmodel

            elif model_type == "SAC":
                self.model = reach.SACmodel

            elif model_type == "DDPG":
                self.model = reach.DDPGmodel
            
            elif model_type == "TD3":
                self.model = reach.TD3model
            

        elif self.environment_name == "FetchPush-v2" or self.environment_name == "FetchPushDense-v2":
            push = Push(self.env, self.log_path)

            if model_type == "TQC":
                self.model = push.TQCmodel

            elif model_type == "PPO":
                self.model = push.PPOmodel

            elif model_type == "SAC":
                self.model = push.SACmodel

            elif model_type == "DDPG":
                self.model = push.DDPGmodel

            elif model_type == "TD3":
                self.model = push.TD3model


        elif self.environment_name == "FetchSlide-v2" or self.environment_name == "FetchSlideDense-v2":
            slide = Slide(self.env, self.log_path)

            if model_type == "TQC":
                self.model = slide.TQCmodel
                
                logPathForBestModel = os.path.join(self.log_path, "bestModels")
                eval_callback = EvalCallback(
                                    self.env, best_model_save_path=f'./{logPathForBestModel}/',
                                    log_path=f'./{logPathForBestModel}/', eval_freq=10000,
                                    deterministic=True, render=False
                                    )
                latest_run_id = get_latest_run_id(self.log_path, self.model.__class__.__name__) + 1
                logPathForCheckpoint = os.path.join(self.log_path, "checkpoints")
                checkpoint_callback = CheckpointCallback(
                                    save_freq=500000,
                                    save_path=logPathForCheckpoint,
                                    name_prefix=f'{self.model.__class__.__name__}_{latest_run_id}',
                                    save_replay_buffer=True,
                                    save_vecnormalize=True,
                                    )
                self.callback_list = CallbackList([eval_callback, checkpoint_callback])

            elif model_type == "PPO":
                self.model = slide.PPOmodel
            
            elif model_type == "DDPG":
                self.model = slide.DDPGmodel

            elif model_type == "SAC":
                self.model = slide.SACmodel

            elif model_type == "TD3":
                self.model = slide.TD3model
        

        elif self.environment_name == "FetchPickAndPlace-v2" or self.environment_name == "FetchPickAndPlaceDense-v2":
            pickAndPlace = PickAndPlace(self.env, self.log_path)

            if model_type == "TQC":
                self.model = pickAndPlace.TQCmodel

            elif model_type == "PPO":
                self.model = pickAndPlace.PPOmodel

            elif model_type == "SAC":
                self.model = pickAndPlace.SACmodel

            elif model_type == "DDPG":
                self.model = pickAndPlace.DDPGmodel

            elif model_type == "TD3":
                self.model = pickAndPlace.TD3model

    def train_model(self, timesteps=1000000):        
        self.model.learn(total_timesteps=timesteps, callback=self.callback_list)
    
    def save_model(self): 
        modelPath = os.path.join(self.log_path, 'Saved Models')
        self.model.save(modelPath)