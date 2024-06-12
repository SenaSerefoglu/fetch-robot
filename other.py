from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3 import PPO
from sb3_contrib import TQC
from models import Reach, Push, PickAndPlace, Slide
from env import Environment
import os

def continue_training_for_slide(model_path:str, log_path: str, buffer_path=None):
    env = Environment("FetchSlide-v2", max_episode_steps=100, render_mode='rgb_array').env
    model = TQC.load(model_path, env=env)

    logPathForBestModel = os.path.join(log_path, "bestModels")
    eval_callback = EvalCallback(
                        env, best_model_save_path=f'./{logPathForBestModel}/',
                        log_path=f'./{logPathForBestModel}/', eval_freq=1000000,
                        deterministic=True, render=False
                        )
    latest_run_id = get_latest_run_id(log_path, model.__class__.__name__) + 1
    logPathForCheckpoint = os.path.join(log_path, "checkpoints")
    checkpoint_callback = CheckpointCallback(
                        save_freq=500000,
                        save_path=logPathForCheckpoint,
                        name_prefix=f'{model.__class__.__name__}_{latest_run_id}',
                        save_replay_buffer=True,
                        save_vecnormalize=True,
                        )
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    model = TQC.load(model_path, env=env, tensorboard_log=log_path)
    model.load_replay_buffer(buffer_path)
    model.learn(11500000, reset_num_timesteps=False, callback=callback_list)


def reach_learning_rate(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.learning_rate = 0.0001    # default is 0.0003
    
    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_learning_rate"))

def reach_learning_rate2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.learning_rate = 0.01    # default is 0.0003
    
    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_learning_rate2"))


def reach_buffer_size(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.buffer_size = 100000  # default is 1000000

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_buffer_size"))

def reach_buffer_size2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.buffer_size = 10000000  # default is 1000000

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_buffer_size2"))


def reach_learning_starts(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.learning_starts = 100   # default is 100

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_learning_starts"))

def reach_learning_starts2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.learning_starts = 10000   # default is 100

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_learning_starts2"))


def reach_batch_size(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.batch_size = 512  # default is 256

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_batch_size"))

def reach_batch_size2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.batch_size = 128  # default is 256

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_batch_size2"))


def reach_tau(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.tau = 0.01    # default is 0.005

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_tau"))

def reach_tau2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.tau = 0.001    # default is 0.005

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_tau2"))


def reach_gamma(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.gamma = 0.99  # default is 0.99

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_gamma"))

def reach_gamma2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.gamma = 0.91  # default is 0.99

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_gamma2"))

def reach_gradient_steps(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.gradient_steps = 64   # default is 1

    reach.learn(100000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_gradient_steps"))


def reach_gradient_steps2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.gradient_steps = 32   # default is 1

    reach.learn(100000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_gradient_steps2"))

def reach_ent_coef(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.ent_coef = 0.1    # default is 'auto'

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_ent_coef"))

def reach_ent_coef2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.ent_coef = 0.5    # default is 'auto'

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_ent_coef2"))


def reach_target_update_interval(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.target_update_interval = 1000 # default is 1

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_target_update_interval"))

def reach_target_update_interval2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.target_update_interval = 100 # default is 1

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_target_update_interval2"))


def reach_target_entropy(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.target_entropy = 0.1  # default is 'auto'

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_target_entropy"))

def reach_target_entropy2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.target_entropy = 0.5  # default is 'auto'

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_target_entropy2"))


def reach_top_quantiles_to_drop_per_net(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.top_quantiles_to_drop_per_net = 1 # default is 2

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_top_quantiles_to_drop_per_net"))

def reach_top_quantiles_to_drop_per_net2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.top_quantiles_to_drop_per_net = 3 # default is 2

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_top_quantiles_to_drop_per_net2"))

def reach_stats_window_size(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.stats_window_size = 1000  # default is 100

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_stats_window_size"))

def reach_stats_window_size2(env, log_path):
    reach = Reach(env, log_path).TQCmodel
    reach.stats_window_size = 50  # default is 100

    reach.learn(500000)
    reach.save(os.path.join(log_path, "parameter_models", "reach_stats_window_size2"))


if __name__ == "__main__":
    """environment = Environment("FetchReach-v2", max_episode_steps=50, render_mode='rgb_array')
    reach_learning_starts(environment.env, environment.log_path)"""
    pass