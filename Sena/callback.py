from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
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
    def __init__(self, eval_env, verbose=0, eval_freq: int = 10000, n_eval_episodes: int = 5, deterministic: bool = True,):
        super().__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.episode_rewards = []
        self.eval_env = eval_env

    def _on_training_start(self) -> None:
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env, 
                                                               n_eval_episodes=self.n_eval_episodes, 
                                                               deterministic=self.deterministic, return_episode_rewards=True)
            self.episode_rewards.extend(episode_rewards)
            
        return True
    
    def get_mean_reward(self):
        mean_reward = np.mean(self.episode_rewards)
        return mean_reward   