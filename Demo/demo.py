import os
import gymnasium as gym
from stable_baselines3 import PPO, TD3, SAC, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy


class demo:

    def __init__(self, environment_name, file_name, algorithm):
        self.environment_name = environment_name
        self.file_name = file_name
        self.algorithm = algorithm

        self.log_path = os.path.join('Logs', self.file_name)
        self.model_path = os.path.join('Logs', self.file_name, 'Saved Models')
        model, env = self.load_model()

        print(self.algorithm)
        self.evaluate_model(model, env)

    def load_model(self):
        env = gym.make(self.environment_name, render_mode='human', max_episode_steps=50)
        model = self.algorithm.load(self.model_path, env=env, tensorboard_log=self.log_path)
        return model, env

    def evaluate_model(self, model, env):
        evaluate_policy(model, env, n_eval_episodes=50, render=True)
        env.close()


if __name__ == '__main__':
    demo('FetchReach-v2', 'Reach_PPO', PPO)
