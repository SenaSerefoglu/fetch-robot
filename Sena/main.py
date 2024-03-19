import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboard import program

environment_name = 'FetchPushDense-v2'
env = gym.make(environment_name, max_episode_steps=100, render_mode='human')
log_path = os.path.join('Training', 'Logs')

env = DummyVecEnv([lambda: env])
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path, device='cpu')
model.learn(total_timesteps=20000)

PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchPushDense-v2')
model.save(PPO_Path)

env = gym.make(environment_name, render_mode='human', max_episode_steps=100)
evaluate_policy(model, env, n_eval_episodes=50, render=True,)
env.close()

# TensorBoard Usage
training_log_path = os.path.join(log_path, 'PPO_14')

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', training_log_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press any key to exit...")