import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import DataLoader
from tensorboard import program

# Assume you have a DataLoader for your dataset
# Replace this with your actual DataLoader setup
# DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

environment_name = 'FetchPushDense-v2'
env = gym.make(environment_name, max_episode_steps=100, render_mode='human')
log_path = os.path.join('Training', 'Logs')

env = DummyVecEnv([lambda: env])
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)

# DataLoader for efficient data loading
data_loader = DataLoader(env, batch_size=64, shuffle=True, num_workers=4)

# Training loop with DataLoader
for batch in data_loader:
    obs, actions, rewards, dones, next_obs = batch

    # Perform training using the batch data
    model.learn(total_timesteps=20000, log_interval=10)  # Adjust log_interval as needed

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