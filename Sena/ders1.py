import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'FetchPushDense-v2'
env = gym.make(environment_name, max_episode_steps=100, render_mode='human')

log_path = os.path.join('Training', 'Logs')
env = DummyVecEnv([lambda: env])
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path, device='cpu')
model.learn(total_timesteps=2000000)

PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_FetchPushDense-v2')
model.save(PPO_Path)


env = gym.make(environment_name, render_mode='human', max_episode_steps=100)
evaluate_policy(model, env, n_eval_episodes=50, render=True,)
env.close()


#TensorBoard Kullanımı

training_log_path = os.path.join(log_path, 'PPO_14')

from tensorboard import program

tracking_address = training_log_path # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
input("Press any key to exit...")


"""episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, terminated, info = env.step(action)
        score += reward
    print('Episode {}\tScore: {}'.format(episode, score))
    print(env.step(action))
env.close()"""


"""del model
PPO_Path"""
"""model.learn(total_timesteps=1000)"""
"""print(log_path, PPO_Path)
model = PPO.load(PPO_Path, env=env, tensorboard_log=log_path)"""



"""vec_env = model.get_env()
obs = vec_env.reset()
terminated = False
while not terminated:
    action, _states = model.predict(obs)
    env.render()
    obs, rewards, terminated, info = env.step(action)"""


"""episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        print(done)
    print('Episode {}\tScore: {}'.format(episode, score))
    print(env.step(action))
env.close()"""