from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer, PPO, SAC, DDPG, TD3


class Reach:
    def __init__(self, env, log_path):
        self.TQCmodel = TQC('MultiInputPolicy', env, learning_rate=0.001, buffer_size=1000000, learning_starts=1000, 
                        batch_size=256, gamma=0.95, ent_coef='auto', 
                        policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
                        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                        replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_path)
        
        self.PPOmodel = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)

        self.SACmodel = SAC('MultiInputPolicy', env, buffer_size=1000000, ent_coef='auto', batch_size=256,
                            gamma=0.95, learning_rate=0.001, learning_starts=1000, replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs= dict(goal_selection_strategy='future', n_sampled_goal=4), policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=1, tensorboard_log=log_path)

        self.DDPGmodel = DDPG('MultiInputPolicy', env, buffer_size=1000000, batch_size=256,
                            gamma=0.95, learning_rate=0.001, learning_starts=1000, replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs= dict(goal_selection_strategy='future', n_sampled_goal=4), policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=1, tensorboard_log=log_path)

        self.TD3model = TD3('MultiInputPolicy', env, buffer_size=1000000, batch_size=256,
                            gamma=0.95, learning_rate=0.001, learning_starts=1000, replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs= dict(goal_selection_strategy='future', n_sampled_goal=4), policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=1, tensorboard_log=log_path)


class Push:
    def __init__(self, env, log_path):
        self.PPOmodel = PPO('MultiInputPolicy', env, verbose=1, batch_size=512, tensorboard_log=log_path)

        self.TQCmodel = TQC('MultiInputPolicy',env, batch_size=1024, gamma=0.95, tau=0.005, train_freq=1, target_entropy=0.01,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 256], n_critics=2), replay_buffer_class=HerReplayBuffer, buffer_size=1000000,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)
        
        self.DDPGmodel = DDPG('MultiInputPolicy', env, batch_size=1024, gamma=0.95, tau=0.005, buffer_size=1000000, train_freq=1,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 256], n_critics=2), replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)

        self.SACmodel = SAC('MultiInputPolicy', env, batch_size=2048, gamma=0.95, tau=0.05, buffer_size=1000000,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
                            replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1,
                            tensorboard_log=log_path)

        self.TD3model = TD3('MultiInputPolicy', env, batch_size=1024, gamma=0.95, tau=0.005, train_freq=1,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 256], n_critics=2), replay_buffer_class=HerReplayBuffer, buffer_size=1000000,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)
            

class PickAndPlace:
    def __init__(self, env, log_path):
        self.PPOmodel = PPO('MultiInputPolicy', env, verbose=1, batch_size=1024, tensorboard_log=log_path)

        self.TQCmodel = TQC('MultiInputPolicy', env, learning_rate=0.001, batch_size=1024, gamma=0.98, tau=0.005, train_freq=1,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)
        
        self.DDPGmodel = DDPG('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                              verbose=1, tensorboard_log=log_path)

        self.SACmodel = SAC('MultiInputPolicy', env, device="cpu", learning_rate=0.01, gamma=0.95, tau=0.05, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
                            batch_size=512, buffer_size=1000000, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                            verbose=1, tensorboard_log=log_path)

        self.TD3model = TD3('MultiInputPolicy', env, learning_rate=0.001, batch_size=1024, gamma=0.98, tau=0.005, train_freq=1,
                            learning_starts=1000, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2), replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), verbose=1, tensorboard_log=log_path)

        
class Slide:
    def __init__(self, env, log_path):
        self.TQCmodel = TQC('MultiInputPolicy', env, learning_rate=0.001, buffer_size=1000000, learning_starts=1000, batch_size=2048,
                            tau=0.05, gamma=0.95, ent_coef=0.01, policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
                            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                            replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_path)
        
        self.PPOmodel = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.001, batch_size=1024, gamma=0.95,
            policy_kwargs=dict(net_arch=[512, 512]), tensorboard_log=log_path)
        
        self.DDPGmodel = DDPG('MultiInputPolicy', env, verbose=1, learning_rate=0.001, batch_size=1024, gamma=0.95, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512], n_critics=2),
            replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4,
            goal_selection_strategy='future'), tensorboard_log=log_path)
        
        self.SACmodel = SAC('MultiInputPolicy', env, verbose=1, learning_rate=0.001, batch_size=1024, gamma=0.95,tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
            tensorboard_log=log_path)

        self.TD3model = TD3('MultiInputPolicy', env, verbose=1, learning_rate=0.001, batch_size=1024, gamma=0.95, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512], n_critics=2),
            replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
            tensorboard_log=log_path)