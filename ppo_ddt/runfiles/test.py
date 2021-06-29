"""
Created by Yaru Niu
"""
import gym
import numpy as np
import random
import os
import torch
from ppo_ddt.rl_helpers import DDTPolicy
from ppo_ddt.agents.vectorized_prolonet_helpers import convert_to_crisp
from ppo_ddt.rl_helpers.save_after_ep_callback import EpCheckPointCallback
import highway_env

from stable_baselines3 import PPO

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# env_kwargs = {
#     'id': 'highway-v0',
#     'config': {
#         "lanes_count": 3,
#         "vehicles_count": 15,
#         "observation": {
#             "type": "Kinematics",
#             "vehicles_count": 10,
#             "features": [
#                 "presence",
#                 "x",
#                 "y",
#                 "vx",
#                 "vy",
#                 "cos_h",
#                 "sin_h"
#             ],
#             "absolute": False
#         },
#         "policy_frequency": 2,
#         "duration": 40,
#     }
# }

env_kwargs = {
    'id': 'highway-v0',
    'config': {
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 4,
            "features": [
                "x",
                "y",
                "vx",
                "vy"
            ],
            "absolute": False
        },
        "policy_frequency": 2,
        "duration": 40,
    }
}

save_folder = 'saved_models/ppo_ddt/small_env/run1'

model = PPO.load("../../" + save_folder + "/best_model")
env = make_configure_env(**env_kwargs)

set_all_seeds(0)
obs = env.reset()
episode_reward_for_reg = []
for _ in range(20):
    done = False
    episode_reward = 0
    t = 0
    while not done:
        t += 1
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # print('action', action)
        # print('obs', obs)
        episode_reward+= reward
        env.render()
        if done:
            obs = env.reset()
            episode_reward_for_reg.append(episode_reward)
            print('duration', t)
            break
print(episode_reward_for_reg)
print(np.mean(episode_reward_for_reg))
print(np.std(episode_reward_for_reg))

model.actor.ddt = convert_to_crisp(model.actor.ddt, training_data=None)
set_all_seeds(0)
obs = env.reset()
discrete_episode_reward_for_reg = []
for _ in range(20):
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print('action', action)
        # print('obs', obs)
        episode_reward += reward
        env.render()
        if done:
            obs = env.reset()
            discrete_episode_reward_for_reg.append(episode_reward)
            break
print(discrete_episode_reward_for_reg)
print(np.mean(discrete_episode_reward_for_reg))
print(np.std(discrete_episode_reward_for_reg))
