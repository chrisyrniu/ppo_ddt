"""
Created by Andrew Silva on 3/26/21
"""
import gym
import numpy as np
from ppo_ddt.rl_helpers import DDTPolicy
from ppo_ddt.agents.vectorized_prolonet_helpers import convert_to_crisp
from ppo_ddt.rl_helpers.save_after_ep_callback import EpCheckPointCallback
import highway_env

from stable_baselines3 import SAC

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
#
set_all_seeds(42)
# eval_env = gym.make("LunarLanderContinuous-v2")
# eval_env = gym.make("InvertedPendulum-v2")
eval_env = gym.make("lane-keeping-v0")
save_folder = 'lk/submodels_hard_node_leaves_8_max_500'
callback = EpCheckPointCallback(eval_env=eval_env, best_model_save_path='../../' + save_folder + '/',
                                eval_freq=1500, minimum_reward=0)
# env = gym.make("LunarLanderContinuous-v2")
# env = gym.make("InvertedPendulum-v2")
env = gym.make("lane-keeping-v0")

# model = SAC("DDTPolicy", env,
#             learning_rate=3e-4,
#             buffer_size=1000000,
#             batch_size=256,
#             ent_coef='auto',
#             train_freq=1,
#             gradient_steps=1,
#             gamma=0.9999,
#             tau=0.01,
#             learning_starts=10000,
#             policy_kwargs={'net_arch': [400, 300]},
#             verbose=1)
# model.learn(total_timesteps=500000, log_interval=4, callback=callback)
# model.save("../../models/mlp_lunar")

# del model # remove to demonstrate saving and loading

model = SAC.load("../../" + save_folder + "/best_model")
set_all_seeds(5)
obs = env.reset()
episode_reward_for_reg = []
for _ in range(20):
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print('action', action)
        # print('obs', obs)
        episode_reward+= reward
        env.render()
        if done:
            obs = env.reset()
            episode_reward_for_reg.append(episode_reward)
            break
print(episode_reward_for_reg)
print(np.mean(episode_reward_for_reg))
print(np.std(episode_reward_for_reg))

model.actor.ddt = convert_to_crisp(model.actor.ddt, training_data=None)
set_all_seeds(5)
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
