import gym
import gym_conservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
r = 0.3
K = 1
# env = make_vec_env("fishing-v1", env_kwargs ={"r":r, "K":K}, n_envs=4)
env = gym.make("conservation-v0", r=r, K=K)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2500)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)

model.save("conservation0-PPO")

