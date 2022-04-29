import gym
import gym_conservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
env = gym.make("conservation-v5")

def objective(trial):
  
  hyperparams = {}
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=5000)
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
  return mean_reward


