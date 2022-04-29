# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import gym
import gym_conservation
# Configure the algorithm.
from ray.tune.registry import register_env

env = gym.make("conservation-v5")
print("environment made:", env)

register_env("cons-v5", lambda env_config: env)
config = {
    "env":"cons-v5",
    "num_workers": 2,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}
# Create our RLlib Trainer.
# trainer = PPOTrainer(env="cons-v5", config=config)
tune.run("PPO", config=config, local_dir="/var/log/tensorboard/tim/ray/tune")
# trainer.restore("../../ray_results/PPOTrainer_cons-v5_2022-04-18_19-54-42y_1ck4c1/checkpoint_004801/checkpoint-4801")
