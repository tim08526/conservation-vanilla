from ray import tune
import gym
import gym_conservation
# Configure the algorithm.
from ray.tune.registry import register_env
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch



env = gym.make("conservation-v5")
print("environment made:", env)

register_env("cons-v5", lambda env_config: env)


# hyperopt_search = HyperOptSearch(space, mode="max")
config = {
    "env":"cons-v5",
    "num_workers": 2,
    "framework": "torch",
    "actor_hiddens": [64,64],
    "critic_hiddens": [64,64],
    "evaluation_num_workers": 1,
    "evaluation_interval": 10,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}
# Create our RLlib Trainer.
# trainer = PPOTrainer(env="cons-v5", config=config)
tune.run("TD3", num_samples=5, config=config, local_dir="/var/log/tensorboard/tim/ray/tune")
