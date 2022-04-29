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

space = {
    "optimization": {
          "actor_learning_rate": hp.loguniform("actor_learning_rate", 1e-4, 0.1),
          "critic_learning_rate": hp.loguniform("critic_learning_rate", 1e-4, 0.1),
          "entropy_learning_rate": hp.loguniform("entropy_learning_rate", 1e-4, 0.1),
      
    },
}

hyperopt_search = HyperOptSearch(space, mode="max")
config = {
    "env":"cons-v5",
    "num_workers": 2,
    "framework": "torch",
    "Q_model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 1,
    "evaluation_interval": 10,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}
# Create our RLlib Trainer.
# trainer = PPOTrainer(env="cons-v5", config=config)
tune.run("SAC", num_samples=5, config=config, search_alg=hyperopt_search, local_dir="/var/log/tensorboard/tim/ray/tune")
