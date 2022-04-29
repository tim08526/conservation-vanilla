# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer
import gym
import gym_conservation
# Configure the algorithm.
from ray.tune.registry import register_env

env = gym.make("conservation-v5")
print("environment made:", env)

register_env("cons-v5", lambda env_config: env)
config = {
    "num_workers": 2,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 1,
    "evaluation_interval":10,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}

# Create our RLlib Trainer.
trainer = PPOTrainer(env="cons-v5", config=config)

# 
# # Run it for n training iterations. A training iteration includes
# # parallel sample collection by the environment workers as well as
# # loss calculation on the collected batch and a model update.
for i in range(5000):
    trainer.train()
    # if i % 200 == 0:
    #    checkpoint = trainer.save()
    #    print("checkpoint saved at", checkpoint)

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
print(trainer.evaluate())
