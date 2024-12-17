import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
import imageio
import os

# Configuration file
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 100000,
    "env_name": "ALE/Galaxian-v5",
    "model_name": "PPO_Galaxian",
    "export_path": "./exports/",
    "videos_path": "./videos/",
    "learning_rate": 0.0002,  # Recommended for PPO
    "ent_coef": 0.02,        # PPO entropy coefficient
    "clip_range": 0.15,       # PPO clip range
    "n_steps": 4096,          # PPO steps per environment
    "batch_size": 64,        # PPO minibatch size
    "gae_lambda": 0.95,      # GAE lambda for PPO
    "gamma": 0.995,           # Discount factor
}

# Wandb setup
run = wandb.init(
    project="ProjectParadigmsGalaxian",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

# Define the environment with frame stacking
def make_env(render_mode=None):
    env = gym.make(config["env_name"], render_mode=render_mode)
    env = Monitor(env, allow_early_resets=False)  # Monitor wrapper for training
    return env

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.cumulative_rewards = 0.0  # Sum of all episode rewards
        self.num_episodes = 0          # Count of episodes

    def _on_step(self) -> bool:
        # Check if 'episode' key exists in the info dictionary
        if 'episode' in self.locals['infos'][0]:
            episode_rewards = self.locals['infos'][0]['episode']['r']
            self.cumulative_rewards += episode_rewards
            self.num_episodes += 1
            average_reward = self.cumulative_rewards / self.num_episodes

            print(f"Episode Reward: {episode_rewards}, Average Reward: {average_reward:.2f}")

            # Log episode and average reward to WandB
            wandb.log({
                "episode_reward": episode_rewards,
                "average_reward": average_reward
            })

        return True

# Create environment with frame stacking for training
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# Define and train the PPO model with custom hyperparameters
model = PPO(
    config["policy_type"],
    env,
    verbose=0,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=config["learning_rate"],
    ent_coef=config["ent_coef"],
    clip_range=config["clip_range"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    gae_lambda=config["gae_lambda"],
    gamma=config["gamma"]
)

# Train the model with the new callback
t0 = datetime.now()
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[RewardCallback(), WandbCallback(verbose=2)]
)
t1 = datetime.now()
print('>>> Training time (hh:mm:ss.ms): {}'.format(t1 - t0))

# Save and export the model
model.save(config["export_path"] + config["model_name"])

