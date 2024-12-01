
import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from ale_py import ALEInterface
ale = ALEInterface()
from ale_py import ALEInterface
ale = ALEInterface()
import ale_py
gym.register_envs(ale_py)

import gymnasium as gym
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as TF
from collections import deque
from collections import namedtuple
import wandb
import os
import warnings
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback



# ===========================
# Wandb Integration
# ===========================
# Initialize wandb
wandb.init(
    project="DQN-Atari-Kaboom",  # Name of your project in wandb
    config={
        "env_name": "ALE/Kaboom-v5",
        "total_timesteps": 1_000_000,
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 10000,
        "train_freq": 4,
        "gradient_steps": 1,
    },
)

# ===========================
# Main Function
# ===========================
if __name__ == "__main__":
    # Environment name
    ENV_NAME = "ALE/Kaboom-v5"

    # Create the environment
    env = gym.make(ENV_NAME)

    # Retrieve wandb config
    config = wandb.config

    # Define the DQN model with CNN policy
    model = DQN(
        "CnnPolicy",         # Policy with CNN for Atari environments
        env,                 # The environment
        verbose=1,           # Verbosity
        buffer_size=config.buffer_size,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        target_update_interval=config.target_update_interval,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        tensorboard_log="./logs/"  # Optional TensorBoard log directory
    )

    # Wandb Callback
    wandb_callback = WandbCallback(
        gradient_save_freq=100,   # Save gradients every 100 steps
        model_save_path="./wandb_models/",  # Directory to save models
        model_save_freq=10000,   # Save models every 10,000 steps
        verbose=1
    )

    # Train the model
    total_timesteps = config.total_timesteps
    print(f"Training DQN on {ENV_NAME} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=wandb_callback)

    # Save the final trained model
    model_path = "dqn_kaboom_final.zip"
    model.save(model_path)
    print(f"Final model saved as {model_path}")

    # Finish wandb logging
    wandb.finish()

    # Close the environment
    env.close()
