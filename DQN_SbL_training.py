
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

warnings.filterwarnings("ignore")

# ===========================
# Main Function
# ===========================
if __name__ == "__main__":
    # Environment name
    ENV_NAME = "ALE/Kaboom-v5"

    # Create the environment
    env = gym.make(ENV_NAME)

    # Define the DQN model with CNN policy
    model = DQN(
        "CnnPolicy",         # Policy with CNN for Atari environments
        env,                 # The environment
        verbose=1,           # Verbosity (set to 0 for silent training)
        buffer_size=100000,  # Replay buffer size
        learning_rate=1e-4,  # Learning rate
        batch_size=32,       # Batch size
        gamma=0.99,          # Discount factor
        exploration_fraction=0.1,  # Fraction of exploration
        exploration_final_eps=0.01,  # Minimum epsilon for exploration
        target_update_interval=10000,  # Update target network every 10k steps
        train_freq=4,        # Train every 4 steps
        gradient_steps=1,    # Gradient steps per training step
        tensorboard_log="./logs/"  # TensorBoard log directory
    )

    # Add a callback for saving checkpoints during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,         # Save every 10,000 steps
        save_path="./checkpoints/",  # Directory to save models
        name_prefix="dqn_kaboom"     # Prefix for the saved models
    )

    # Train the model
    total_timesteps = 1_000_000  # Set the number of timesteps
    print(f"Training DQN on {ENV_NAME} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final trained model
    model.save("dqn_kaboom_final")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward after training: {mean_reward} ± {std_reward}")

    # Close the environment
    env.close()
