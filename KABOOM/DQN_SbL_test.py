import gymnasium as gym
from stable_baselines3 import DQN
import torch
import wandb
from PIL import Image
import numpy as np

# ===========================
# Watch Agent Function
# ===========================
def watch_agent(env, model, max_steps=1000, save_as_gif=True, gif_name="agent_performance.gif"):
    obs, info = env.reset()
    total_reward = 0
    images = []

    for step in range(max_steps):
        # Get Q-values and choose the best action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

        # Accumulate reward
        total_reward += reward

        # Capture the rendered frame
        img = env.render()
        images.append(Image.fromarray(img))

        # If the episode is finished
        if done or truncated:
            break

    # Save the performance as a GIF
    if save_as_gif and images:
        images[0].save(
            gif_name,
            save_all=True,
            append_images=images[1:],
            duration=60,  # Frame duration in milliseconds
            loop=0
        )
        print(f"Performance video saved as {gif_name}")

    print(f"Total Reward: {total_reward}")
    env.close()
    return total_reward


# ===========================
# Main Testing Script
# ===========================
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="DQN-Atari-Kaboom",
        name="Model Testing",
        config={
            "env_name": "ALE/Kaboom-v5",
            "max_steps": 1000,
            "model_path": "./dqn_kaboom_final.zip",  # Path to the saved model
        },
    )

    # Retrieve configuration
    config = wandb.config

    # Create the environment
    ENV_NAME = config.env_name
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # Load the trained model
    model = DQN.load(config.model_path)
    print(f"Loaded model from {config.model_path}")

    # Evaluate the model
    print("Evaluating the model...")
    total_reward = watch_agent(
        env, model, max_steps=config.max_steps, save_as_gif=True, gif_name="dqn_kaboom_test.gif"
    )

    # Log results to wandb
    wandb.log({"total_reward": total_reward})

    # Finish wandb logging
    wandb.finish()
