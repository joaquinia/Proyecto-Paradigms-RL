import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Function to evaluate a trained PPO model
def evaluate_model(model_path, env_name="PongNoFrameskip-v4", n_episodes=10):
    env = make_atari_env(env_name, n_envs=1, seed=42)

    # Stack frames for better performance
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    model = PPO.load(model_path) 

    total_rewards = []

    # Evaluate the model for the selected episodes
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Predict the action using the trained model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()  

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = sum(total_rewards) / n_episodes
    print(f"Average Reward over {n_episodes} episodes: {avg_reward}")

    # Close the environment
    env.close()

# Path to the trained model
model_path = "ppo_pong"

# Evaluate the model
evaluate_model(model_path, n_episodes=10)
