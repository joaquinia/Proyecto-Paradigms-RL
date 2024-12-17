# Imports
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import os

#TEST FOR ACTOR CRITIC

# Configuration
config = {
    "env_name": "ALE/Galaxian-v5",
    "model_path": "/content/A2C_Galaxian (1).zip", #PUT THE CORRESPONDING PATH TO UPLOAD THE MODEL FROM TRAINING!!!
    "videos_path": "./videos/"
}

#environment creation function
def make_env(render_mode=None):
    env = gym.make(config["env_name"], render_mode=render_mode)
    return env

#load the trained model
print("Loading the trained model...")
model = A2C.load(config["model_path"])
print(f"Model loaded from {config['model_path']}")

#environment
eval_env = DummyVecEnv([make_env])
eval_env = VecFrameStack(eval_env, n_stack=4)

#evaluate the model
print("Evaluating the model...")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# recording of the gameplay as a video
def record_video(model, video_path, num_episodes=1):
    raw_env = make_env(render_mode="rgb_array")
    env = VecFrameStack(DummyVecEnv([lambda: raw_env]), n_stack=4)

    for episode in range(num_episodes):
        obs = env.reset()
        frames = []
        done = False
        while not done:
            frames.append(raw_env.render())  # Capture frames
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

        # Save the video
        os.makedirs(video_path, exist_ok=True)
        video_filename = os.path.join(video_path, f"episode_{episode}.mp4")
        imageio.mimsave(video_filename, frames, fps=30)
        print(f"Saved video: {video_filename}")

    env.close()

#episode for demonstration
print("Recording gameplay...")
record_video(model, config["videos_path"], num_episodes=1)

# Close environment
eval_env.close()
