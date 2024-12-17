# Imports
import gymnasium as gym
import imageio
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

# configuration
config = {
    "env_name": "ALE/Galaxian-v5",
    "model_path": "/content/PPO_Galaxian.zip", #PUT THE PATH OF THE TRAINED MODEL 
    "videos_path": "./videos/",
}

#environment
def make_env(render_mode=None):
    env = gym.make(config["env_name"], render_mode=render_mode)
    return env

# we load the trained model
print("Loading the trained model...")
model = PPO.load(config["model_path"])
print(f"Model loaded from {config['model_path']}")

# evaluate the model
eval_env = DummyVecEnv([make_env])
eval_env = VecFrameStack(eval_env, n_stack=4)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# record gameplay as a video
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

        # we save the video
        os.makedirs(video_path, exist_ok=True)
        video_filename = os.path.join(video_path, f"episode_{episode}.mp4")
        imageio.mimsave(video_filename, frames, fps=30)
        print(f"Saved video: {video_filename}")

    env.close()

# record one episode for demonstration
print("Recording gameplay...")
record_video(model, config["videos_path"], num_episodes=1)

# close the evaluation environment
eval_env.close()
