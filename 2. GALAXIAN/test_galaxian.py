import gymnasium as gym
import ale_py
from ale_py import ALEInterface
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


env_name = "ALE/Galaxian-v5"  

# Setup evaluation callback
eval_env = make_atari_env(env_name, n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

print("Loading the model...")
# Load and Evaluate the Trained Model
model = PPO.load(
    "ppo_galaxian_wandb3",
    custom_objects={
        "clip_range": 0.1, 
        "lr_schedule": lambda _: 2.5e-4  
    }
)
print("...Model loaded successfully.")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=50, deterministic=True
)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

video_folder = "./videos_test/"
eval_env = VecVideoRecorder(eval_env, video_folder, record_video_trigger=lambda step: step == 0, video_length=2000)
obs = eval_env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
eval_env.close()

print("Video of gameplay recorded in ./videos_test/")