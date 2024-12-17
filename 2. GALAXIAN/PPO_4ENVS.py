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

# Define the Custom WandB Callback
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_count = 0 

    def _on_step(self) -> bool:
        # Get episode rewards and lengths from the environment
        if "episode" in self.locals.get("infos", [{}])[0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            episode_length = self.locals["infos"][0]["episode"]["l"]

            self.episode_count += 1
            wandb.log({
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "episode": self.episode_count,
                "total_steps": self.num_timesteps
            })
        return True 

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

wandb.init(
    project="ProjectParadigms",
    entity="1665890", 
    config={
        "env_name": "ALE/Galaxian-v5",
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.99,
        "clip_range": 0.1,
        "total_timesteps": 7_500_000,
    }
)

# Environment Setup
env_name = "ALE/Galaxian-v5"
env = make_atari_env(env_name, n_envs=4, seed=42)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Initialize Model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_logs",
    learning_rate=wandb.config.learning_rate,
    n_steps=wandb.config.n_steps,
    batch_size=wandb.config.batch_size,
    n_epochs=wandb.config.n_epochs,
    gamma=wandb.config.gamma,
    clip_range=wandb.config.clip_range,
)

# Evaluation environment
eval_env = make_atari_env(env_name, n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

# Train the PPO Model with WandB logging
timesteps = wandb.config.total_timesteps

model.learn(
    total_timesteps=timesteps,
    callback=[eval_callback, WandbCallback(), CustomWandbCallback()] 
)

# Save the trained model
model.save("ppo_galaxian_wandb3")
print("Training complete. Model saved as ppo_galaxian_wandb_vm.zip.")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=50, deterministic=True
)

print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Record Video for proof
video_folder = "./videosVM/"

eval_env = VecVideoRecorder(
    eval_env, video_folder, record_video_trigger=lambda step: step == 0, video_length=5000
)

obs = eval_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    
eval_env.close()

print("Video of gameplay recorded in ./videosVM/")

wandb.finish()