# Imports
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
import wandb

#TRAINING FOR ACTOR CRITIC

# Configuration
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 100000,
    "env_name": "ALE/Galaxian-v5",
    "model_name": "A2C_Galaxian",
    "export_path": "./exports/",
    "learning_rate": 0.0003,  # parameter that we have changed in order to perform hyperparameter fine-tuning
    "ent_coef": 0.03, # entropy coefficient (also parameter to change)
}

# Wandb setup
run = wandb.init(
    project="ProjectParadigmsGalaxian",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

# we create the environment 
def make_env(render_mode=None):
    env = gym.make(config["env_name"], render_mode=render_mode)
    env = Monitor(env, allow_early_resets=False)  #wrapper for training
    return env

#reward logging callback
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.cumulative_rewards = 0.0  # Sum of all episode rewards
        self.num_episodes = 0  # Count of episodes

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            episode_rewards = self.locals['infos'][0]['episode']['r']
            self.cumulative_rewards += episode_rewards
            self.num_episodes += 1
            average_reward = self.cumulative_rewards / self.num_episodes

            print(f"Episode Reward: {episode_rewards}, Average Reward: {average_reward:.2f}")
            wandb.log({"episode_reward": episode_rewards, "average_reward": average_reward})
        
        return True

# create environment
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# model definition
model = A2C(
    config["policy_type"],
    env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=config["learning_rate"],
    ent_coef=config["ent_coef"]
)

# Training
print("Training started...")
t0 = datetime.now()
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[RewardCallback()]
)
t1 = datetime.now()
print('>>> Training time (hh:mm:ss.ms): {}'.format(t1 - t0))

# Save model
os.makedirs(config["export_path"], exist_ok=True)
model.save(config["export_path"] + config["model_name"])
print(f"Model saved to {config['export_path'] + config['model_name']}")

# Close environment
env.close()