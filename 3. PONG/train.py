import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

wandb.init(project="ProjectParadigmsPong", entity="1665890")

# Custom callback to log metrics to wandb
class WandbCallback(BaseCallback):
    def __init__(self):
        super(WandbCallback, self).__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_steps += 1
        if "episode" in self.locals["infos"][0]:
            # Extract episode information
            episode_info = self.locals["infos"][0]["episode"]
            episode_reward = episode_info["r"]
            episode_length = episode_info["l"]
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1

            # Calculate the average reward
            average_reward = sum(self.episode_rewards) / len(self.episode_rewards)

            # Log metrics to wandb
            wandb.log({
                "average_reward": average_reward,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "episode": self.episode_count,
                "total_steps": self.total_steps
            })

            # Episode metrics
            print(f"Episode {self.episode_count}: Reward = {episode_reward}, Length = {episode_length}")

        return True

    def _on_rollout_end(self) -> None:
        # Log loss metrics at the end of each rollout
        if "loss" in self.locals:
            wandb.log({"loss": self.locals["loss"]})

# Create environment (NoFrameskip version)
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=42)

# Stack frames for better performance
env = VecFrameStack(env, n_stack=4)

# Instantiate the PPO model
model = PPO(
    "CnnPolicy",  # Convolutional Neural Network policy
    env,           # Environment
    verbose=1,     # Display training logs
    learning_rate=2.5e-4,  # Standard learning rate for PPO
    n_steps=128,   # Number of steps per update
    batch_size=256, # Minibatch size
    n_epochs=4,    # Number of epochs
    gamma=0.99,    # Discount factor
    gae_lambda=0.95,  # GAE parameter
    clip_range=0.1,  # Clip range for PPO
    tensorboard_log="./ppo_pong_tensorboard/"  # TensorBoard log directory
)

# Train the agent with WandbCallback
model.learn(total_timesteps=int(1e7), callback=WandbCallback())  # Train for 10 million timesteps

# Save the trained model
model.save("ppo_pong")

# Load and evaluate the model
loaded_model = PPO.load("ppo_pong")
obs = env.reset()

done = [False] * env.num_envs
while not all(done):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()

wandb.finish()
