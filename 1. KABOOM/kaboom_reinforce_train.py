import gymnasium as gym
import ale_py
from ale_py import ALEInterface
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb
import os
import cv2

import warnings
warnings.filterwarnings('ignore')
ale = ALEInterface()
ale = ALEInterface()
gym.register_envs(ale_py)

# PGAgent definition

class PGAgent(nn.Module):
    def __init__(self, number_of_actions, input_shape, HL_size, gamma, exploration, device, clip_grads):
        super(PGAgent, self).__init__()
        self.device = device
        self.clip_grads = clip_grads
        self.gamma = gamma
        self.explore_factor = exploration

        # Convolutional layers for processing frames
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened feature size
        self.feature_size = self._calculate_feature_size(input_shape)

        # Fully connected layers for policy and value functions
        self.fc1 = nn.Linear(self.feature_size, HL_size)
        self.policy_head = nn.Linear(HL_size, number_of_actions)
        self.value_head = nn.Linear(HL_size, 1)

        self.optimizer = None
        self.saved_log_probs = []
        self.rewards = []
        self.save_value_function = []

    def _calculate_feature_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value

    def set_optimizer(self, optimizer):
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def should_explore(self):
        explore = Categorical(torch.tensor([1 - self.explore_factor, self.explore_factor])).sample()
        return explore == 1
    
    @staticmethod
    def preprocess_frame(frame, new_shape=(84, 84)):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        cropped_frame = gray[34:194, :]  # Crop irrelevant parts
        resized_frame = cv2.resize(cropped_frame, new_shape, interpolation=cv2.INTER_AREA)  # Resize to 84x84
        normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
        return torch.tensor(normalized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def select_action(self, state):
        # Preprocess the frame
        state = self.preprocess_frame(state)  # Preprocess the input frame
        state = state.to(self.device)

        # Forward pass
        probs, V_s = self.forward(state)  # Get action probabilities and value
        m = Categorical(probs)

        # Sample action
        if self.should_explore():
            action = torch.randint(0, self.policy_head.out_features, (1,), device=self.device)  # Random action for exploration
        else:
            action = m.sample()

        # Save log probabilities and value
        self.saved_log_probs.append(m.log_prob(action))
        self.save_value_function.append(V_s)
        return action.item()
    
    def update(self, episode):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # Normalize rewards

        for log_prob, reward, value in zip(self.saved_log_probs, returns, self.save_value_function):
            advantage = reward - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value.squeeze(), reward))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.save_value_function[:]
        return loss.item()
    
    def save_checkpoint(self, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True) 
        filepath = os.path.join(save_dir, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        self.load_state_dict(torch.load(filepath))
        print(f"Checkpoint loaded from {filepath}")

if __name__ == "__main__":
    # Hyperparameters
    env_name = "ALE/Kaboom-v5"
    input_shape = (1, 84, 84)
    HL_size = 128
    gamma = 0.99 # For the other model: 0.98
    exploration = 0.15 # 0.2 in the first run /// For the other model: 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 40000
    checkpoint_dir = "checkpoints_kaboom_PG_"
    checkpoint_file = os.path.join(checkpoint_dir, "agent3_.pth")
    lr = 0.00015 # 0.0002 in the first run (20.000 episodes) /// For the other model: 0.00025
    load = True
    moving_avg_window = 10
    
    # Initialize wandb
    wandb.init(
        project="ProjectParadigms",
        entity="1665890",
        config={
            "env_name": env_name,
            "gamma": gamma,
            "learning_rate": lr,
            "hidden_layer_size": HL_size,
            "num_episodes": num_episodes,
        },
    )

    # Initialize environment
    env = gym.make(env_name)
    number_of_actions = env.action_space.n
    action_dict = {i: action for i, action in enumerate(env.unwrapped.get_action_meanings())}

    # Initialize the Agent
    agent = PGAgent(
        number_of_actions=number_of_actions,
        input_shape=input_shape,
        HL_size=HL_size,
        gamma=gamma,
        exploration=exploration,
        device=device,
        clip_grads=False,
    ).to(device)
    agent.optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Load checkpoint if specified
    start_episode = 0
    if load and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        agent.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_episode = checkpoint["episode"] + 1
        print(f"Resuming training from episode {start_episode}")
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)

    episode_rewards_history = []
    episode_losses_history = []

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        episode_rewards = []
        agent.rewards = []
        agent.saved_log_probs = []
        agent.save_value_function = []
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Perform action in environment
            next_state, reward, done, _, _ = env.step(action)
            agent.rewards.append(reward)
            episode_rewards.append(reward)

            state = next_state

        # Update agent
        loss = agent.update(episode)
        episode_rewards_history.append(sum(episode_rewards))
        episode_losses_history.append(loss)

        avg_reward = np.mean(episode_rewards_history[-moving_avg_window:])
        avg_loss = np.mean(episode_losses_history[-moving_avg_window:])

        # Print episode metrics
        print(
            f"Episode {episode + 1}, Reward: {sum(episode_rewards)}, Loss: {loss}, "
            f"Average Reward (last {moving_avg_window}): {avg_reward}, Average Loss (last {moving_avg_window}): {avg_loss}"
        )

        # Log metrics to wandb
        wandb.log({
            "episode": episode + 1,
            "reward": sum(episode_rewards),
            "loss": loss,
            "average_reward": avg_reward,
            "average_loss": avg_loss,
        })

        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            torch.save({
                "episode": episode,
                "model_state": agent.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "rewards_history": episode_rewards_history,
                "losses_history": episode_losses_history,
            }, checkpoint_file)
            print(f"Checkpoint saved at episode {episode + 1}")

    # Save the final model
    final_model_file = os.path.join(checkpoint_dir, "final_model4.pth")
    torch.save(agent.state_dict(), final_model_file)
    print(f"Final model saved at {final_model_file}")

    wandb.finish()