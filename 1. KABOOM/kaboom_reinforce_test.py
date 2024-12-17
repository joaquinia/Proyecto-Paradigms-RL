import gymnasium as gym
import ale_py
from ale_py import ALEInterface
import torch
import numpy as np
import cv2
import os
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore') 
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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.feature_size = self._calculate_feature_size(input_shape)

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
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError('The given optimizer is not supported. Please provide an optimizer that is an instance of torch.optim.Optimizer')
        self.optimizer = optimizer

    def should_explore(self):
        explore = Categorical(torch.tensor([1 - self.explore_factor, self.explore_factor])).sample()
        return explore == 1
    
    def random_action(self):
        uniform_sampler = Categorical(torch.tensor([1 / self.policy_head.out_features] * self.policy_head.out_features))
        return uniform_sampler.sample()
    
    @staticmethod
    def preprocess_frame(frame, new_shape=(84, 84)):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        cropped_frame = gray[34:194, :]
        resized_frame = cv2.resize(cropped_frame, new_shape, interpolation=cv2.INTER_AREA)
        normalized_frame = resized_frame / 255.0  
        return torch.tensor(normalized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def select_action(self, state):
        state = self.preprocess_frame(state)
        state = state.to(self.device)  

        probs, V_s = self.forward(state)
        m = Categorical(probs)

        if self.should_explore():
            action = self.random_action()
        else:
            action = m.sample()

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
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)  

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

def test_agent(agent, env, num_episodes, device, wandb_enabled=True):
    total_rewards = []
    episode_lengths = []
    reward_threshold = 500

    print(f"Environment reward threshold: {reward_threshold}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            state = next_state

        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        if wandb_enabled:
            wandb.log({
                "episode": episode + 1,
                "reward": total_reward,
                "episode_length": episode_length,
            })

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Episode Length: {episode_length}")

    # Aggregate metrics
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    episodes_above_threshold = sum(r >= reward_threshold for r in total_rewards)

    if wandb_enabled:
        wandb.log({
            "average_reward": avg_reward,
            "average_episode_length": avg_length,
            "episodes_above_threshold": episodes_above_threshold,
            "reward_threshold": reward_threshold
        })

    print(f"Average Reward: {avg_reward}, Average Episode Length: {avg_length}, "
          f"Episodes Above Threshold: {episodes_above_threshold}")

    # Plot rewards with threshold (inveted, just used to compare both models)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', label="Episode Reward")
    plt.axhline(y=reward_threshold, color='r', linestyle='--', label="Reward Threshold")
    plt.title("Episode Rewards vs Reward Threshold")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig("reward_threshold_plot_PG.png")
    plt.show()

    print("Plot saved as 'reward_threshold_plot.png'")




if __name__ == "__main__":
    wandb.init(
        project="Paradigms",
        entity="mikelottogc-universitat-aut-noma-de-barcelona", 
        config={
            "env_name": "ALE/Kaboom-v5",
            "num_test_episodes": 20,
        },
    )

    env_name = "ALE/Kaboom-v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = "final_model3.pth"

    # Create environment with video recording enabled
    env = gym.make(env_name, render_mode = "rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="videos_35000", episode_trigger=lambda e: True)

    number_of_actions = env.action_space.n

    agent = PGAgent(
        number_of_actions=number_of_actions,
        input_shape=(1, 84, 84),
        HL_size=128,
        gamma=0.99,
        exploration=0,
        device=device,
        clip_grads=False,
    ).to(device)

    # Load trained model
    agent.load_state_dict(torch.load(checkpoint_file, map_location=device))

    # Testing the agent and logging metrics
    num_test_episodes = 20
    test_agent(agent, env, num_test_episodes, device)

    wandb.finish()
