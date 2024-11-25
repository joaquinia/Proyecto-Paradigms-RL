import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from ale_py import ALEInterface
ale = ALEInterface()
from ale_py import ALEInterface
ale = ALEInterface()
import ale_py
gym.register_envs(ale_py)

import gymnasium as gym
import numpy as np
import random
import collections
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as TF
from collections import deque
import wandb

#THIS VERSION OF THE DQN IS A BASELINE IMPLEMENTATION BUT WITH CHANGES REGARDING EFFICIENCY COMPARED TO THE FIRST VERSION

# ===========================
# Preprocessing
# ===========================
def preprocess_frame(frame, new_shape=(84, 84)):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped_frame = gray[34:194, :]
    tensor_frame = torch.from_numpy(cropped_frame).unsqueeze(0).float() / 255.0
    resized_frame = TF.resize(tensor_frame, new_shape)
    return (resized_frame * 2 - 1).squeeze(0).numpy()

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, env):
        obs, info = env.reset()
        frame = preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def step(self, env, action):
        next_obs, reward, terminated, truncated, info = env.step(action)
        frame = preprocess_frame(next_obs)
        self.frames.append(frame)
        done = terminated or truncated
        return np.stack(self.frames, axis=0), reward, done, info

# ===========================
# DQN Definition
# ===========================
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ===========================
# Replay Buffer
# ===========================
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        self.buffer.append((state_tensor, action, reward, next_state_tensor, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state),
            torch.tensor(action, dtype=torch.int64).to(self.device),
            torch.tensor(reward, dtype=torch.float32).to(self.device),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.uint8).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)

# ===========================
# Epsilon-Greedy Policy
# ===========================
def epsilon_greedy_policy(q_values, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return torch.argmax(q_values).item()

# ===========================
# Main Training Loop
# ===========================
def train(env_name="ALE/Kaboom-v5",
          episodes=500,
          batch_size=32,
          gamma=0.99,
          lr=1e-4,
          buffer_capacity=100000,
          epsilon_start=1.0,
          epsilon_end=0.1,
          epsilon_decay=100000,
          target_update_freq=5000):
    
    # Initialize wandb
    wandb.init(
        project="ProjectParadigms",  
        entity="1665890",
        config={
            "env_name": env_name,
            "episodes": episodes,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": lr,
            "buffer_capacity": buffer_capacity,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "target_update_freq": target_update_freq,
        }
    )
    
    env = gym.make(env_name)
    n_actions = env.action_space.n
    input_shape = (4, 84, 84)

    # Frame stacker
    frame_stack = FrameStack(k=4)

    # DQNs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Replay buffer and optimizer
    replay_buffer = ReplayBuffer(buffer_capacity, device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Epsilon decay schedule
    epsilon = epsilon_start
    epsilon_decay_step = (epsilon_start - epsilon_end) / epsilon_decay

    # Training loop
    total_steps = 0
    for episode in range(episodes):
        state = frame_stack.reset(env)
        episode_reward = 0
        episode_loss = 0
        loss_count = 0

        while True:
            total_steps += 1
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                q_values = policy_net(state_tensor)
            action = epsilon_greedy_policy(q_values, epsilon, n_actions)

            next_state, reward, done, _ = frame_stack.step(env, action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            epsilon = max(epsilon_end, epsilon_start - total_steps * epsilon_decay_step)

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                with torch.cuda.amp.autocast():
                    q_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0]
                        targets = rewards + gamma * next_q_values * (1 - dones)
                    loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                episode_loss += loss.item()
                loss_count += 1

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0

        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "epsilon": epsilon, 
            "average_loss": avg_loss,
            "steps": total_steps,
        })

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()
    wandb.finish()

train()