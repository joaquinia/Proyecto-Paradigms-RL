#IMPLEMENTATION OF A RAINBOW DQN 

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as TF
import numpy as np
import random
import cv2
import wandb
import os
from collections import deque

# ===========================
# Preprocessing
# ===========================
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = resized / 255.0
    return normalized

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
# Rainbow DQN Model
# ===========================
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_atoms, v_min, v_max):
        super(RainbowDQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_size = 64 * 7 * 7

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_atoms)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature_extractor(x)

        value = self.value_stream(features).view(batch_size, 1, self.n_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.n_actions, self.n_atoms)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = torch.softmax(q_atoms, dim=2)  # Convert to probability distribution

        return q_dist

    def get_q_values(self, x):
        q_dist = self.forward(x)
        support = self.support.to(q_dist.device)  # Move support to the same device as q_dist
        q_values = torch.sum(q_dist * support, dim=2)  # Expectation of distribution
        
        return q_values

# ===========================
# Prioritized Replay Buffer
# ===========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha, beta_start, beta_frames, device='cuda'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.device = device

        self.buffer = []
        self.priorities = deque([0] * capacity, maxlen=capacity)
        self.pos = 0

    def append(self, state, action, reward, done, next_state):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, done, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, done, next_state)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample_batch(self, batch_size):
        priorities = np.array(self.priorities)[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*samples)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions).to(self.device),
            torch.tensor(rewards).to(self.device),
            torch.tensor(dones, dtype=torch.bool).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(weights, dtype=torch.float32).to(self.device),
            indices,
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

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
# Main Training Loop (Rainbow)
# ===========================
def train_rainbow_dqn(
    env_name="ALE/Kaboom-v5",
    episodes=10000,
    batch_size=128,
    gamma=0.99,
    lr=1e-4,
    buffer_capacity=50000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=100000,
    target_update_freq=5000,
    n_atoms=51,
    v_min=-10,
    v_max=10,
    alpha=0.6,
    beta_start=0.4,
    beta_frames=100000
):
    
    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    save_model_path = os.path.join(save_dir, "rainbow_dqn_model.pth")
    
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
            "n_atoms": n_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "alpha": alpha,
            "beta_start": beta_start,
            "beta_frames": beta_frames
        }
    )
    
    env = gym.make(env_name)
    n_actions = env.action_space.n
    input_shape = (4, 84, 84)
    
    frame_stack = FrameStack(4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = RainbowDQN(input_shape, n_actions, n_atoms, v_min, v_max).to(device)
    target_net = RainbowDQN(input_shape, n_actions, n_atoms, v_min, v_max).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_capacity,
        state_shape=input_shape,
        alpha=alpha,
        beta_start=beta_start,
        beta_frames=beta_frames,
        device=device,
    )

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    epsilon = epsilon_start
    epsilon_decay_step = (epsilon_start - epsilon_end) / epsilon_decay

    total_rewards = []  # Track total rewards for averaging
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
                q_values = policy_net.get_q_values(state_tensor)
            action = epsilon_greedy_policy(q_values, epsilon, n_actions)

            next_state, reward, done, _ = frame_stack.step(env, action)
            replay_buffer.append(state, action, reward, done, next_state)
            state = next_state
            episode_reward += reward

            epsilon = max(epsilon_end, epsilon_start - total_steps * epsilon_decay_step)

            if len(replay_buffer) > batch_size:
                states, actions, rewards, dones, next_states, weights, indices = replay_buffer.sample_batch(batch_size)
                
                # Multi-step return calculation
                n_step_returns = rewards + gamma * (1 - dones.float())

                with torch.cuda.amp.autocast():
                    q_dist = policy_net(states)  # Shape: [batch_size, n_actions, n_atoms]
                    action_q_values = q_dist.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_atoms))  # [batch_size, 1, n_atoms]
                    action_q_values = action_q_values.squeeze(1)  # [batch_size, n_atoms]
                    
                    with torch.no_grad():
                        next_q_dist = target_net(next_states)  # [batch_size, n_actions, n_atoms]
                        next_q_values = torch.sum(next_q_dist * policy_net.support.to(next_q_dist.device), dim=2)  # [batch_size, n_actions]
                        max_next_q_values, _ = torch.max(next_q_values, dim=1)  # [batch_size]
                        targets = n_step_returns + gamma * max_next_q_values * (1 - dones.float())  # [batch_size]
                    
                    # Corrected target distribution calculation
                    targets_dist = torch.zeros_like(action_q_values)
                    for i in range(batch_size):
                        tz = (targets[i].item() - v_min) / policy_net.delta_z
                        l = int(torch.floor(torch.tensor(tz)).clamp(0, n_atoms - 1))
                        u = int(torch.ceil(torch.tensor(tz)).clamp(0, n_atoms - 1))
                        
                        # Use probabilities proportional to the distance from the target
                        dist_l = abs(u - tz)
                        dist_u = abs(tz - l)
                        
                        targets_dist[i, l] = dist_u
                        targets_dist[i, u] = dist_l
                        
                        # Normalize to create a valid probability distribution
                        targets_dist[i] /= targets_dist[i].sum()
                    
                    # KL Divergence Loss
                    loss = nn.KLDivLoss(reduction='batchmean')(
                        torch.log(action_q_values + 1e-10), 
                        targets_dist
                    )

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                episode_loss += loss.item()
                loss_count += 1

                # Update priorities
                priorities = action_q_values.detach().max(dim=1)[0].cpu().numpy()
                replay_buffer.update_priorities(indices, priorities)

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-100:])  # Average over last 100 episodes

        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "average_reward": avg_reward,
            "epsilon": epsilon,
            "average_loss": avg_loss,
            "steps": total_steps,
        })

        print(f"Episode {episode + 1}, Reward: {episode_reward}, Average Reward: {avg_reward}")

        # Early stopping or saving based on performance
        if len(total_rewards) >= 100 and avg_reward > 500:  # Adjust threshold as needed
            torch.save(policy_net.state_dict(), save_model_path)
            print(f"Model saved to {save_model_path}")
            break

    # Final model save
    torch.save(policy_net.state_dict(), save_model_path)
    print(f"Final model saved to {save_model_path}")

    # Log final model as artifact
    artifact = wandb.Artifact("rainbow_dqn_model", type="model")
    artifact.add_file(save_model_path)
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()

# Run the training
if __name__ == "__main__":
    train_rainbow_dqn()