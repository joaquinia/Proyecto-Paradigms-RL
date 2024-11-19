import gymnasium as gym
import numpy as np
import random
import collections
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ===========================
# Preprocessing
# ===========================
def preprocess_frame(frame, new_shape=(84, 84)):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped_frame = gray[34:194, :]  # Adjust based on the game
    resized_frame = cv2.resize(cropped_frame, new_shape, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0 * 2 - 1
    return normalized_frame

class FrameStack:
    def _init_(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, env):
        obs, _ = env.reset()
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
          target_update_freq=1000):
    
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
    replay_buffer = ReplayBuffer(buffer_capacity)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Epsilon decay schedule
    epsilon = epsilon_start
    epsilon_decay_step = (epsilon_start - epsilon_end) / epsilon_decay

    # Training loop
    total_steps = 0
    for episode in range(episodes):
        state = frame_stack.reset(env)
        episode_reward = 0

        while True:
            total_steps += 1
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = epsilon_greedy_policy(q_values, epsilon, n_actions)

            next_state, reward, done, _ = frame_stack.step(env, action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            epsilon = max(epsilon_end, epsilon - epsilon_decay_step)

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions).unsqueeze(-1).to(device)
                rewards = torch.tensor(rewards).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.uint8).to(device)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)

                q_values = policy_net(states).gather(1, actions).squeeze(-1)
                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        #print(f"Episode {episode + 1}, Reward: {episode_reward}")
        with open("rewards.txt", "a") as f:
            f.write(f"Episode {episode + 1}, Reward: {episode_reward}\n")
            


    env.close()