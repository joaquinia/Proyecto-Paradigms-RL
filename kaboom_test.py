import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')
from ale_py import ALEInterface
ale = ALEInterface()
import ale_py
gym.register_envs(ale_py)

ENV_NAME = "ALE/Kaboom-v5"

test_env = gym.make(ENV_NAME)

import os
import gymnasium as gym
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as TF
from collections import deque
from collections import namedtuple
import wandb

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
# Optimized Replay Buffer
# ===========================
class experienceReplayBuffer:
    def __init__(self, capacity, state_shape, device='cuda', store_on_gpu=True):
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.store_on_gpu = store_on_gpu
        
        # Use float16 for reduced memory usage
        dtype = torch.float16 if store_on_gpu else torch.float32
        
        # Allocate memory
        self.states = torch.zeros((capacity, *state_shape), dtype=dtype, device=device if store_on_gpu else 'cpu')
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device if store_on_gpu else 'cpu')
        self.rewards = torch.zeros(capacity, dtype=dtype, device=device if store_on_gpu else 'cpu')
        self.next_states = torch.zeros((capacity, *state_shape), dtype=dtype, device=device if store_on_gpu else 'cpu')
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device if store_on_gpu else 'cpu')
        self.idx = 0
        self.size = 0

    def append(self, state, action, reward, done, next_state):
        idx = self.idx % self.capacity
        self.states[idx] = torch.tensor(state, dtype=self.states.dtype, device=self.states.device)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_states[idx] = torch.tensor(next_state, dtype=self.next_states.dtype, device=self.next_states.device)
        
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        
        # Return data directly if on GPU; otherwise, move to GPU during sampling
        device = 'cuda' if self.store_on_gpu else self.device
        return (
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.rewards[indices].to(device),
            self.dones[indices].to(device),
            self.next_states[indices].to(device),
        )
    
    def __len__(self):
        return self.size

# ===========================
# Epsilon-Greedy Policy
# ===========================
def epsilon_greedy_policy(q_values, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return torch.argmax(q_values).item()
    

# params
model = path_to_model

env = make_env(ENV_NAME, render_mode="rgb_array")
net = DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(model, map_location=torch.device(device)))

"TEST"
import time
import collections
from PIL import Image

# params
visualize = True
images = []

state, _ = env.reset()
total_reward = 0.0

while True:
    start_ts = time.time()
    if visualize:
        img = env.render()
        images.append(Image.fromarray(img))

    state_ = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_).data.numpy()[0]
    action = np.argmax(q_vals)

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if done:
        break

print("Total reward: %.2f" % total_reward)

"EXPORT THE GIF"
# params
gif_file = "video.gif"

# duration is the number of milliseconds between frames; this is 40 frames per second
images[0].save(gif_file, save_all=True, append_images=images[1:], duration=60, loop=0)

print("Episode export to '{}'".format(gif_file))
