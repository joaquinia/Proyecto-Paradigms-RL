"""
Necessary Instalations

pip install swig
pip install box2d-py
pip install gymnasium
pip install "gymnasium[atari,accept-rom-license]"
pip install "stable-baselines3[extra]"
pip install "pettingzoo[all]"
pip install supersuit
pip install "autorom[accept-rom-license]"

"""
#Environment Initialization: 
from pettingzoo.atari import pong_v3
from stable_baselines import PPOç
env = pong_v3.env(render_mode="human") #Notice that this is a render mode for visualization

#Wrap the environment with Stable Baselines3 Compatibility
from supersuit import aec_to_parallel
parallel_env = aec_to_parallel(env) #This converts AEC to parallel environmemnt

#Set up the RL MOdel:
model = PPO("CnnPolicy", parallel_env, verbose=1)

#TUne Hyperparameters:
"""
Learning Rate: [1e-3, 1e-4, 1e-5]
Discount factor (gamma): [0.95, 0.99, 0.999]
Batch Size: [64, 128, 256]
"""
model = PPO("CnnPolicy", parallel_env, learning_rate = 1e-4, gamma=0.99, batch_size=128, verbose=1)

#Train the model
model.learn(total_timesteps=500000)

#Evaluate performance of the model after training
obs = parallel_env.reset()
for _ in range(100):
    action = model.predict(obs, deterministic=True)[0]
    obs, rewards, dones, info = parallel_env.step(action)

#Save and export the model
model.save("pong_agent")

#Load the model
loaded_model = PPO.load("pong_agent")

#Testing:

win_count = 0
for _ in range(100):
    obs = parallel_env.reset()
    done = False
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, rewards, dones, info = parallel_env.step(action)
        if dones[0]:
            win_count += rewards[0] > 0  # Example reward check
print(f"Win rate: {win_count}%")


from gymnasium.wrappers import RecordVideo

env = RecordVideo(parallel_env, video_folder="./videos")
obs = env.reset()
for _ in range(1000):  # Adjust for the desired number of steps
    action = model.predict(obs, deterministic=True)[0]
    obs, _, done, _ = env.step(action)
    if done:
        break
env.close()
