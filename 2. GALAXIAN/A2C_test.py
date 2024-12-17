import matplotlib.pyplot as plt

# Define reward threshold
reward_threshold = 1000

# Testing Phase: Evaluate the trained model
eval_env = DummyVecEnv([lambda: gym.make(config["env_name"])]);
eval_env = VecFrameStack(eval_env, n_stack=4)

# Record rewards for each evaluation episode
n_eval_episodes = 10
rewards = []

for _ in range(n_eval_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        episode_reward += reward[0]  # reward is a numpy array, extract scalar
    rewards.append(episode_reward)

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

# Plotting the comparison of episode rewards with the reward threshold
def plot_episode_rewards(rewards, reward_threshold):
    plt.figure(figsize=(10, 6))

    # Line plot for episode rewards
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', label="Episode Reward")

    # Threshold line
    plt.axhline(y=reward_threshold, color='red', linestyle='--', label='Reward Threshold')

    # Adding labels and title
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards vs Reward Threshold")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Call the function to plot the graph
plot_episode_rewards(rewards, reward_threshold)

# Close the evaluation environment
eval_env.close()