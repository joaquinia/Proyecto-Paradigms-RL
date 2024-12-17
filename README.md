# FINAL PROJECT PARADIGMS OF MACHINE LEARNING - GROUP 9 

This project applies reinforcement learning techniques to solve challenges in the Arcade Learning Environment (ALE), which emulates Atari 2600 games. The primary goal is to train agents to perform well in selected ALE environments using various RL algorithms.

## Project Structure
The project is divided into three main parts:

1. **Simpler Environment**: Evaluates the performance of Deep Q-Networks (DQN) and REINFORCE algorithms.
2. **Complex Environment**: Tackles a more challenging setting using advanced algorithms like Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO).
3. **Pong Environment**: Focuses exclusively on training agents for Pong.

## Repository Contents

### 1. **KABOOM**
Contains both the training and testing Python files for:
- **DQN**: Requires the trained model `kaboom_dqn_model.pth` found in the `FINAL_MODELS` folder.
- **REINFORCE**: Requires either `kaboom_PG_35k.pth` or `kaboom_PG_40K.pth` from the `FINAL_MODELS` folder. Both models were trained using the same code, but one trained for 35k episodes and the other for 40k episodes. The significance of these models is detailed in the project report.

### 2. **GALAXIAN**
Contains both the training and testing Python files for:
- **A2C**: Advanced RL algorithm for the complex environment.
- **PPO**: Another advanced RL algorithm.

Trained models for these algorithms are provided as external files (zip files) due to their large size. These are included in the project submission but not directly uploaded to the repository.To test the models the path of the corresponding zip file must be put in the loading. 

### 3. **PONG**
Includes both the training and testing Python files for the Pong environment.The trained model for this algorithm is also in an external file in the submission of the project in the Campus Virtual. 

### 4. **FINAL_MODELS**
This folder contains the trained models for different environments and algorithms:
- `kaboom_dqn_model.pth`
- `kaboom_PG_35k.pth`
- `kaboom_PG_40K.pth`

### 5. **VIDEOS**
Includes videos showcasing the performance of trained agents across different games.

### 6. **Requirements.txt**
Specifies the dependencies required to execute the code, along with their respective versions.

### 7. **run_sh.sh**
A script for executing Python files on a cluster using GPU resources.

## Execution Notes
- Ensure all dependencies from `Requirements.txt` are installed.
- Trained models need to be placed in the appropriate folder (`FINAL_MODELS`) for testing to work correctly.
- For Galaxian’s A2C and PPO models, unzip the provided external files and place them in the corresponding directory.

---

For additional details, refer to the project report included in the submission.
