import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import ale_py
import numpy as np
import cv2

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the environment with limited actions
class LimitedActionsEnv(gym.ActionWrapper):
    def __init__(self, env, actions):
        super(LimitedActionsEnv, self).__init__(env)
        self.actions = actions
        self.action_space = gym.spaces.Discrete(len(actions))  # Update the action space

    def action(self, act):
        return self.actions[act]

    def reverse_action(self, act):
        return self.actions[act]

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")
env = LimitedActionsEnv(env, [0, 2, 5])  # Only use actions 0, 2, and 5

obs_space = env.observation_space.shape
input_channels = 1  # Set to 1 channel for grayscale input
num_actions = 3  # Use only 3 actions

# Print out spaces for verification
print(f"Observation Space Shape: {obs_space}")
print(f"Number of Actions: {num_actions}")

# Hyperparameters
learning_rate = 0.00025
gamma = 0.99
epsilon = 1
epsilon_decay = 0.0003
epsilon_min = 0.01
batch_size = 64
memory_size = 50000
episodes = 500

device = torch.device("cuda")

# Replay memory
memory = deque(maxlen=memory_size)

# Function to preprocess frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

# Function to select an action
def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1, 2])  # Choose randomly from the limited action space
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()

# Function to optimize the model
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    state_batch = np.array(batch[0])
    next_state_batch = np.array(batch[3])

    state_batch = torch.tensor(state_batch, dtype=torch.float32).unsqueeze(1).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).unsqueeze(1).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize the networks
policy_net = DQN(input_channels, num_actions).to(device)
target_net = DQN(input_channels, num_actions).to(device)

# Load the saved model if it exists
try:
    policy_net.load_state_dict(torch.load('dqn_policy.pth', map_location=device))
    target_net.load_state_dict(torch.load('dqn_target.pth', map_location=device))
    print("Loaded saved models.")
except FileNotFoundError:
    print("No saved models found, starting training from scratch.")

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
rewards = []

# Training the agent
for episode in range(episodes):
    print(f"Starting Episode {episode + 1}/{episodes}")
    state, info = env.reset(seed=42)
    state = preprocess_frame(state)
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, policy_net, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_frame(next_state)
        if reward == 1:  # Positive reward for scoring
            reward = 1
        elif reward == -1: # Negative reward for missing
            reward = -1
        memory.append((state, action, reward, next_state, done))
        state = next_state
        optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(epsilon_min,(epsilon - epsilon_decay))

    print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}, Epsilon: {epsilon}")

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save the model after every episode
    torch.save(policy_net.state_dict(), 'dqn_policy.pth')
    torch.save(target_net.state_dict(), 'dqn_target.pth')

env.close()

# Plotting the results
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()
