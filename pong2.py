import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import ale_py
import cv2

# Define the Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_adv = nn.Linear(64 * 7 * 7, 512)
        self.fc1_val = nn.Linear(64 * 7 * 7, 512)
        self.fc2_adv = nn.Linear(512, num_actions)
        self.fc2_val = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = torch.relu(self.fc1_adv(x))
        val = torch.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), adv.size(1))
        x = val + adv - adv.mean(1, keepdim=True).expand(x.size(0), adv.size(1))
        return x

# Class for managing stacked frames
class StackedFrames:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        processed_frame = preprocess_frame(frame)
        self.frames = deque([processed_frame] * self.stack_size, maxlen=self.stack_size)
        return np.stack(self.frames, axis=0)

    def add_frame(self, frame):
        processed_frame = preprocess_frame(frame)
        self.frames.append(processed_frame)
        return np.stack(self.frames, axis=0)

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
stack_size = 4
input_channels = stack_size
num_actions = 3  # Use only 3 actions

print(f"Observation Space Shape: {obs_space}")
print(f"Number of Actions: {num_actions}")

# Hyperparameters
learning_rate = 0.0001
gamma = 0.99
epsilon = 1
epsilon_decay = 0.99
epsilon_min = 0.001
batch_size = 64
memory_size = 5000
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
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
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

    state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize the networks
policy_net = DuelingDQN(input_channels, num_actions).to(device)
target_net = DuelingDQN(input_channels, num_actions).to(device)

# Load the saved model if it exists
try:
    policy_net.load_state_dict(torch.load('dqn_policy2.pth', map_location=device))
    target_net.load_state_dict(torch.load('dqn_target2.pth', map_location=device))
    print("Loaded saved models.")
except FileNotFoundError:
    print("No saved models found, starting training from scratch.")

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
rewards = []

# Initialize the stacked frames handler
stacked_frames = StackedFrames(stack_size)

# Training the agent
for episode in range(episodes):
    print(f"Starting Episode {episode + 1}/{episodes}")
    frame, info = env.reset(seed=42)
    state = stacked_frames.reset(frame)  # Initialize stacked frames
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, policy_net, epsilon)
        next_frame, reward, done, truncated, info = env.step(action)
        next_state = stacked_frames.add_frame(next_frame)  # Update stacked frames
        if reward == 1:  # Positive reward for scoring
            reward = 1
        elif reward == -1:  # Negative reward for missing
            reward = -1
        memory.append((state, action, reward, next_state, done))
        state = next_state
        optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(epsilon_min, (epsilon * epsilon_decay))

    print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}, Epsilon: {epsilon}")

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save the model after every episode
    torch.save(policy_net.state_dict(), 'dqn_policy3.pth')
    torch.save(target_net.state_dict(), 'dqn_target3.pth')

env.close()

# Plotting the results
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()
