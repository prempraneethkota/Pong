import gymnasium as gym
import torch
import torch.nn as nn
import ale_py
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
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

# Function to preprocess frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

# Function to select an action
def select_action(state, policy_net):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state)
        return q_values.max(1)[1].item()

# Initialize the environment
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")
env = LimitedActionsEnv(env, [0, 2, 5])  # Only use actions 0, 2, and 5

obs_space = env.observation_space.shape
stack_size = 4
input_channels = stack_size
num_actions = 3  # Use only 3 actions

print(f"Observation Space Shape: {obs_space}")
print(f"Number of Actions: {num_actions}")

device = torch.device("cuda")

# Initialize the networks
policy_net = DuelingDQN(input_channels, num_actions).to(device)

# Load the trained model
policy_net.load_state_dict(torch.load('dqn_policy2.pth', map_location=device))
print("Loaded saved model.")

# Initialize the stacked frames handler
stacked_frames = StackedFrames(stack_size)

# Play the game using the trained model
def play_game():
    frame, info = env.reset(seed=42)
    state = stacked_frames.reset(frame)  # Initialize stacked frames
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, policy_net)
        next_frame, reward, done, truncated, info = env.step(action)
        next_state = stacked_frames.add_frame(next_frame)  # Update stacked frames
        state = next_state
        total_reward += reward

    return total_reward

# Run multiple episodes to evaluate the trained model
num_episodes = 10
total_rewards = []

for episode in range(num_episodes):
    reward = play_game()
    total_rewards.append(reward)
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

env.close()

# Plotting the results
plt.plot(total_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()
