# Dueling DQN and Standard DQN for Pong

This project trains both a Dueling DQN (Deep Q-Network) and a standard DQN to play the game Pong using the OpenAI Gym environment. It includes code for preprocessing frames, managing stacked frames, and optimizing the model using experience replay.

## Requirements

- Python 3.x
- Gymnasium
- PyTorch
- OpenCV
- Numpy
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository/pong-dueling-dqn.git
    cd pong-dueling-dqn
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training and Playing with Standard DQN

To train the standard DQN model and play the game, run the script:
```bash
python pong.py
```
Training and Playing with Dueling DQN
To train the Dueling DQN model and play the game, run the script:
```bash
python pong2.py
```
## Explanation of Files
- pong.py: Script for training the standard DQN model and playing the game.

- pong2.py: Script for training the Dueling DQN model and playing the game.
## Code Overview
Model Definition
Standard DQN (pong.py)
```python
import torch
import torch.nn as nn

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
```
Dueling DQN (pong2.py)
```python
import torch
import torch.nn as nn

class SimpleDuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(SimpleDuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 9 * 9, 256)
        self.fc_adv = nn.Linear(256, num_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), adv.size(1))
        return val + adv - adv.mean(1, keepdim=True).expand(x.size(0), adv.size(1))
```
## Results
The results of the training are plotted and saved, showing the total rewards per episode. Here's an example of how to include graphs in your results:
```python
# Plotting the results
import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()
```
![Figure_3](https://github.com/user-attachments/assets/9d50cd7e-5c52-461c-b45f-1ffcc79edbb8)
Graph shows traing rewards using duel DQN for 500 plays using pong2.py.

