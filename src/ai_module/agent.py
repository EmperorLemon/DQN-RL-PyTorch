from tensorboardX import SummaryWriter

from .model import DQN
from .memory import ReplayMemory, Transition

from typing import Any

from torch import nn
from torch import optim

import numpy as np
import random
import torch


class DQNAgent:
    def __init__(
        self,
        device,
        writer: SummaryWriter | None,
        hp: dict[str, Any],  # Hyperparameters
    ):
        self.device = device
        self.writer = writer

        self.batch_size = hp.get("BATCH_SIZE")

        # Learning rate of agent
        self.lr = hp.get("LR")
        self.epsilon = hp.get("EPS_START")
        self.epsilon_end = hp.get("EPS_END")
        self.epsilon_decay = hp.get("EPS_DECAY")
        self.gamma = hp.get("GAMMA")

        self.memory = ReplayMemory(hp.get("BUFFER_SIZE"))

        self.actions = np.array([0, 1, 2, 3])

    def create_agent_models(self, state_size: int, action_size: int):
        # Initialize policy and target networks
        self.policy_net = DQN(state_size, action_size, [64, 64, 64]).to(self.device)
        self.target_net = DQN(state_size, action_size, [64, 64, 64]).to(self.device)

        # Copy weights from policy network to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def learn(self):
        # Check if there is enough samples in memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert numpy arrays to PyTorch tensors and move to device
        state_batch = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).long().to(self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().to(self.device)
        next_state_batch = (
            torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        )
        done_batch = torch.from_numpy(np.array(batch.done)).float().to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        # Compute the expected Q values
        # If done, next_state_value is 0
        expected_state_action_values = (
            next_state_values * self.gamma * (1 - done_batch)
        ) + reward_batch

        # Compute loss
        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def select_action(self, state: np.ndarray):
        state = torch.from_numpy(state).to(self.device)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploit: choose the best action according to the policy
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, *args):
        self.memory.push(*args)
