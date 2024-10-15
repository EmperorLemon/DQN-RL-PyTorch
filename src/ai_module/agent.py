from .model import DQN
from .memory import ReplayMemory, Transition

from utils.globals import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

from typing import Any

from torch import nn
from torch import optim

import numpy as np
import random
import torch


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: dict[str, Any],  # Hyperparameters
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.state_size = state_size
        self.action_size = action_size
        
        self.device = device
        
        self.memory = ReplayMemory(config.get("BUFFER_SIZE", 100000))
        self.batch_size = config.get("BATCH_SIZE", 128)
        
        self.gamma = config.get("GAMMA", 0.99)
        self.lr = config.get("LR", 1e-3)
        
        self.policy_net = DQN(state_size, action_size, config.get("HIDDEN_LAYERS", [128, 128])).to(device)
        self.target_net = DQN(state_size, action_size, config.get("HIDDEN_LAYERS", [128, 128])).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=config.get("WEIGHT_DECAY", 1e-8), amsgrad=True)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.lr, momentum=0.9, weight_decay=config.get("WEIGHT_DECAY", 1e-8), nesterov=True)
        self.criterion = nn.MSELoss()
        
    def select_action(self, state: np.ndarray, valid_actions, epsilon: float):
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_values = self.policy_net(state)
                valid_action_values = action_values[0][valid_actions]
                return valid_actions[valid_action_values.argmax().item()]
            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).long().unsqueeze(1).to(self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().unsqueeze(1).to(self.device)
        next_state_batch = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        done_batch = torch.from_numpy(np.array(batch.done)).float().unsqueeze(1).to(self.device)
        
        self.optimizer.zero_grad()
        
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
        loss = self.criterion(q_values, expected_q_values)
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def store_transition(self, *args):
        self.memory.push(*args)
        
    def train(self):
        self.policy_net.train()
        
    def eval(self):
        self.policy_net.eval()