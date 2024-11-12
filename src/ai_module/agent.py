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
        size: int,
        config: dict[str, Any],  # Hyperparameters
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.state_size = size * size
        self.action_size = 4
        
        self.device = device
        
        # Initialize replay memory and hyperparameters
        self.memory = ReplayMemory(config.get("BUFFER_SIZE", 100_000))
        self.batch_size = config.get("BATCH_SIZE", 128)        
        self.gamma = config.get("GAMMA", 0.99) # Discount factor
        self.lr = config.get("LR", 1e-3)
        
        # Initialize policy and target networks
        self.policy_net = DQN(self.state_size, self.action_size, config.get("HIDDEN_LAYERS", [128, 128])).to(device)
        self.target_net = DQN(self.state_size, self.action_size, config.get("HIDDEN_LAYERS", [128, 128])).to(device)
        
        # Copy policy network weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss() # Huber loss
        
    def act(self, state: np.ndarray, epsilon: float, valid_actions: list[int], eval: bool = True):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current game state as numpy array
            epsilon: Exploration rate (0-1)
            valid_actions: List of valid actions at current state
            eval: If True, keeps network in eval mode (for evaluation/testing)
        """
        
        # Exploration: randomly sample from valid actions with probability epsilon
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: Use policy network to select best action
        
        # Convert state to tensor and add batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Disable dropout for prediction
        self.policy_net.eval()
        
        # Get Q-values without computing gradients
        with torch.no_grad():
            q_values = self.policy_net(state)
         
        # Mask invalid actions by only considering Q-values of valid actions
        valid_q_values = q_values[0][valid_actions]
        
        # Select highest Q-value action
        best_action_idx = valid_q_values.argmax().item()
        
        # Map back to original action space
        best_action = valid_actions[best_action_idx]
        
        # Return to training mode if not in evaluation
        if not eval:
            self.policy_net.train()
    
        return best_action
            
    def optimize_model(self) -> float:
        """
        Perform one step of optimization on the DQN
        Returns the loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample and prepare batch from memory buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for 
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        action_batch = torch.tensor(batch.action).long().to(self.device)
        reward_batch = torch.tensor(batch.reward).float().to(self.device)
        next_state_batch = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        done_batch = torch.tensor(batch.done).float().to(self.device)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Current Q-values
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next state Q-values (following Bellman equation)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Expected Q-values: R + γ * max(Q(s', a')) * (1 - done)
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
            
        # Compute loss and optimize
        loss = self.criterion(q_values, expected_q_values.unsqueeze(1))
        
        loss.backward()        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())