from tensorboardX import SummaryWriter

from utils.globals import ACTION_SET

from torch import optim

from typing import Any
from collections import defaultdict

from tqdm import tqdm

from .agent import DQNAgent
from game_environment.env import GameEnv

import numpy as np


class Trainer:
    def __init__(self, env: GameEnv, agent: DQNAgent, 
                 writer: SummaryWriter | None, 
                 config: dict[str, Any]):
        self.env = env
        self.agent = agent
        self.writer = writer
        
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.agent.optimizer)
        
        self.done = False
        
        self.max_episodes = config.get("MAX_EPISODES", 1000)
        
        self.update_frequency = config.get("UPDATE_FREQ", 25)
        
        self.epsilon_start = config.get("EPS_START", 1.0)
        self.epsilon_end = config.get("EPS_END", 0.05)
        self.epsilon_decay = config.get("EPS_DECAY", 0.99)
        
        self.epsilon = self.epsilon_start
        self.global_step = 0

    def train(self):
        tile_frequencies = defaultdict(int)
        
        progress_bar = tqdm(range(self.max_episodes), desc="Training")
        
        for episode in progress_bar:
            state = self.env.reset()
            
            action_frequencies = defaultdict(int)
            
            total_loss = 0
            episode_steps = 0
            episode_reward = 0
            done = False
            
            self.agent.train()
            
            while not done:
                action = self.agent.act(state, self.epsilon, self.env.get_valid_actions())
                next_state, reward, done = self.env.step(action)
                
                self.agent.store_transition(state, action, next_state, reward, done)
                
                total_loss += self.agent.optimize_model()
                
                if self.global_step % self.update_frequency == 0:
                    self.agent.update_target_network()
                    
                state = next_state
        
                episode_steps += 1
                episode_reward += reward
                
                action_frequencies[action] += 1
                
                self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"reward": f"{episode_reward:.2f}"})
            
            score = self.env.get_score()
            max_tile = self.env.get_max_tile()
                
            tile_frequencies[max_tile] += 1
                
            # Log metrics to TensorBoard
            self.writer.add_scalar("Reward/Train", episode_reward, episode)
            self.writer.add_scalar("Score/Train", score, episode)
            self.writer.add_scalar("Score/Tile/Train", max_tile, episode)
            self.writer.add_scalar("Steps/Train", episode_steps, episode)
            self.writer.add_scalar("Epsilon", self.epsilon, episode)
            
            # Log tile frequencies
            for tile in [512, 1024, 2048]:
                self.writer.add_scalar(f"Tile/Frequency/{tile}", tile_frequencies[tile], episode)

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            # self.scheduler.step(total_loss)