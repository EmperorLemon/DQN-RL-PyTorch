from tensorboardX import SummaryWriter

from typing import Any
from tqdm import tqdm

from utils.globals import CHECKPOINTS_DIR
from utils.utils import join_path

from .agent import DQNAgent
from .utils import save_checkpoint

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
        
        # Training hyperparameters
        self.train_episodes = config.get("TRAIN_EPISODES", 1000)
        self.eval_episodes = config.get("EVAL_EPISODES", 20)
        self.update_frequency = config.get("UPDATE_FREQ", 5)
        self.eval_frequency = config.get("EVAL_FREQ", 50)
        
        # Epsilon hyperparameters
        self.epsilon_start = config.get("EPS_START", 0.9)
        self.epsilon_end = config.get("EPS_END", 0.05)
        self.epsilon_decay = config.get("EPS_DECAY", 0.99)
        self.epsilon = self.epsilon_start
        
        # Track best model
        self.best_score = float("-inf")
        

    def train(self):
        progress_bar = tqdm(range(self.train_episodes), desc="Training", leave=False)
        
        for episode in progress_bar:
            
            train_metrics = {
                "loss": 0,
                "steps": 0,
                "reward": 0,
                "score": 0,
                "max_tile": 0,
            }
            
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.act(state, self.epsilon, self.env.get_valid_actions(), eval=False)
                next_state, reward, done = self.env.step(action)
                
                self.agent.memory.push(state, action, next_state, reward, done)
                state = next_state
                
                # Perform one step of the optimization (on the policy network)
                train_metrics["loss"] += self.agent.optimize_model()
                train_metrics["steps"] += 1
                train_metrics["reward"] += reward
                    
            # Calculate averages and get final metrics
            train_metrics["loss"] /= max(train_metrics["steps"], 1)
            train_metrics["score"] = self.env.get_score()
            train_metrics["max_tile"] = self.env.get_max_tile()
            
            # Log training metrics
            self._log_metrics(episode, train_metrics, train=True)

            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % self.update_frequency == 0:
                self.agent.update_target_network()
            
            if episode % self.eval_frequency == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(episode, eval_metrics, train=False)
                
                if eval_metrics["avg_score"] > self.best_score:
                    self.best_score = eval_metrics["avg_score"]
                    self.save(join_path(CHECKPOINTS_DIR, f"checkpoint_{episode}.pth"))
                    
                # Update progress bar with both train and eval metrics
                progress_bar.set_postfix({
                    "train_reward": f"{train_metrics['reward']:.2f}",
                    "eval_reward": f"{eval_metrics['avg_reward']:.2f}",
                    "max_tile": eval_metrics['avg_max_tile']
                })
            else:
                # Regular training progress
                progress_bar.set_postfix({"reward": f"{train_metrics['reward']:.2f}", "loss": f"{train_metrics['loss']:.2f}"})
                
            # Not using LR scheduler since it was having issues
            # self.scheduler.step(total_loss)
            
    def _log_metrics(self, episode: int, metrics: dict, train: bool) -> None:
        """Helper method to log metrics"""
        for name, value in metrics.items():
            if train:
                self.writer.add_scalar(f"Train/{name}", value, episode)
            else:
                self.writer.add_scalar(f"Eval/{name}", value, episode)
        
        if train:
            self.writer.add_scalar("Train/Epsilon", self.epsilon, episode)
    
    def evaluate(self):
        """
        Evaluate the current policy without exploration
        Returns the evaluation metrics dictionary
        """
        
        eval_metrics: dict = {
            "rewards": [],
            "max_tiles": [],
            "scores": [],
            "steps": [],
            "success_512": 0,
            "success_1024": 0,
            "success_2048": 0
        }
        
        progress_bar = tqdm(range(self.eval_episodes), desc="Evaluating", leave=False)
        
        for _ in progress_bar:
            state = self.env.reset()
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # No exploration during evaluation
                action = self.agent.act(state, epsilon=0, valid_actions=self.env.get_valid_actions())
                    
                next_state, reward, done = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
            # Collect metrics
            max_tile = self.env.get_max_tile()
            score = self.env.get_score()
            
            eval_metrics["rewards"].append(episode_reward)
            eval_metrics["max_tiles"].append(max_tile)
            eval_metrics["scores"].append(score)
            eval_metrics["steps"].append(steps)
            
            # Track success rates
            if max_tile >= 512: eval_metrics["success_512"] += 1
            if max_tile >= 1024: eval_metrics["success_1024"] += 1
            if max_tile >= 2048: eval_metrics["success_2048"] += 1
            
            # Calculate averages
            metrics = {
                "avg_reward": np.mean(eval_metrics["rewards"]),
                "avg_max_tile": np.mean(eval_metrics["max_tiles"]),
                "avg_score": np.mean(eval_metrics["scores"]),
                "avg_steps": np.mean(eval_metrics["steps"]),
                "success_rate_512": float(eval_metrics["success_512"] / self.eval_episodes) * 100.0,
                "success_rate_1024": float(eval_metrics["success_1024"] / self.eval_episodes) * 100.0,
                "success_rate_2048": float(eval_metrics["success_2048"] / self.eval_episodes) * 100.0
            }
            
            return metrics
        
    def save(self, path: str):
        """Save model state dict and training info"""
        checkpoint = {
            "policy_state_dict": self.agent.policy_net.state_dict(),
            "target_state_dict": self.agent.target_net.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "best_score": self.best_score
        }
        
        save_checkpoint(checkpoint, path)