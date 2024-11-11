from ai_module.utils import load_checkpoint
from ai_module.agent import DQNAgent
from game_environment.env import GameEnv

from utils.globals import ACTION_NAMES, CHECKPOINTS_DIR
from utils.utils import join_path, list_files

import os

def play_2048(env: GameEnv, agent: DQNAgent, num_games: int = 5):
    
    best_score = float("-inf")
    
    checkpoint = None
    
    for filepath in list_files(CHECKPOINTS_DIR):
        cp = load_checkpoint(filepath)
        cp_best_score = cp["best_score"]
        
        if cp_best_score > best_score:
            best_score = cp_best_score
            checkpoint = cp
            
            print(f"Highest Score in checkpoint: {os.path.basename(filepath)}")
            print(f"Best Score: {cp_best_score:.2f}")
    
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_net.eval()
    
    for game in range(num_games):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action with no exploration
            action = agent.act(state, epsilon=0, valid_actions=env.get_valid_actions())
            
            state, reward, done = env.step(action)
            total_reward += reward
        
            
        print(f"\nGame {game + 1} Summary:")
        env.render()
        print(f"Final Score: {env.get_score()}")
        print(f"Max Tile: {env.get_max_tile()}")
        print(f"Total Reward: {total_reward:.2f}")