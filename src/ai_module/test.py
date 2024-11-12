from .agent import DQNAgent
from .utils import load_best_checkpoint
from typing import Any

from game_environment.env import GameEnv

from utils.globals import CHECKPOINTS_DIR

def prepare_agent(agent: DQNAgent):
    # print(f"Now Playing {num_games} Games of 2048")
    checkpoint = load_best_checkpoint(CHECKPOINTS_DIR)
    
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_net.eval()

def play_2048(env: GameEnv, agent: DQNAgent, state: Any):
    # Select action with no exploration
    action = agent.act(state, epsilon=0, valid_actions=env.get_valid_actions())
    
    next_state, _, _ = env.step(action)
    # total_reward += reward
    # for game in range(num_games):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
        
    #     while not done:

        
            
    #     print(f"\nGame {game + 1} Summary:")
    #     print(f"Final Score: {env.get_score()}")
    #     print(f"Max Tile: {env.get_max_tile()}")
    #     print(f"Total Reward: {total_reward:.2f}")