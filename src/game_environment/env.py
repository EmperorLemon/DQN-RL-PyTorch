from abc import ABC, abstractmethod
from .board import Board, BoardLogic

from utils.globals import ACTION_SET

import numpy as np


class IEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

class GameEnv(IEnv):
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = Board(size=board_size)
        
        self.score = 0
        self.done = False

    def reset(self):
        self.board.reset()

        self.score = 0
        self.done = False

        return self.get_state()
    
    def _board(self):
        return self.board.get_board()
    
    def get_valid_actions(self):
        valid_actions = []
        
        for action in ACTION_SET:
            _, valid_move, _ = BoardLogic.move(self._board(), action)
            
            if valid_move:
                valid_actions.append(action)
                
        return valid_actions

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """Execute one step in the environment.
        
        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            tuple: (next_state, reward, done)
                - next_state: The new board state after the action
                - reward: The reward received for the action
                - done: Whether the game is over
        """
        
        prev_max_tile = np.max(self._board())
        
        new_board, valid_move, move_score = BoardLogic.move(self._board(), action)

        if valid_move:
            self.board.set_board(new_board)
            self.board.spawn_tile()
            
            self.score += move_score
            
            max_tile = np.max(self._board())
            
            # Merge reward, use log2 to prevent reward explosion
            move_reward = np.log2(move_score) if move_score > 0 else 0
            
            # Reward for achieving new highest tile
            new_tile_reward = (np.log2(max_tile) - np.log2(prev_max_tile)
                               if max_tile > prev_max_tile else 0)
            
            # Add a small reward for empty spaces
            empty_spaces_reward = 0.1 * np.sum(self._board() == 0)
            
            # Reward is the sum of all previous rewards plus a small bonus for each step
            reward = move_reward + new_tile_reward + empty_spaces_reward + 0.1
        else:
            # Penalize invalid moves
            reward = -2
        
        self.done = BoardLogic.game_over(self._board())
        
        return self.get_state(), reward, self.done

    def get_state(self) -> np.ndarray:
        state = self._board().flatten()
        return np.log2(np.where(state > 0, state, 1)).astype(float)

    def get_score(self) -> int:
        return self.score

    def get_max_tile(self) -> int:
        return np.max(self._board())
    
    def render(self) -> None:
        print(self.board)