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
    def get_state(self):
        pass


class GameEnv(IEnv):
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = Board(size=board_size)
        
        self.score = 0

    def reset(self):
        self.board.reset()

        self.score = 0

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
        if action not in self.get_valid_actions():
            return self.get_state(), 0, self.done
        
        prev_max_tile = np.max(self._board())
        
        new_board, valid_move, move_score = BoardLogic.move(self._board(), action)

        if valid_move:
            self.board.set_board(new_board)
            self.board.spawn_tile()
            
            self.score += move_score
            
            max_tile = np.max(self._board())
            
            move_reward = np.log2(move_score + 1)
            new_tile_reward = (2 ** (np.log2(max_tile) - np.log2(prev_max_tile)) 
                               if max_tile > prev_max_tile else 0)
            
            empty_spaces_reward = 0.1 * np.sum(self._board() == 0)
            
            reward = move_reward + new_tile_reward + empty_spaces_reward + 0.1
        else:
            # Penalize invalid moves
            reward = -2
        
        done = BoardLogic.game_over(self._board())
        
        return self.get_state(), reward, done

    def get_state(self) -> np.ndarray:
        return self._board().flatten()

    def get_score(self) -> int:
        return self.score

    def get_max_tile(self) -> int:
        return np.max(self._board())