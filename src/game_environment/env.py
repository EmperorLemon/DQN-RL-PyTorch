from abc import ABC, abstractmethod
from .board import Board, BoardLogic

from ai_module.agent import DQNAgent

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
    def __init__(self, size: int):
        self.state_size = size * size
        self.action_size = 4
        self.max_steps = 1000  # Maximum steps per episode

        self.game_board = Board(size=size)
        self.steps = 0
        self.score = 0
        self.done = False

    def reset(self):
        self.game_board.reset()

        self.steps = 0
        self.score = 0
        self.done = False

        return self.get_state()

    def step(self, action):
        self.steps += 1

        board, valid_move, merge_score = BoardLogic.move(
            self.game_board.get_board(), action
        )

        if valid_move:
            self.game_board.set_board(board)
            self.game_board.spawn_tile()
            self.score += merge_score
            self.reward = merge_score
        else:
            # Penalize non-merging moves
            self.reward = -1

        self.done = (
            BoardLogic.game_over(self.game_board.get_board())
            or self.steps >= self.max_steps
        )

        return (self.get_state(), self.reward, self.done)

    # Get flattened state
    def get_state(self) -> np.ndarray:
        state = self.game_board.get_board().flatten()
        return np.log2(np.where(state > 0, state, 1)).astype(np.float32)

    def get_score(self):
        return self.score

    def get_max_tile(self):
        return np.max(self.game_board.get_board())
