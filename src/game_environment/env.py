from abc import ABC, abstractmethod
from .board import Board, BoardLogic

import numpy as np


class IEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass


class GameEnv(IEnv):
    def __init__(self, size: int = 4):
        self.state_size = size * size
        self.action_size = 4

        self.game_board = Board(size=size)

    def reset(self):
        self.game_board.reset()

        self.score = np.sum(self.game_board.get_board())
        self.reward = 0

        self.done = False

    def step(self, action):
        board, valid_move, reward = BoardLogic.move(self.game_board.get_board(), action)

        if not valid_move:
            pass

        self.reward += reward

        self.game_board.set_board(board)
        self.game_board.spawn_tile()

        self.score = np.sum(self.game_board.get_board())
        self.done = BoardLogic.game_over(self.game_board.get_board())

        return (board, self.reward, self.done)
