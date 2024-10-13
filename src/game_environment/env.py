from .board import Board, BoardLogic

import numpy as np


class GameEnv:
    def __init__(self, size: int = 4, seed: int = 42, move_penalty: float = 0.1):
        self.state_size = size * size
        self.action_size = 4

        np.random.seed(seed)

        self.move_penalty = move_penalty

        self.game_board = Board(size=size)

    def reset(self, step_penalty: float = 0.0):
        self.score = np.sum(self.game_board.get_board())
        self.reward = 0

        self.current_move_penalty = 0
        self.done = False

        self.rewards_list = []
        self.scores_list = []

        self.steps = 0
        self.step_penalty = step_penalty

        self.memory = []

    def step(self, action):
        board, moved = BoardLogic.move(self.game_board.get_board(), action)

        if not moved:
            return (self.game_board.get_board(), 0, self.done)

        self.reward = self.reward - self.step_penalty
        self.score = np.sum(self.game_board.get_board())
        self.done = BoardLogic.game_over(board)

        self.game_board.set_board(board)
        self.game_board.spawn_tile()

        return (board, self.reward, self.done)
