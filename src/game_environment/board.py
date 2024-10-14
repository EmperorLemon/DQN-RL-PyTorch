from utils.globals import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

import numpy as np
import random

BOARD_COLORS = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}


class Board:
    def __init__(self, size=4):
        self.size = size
        self._board = None

        self.reset()

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    def get_board(self) -> np.ndarray:
        return self._board

    def set_board(self, board: np.ndarray):
        self._board = board

    def reset(self):
        self._board = np.zeros((self.size, self.size), dtype=int)

        self.spawn_tile()
        self.spawn_tile()

    # Add new tile to the board.
    def spawn_tile(self):
        # Find all empty cells in board
        empty_cells = np.argwhere(self._board == 0)

        # If there are empty cells on the board
        if empty_cells.size > 0:
            i, j = random.choice(empty_cells)
            self._board[i, j] = 2 if random.random() < 0.9 else 4

    def __str__(self):
        return "\n".join(
            [" ".join([f"{tile:4}" for tile in row]) for row in self._board]
        )


class BoardLogic:
    @staticmethod
    def move(board: np.ndarray, action: int) -> tuple[np.ndarray, bool, float]:
        original_board = board.copy()

        reward: float = 0.0

        def merge(row):
            nonlocal reward

            # Remove zeros and get non-zero values
            row = row[row != 0]

            # Merge adjacent equal values
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    row[i] *= 2
                    reward += np.log2(row[i])
                    row[i + 1] = 0

            # Remove zeros again and pad with zeros
            row = row[row != 0]
            return np.pad(row, (0, 4 - len(row)), "constant")

        if action == ACTION_UP:
            board = np.apply_along_axis(merge, 0, board)
        elif action == ACTION_DOWN:
            board = np.apply_along_axis(lambda x: merge(x[::-1])[::-1], 0, board)
        elif action == ACTION_LEFT:
            board = np.apply_along_axis(merge, 1, board)
        elif action == ACTION_RIGHT:
            board = np.apply_along_axis(lambda x: merge(x[::-1])[::-1], 1, board)
        else:
            raise ValueError(f"Invalid action: {action}")

        return board, not np.array_equal(original_board, board), reward

    @staticmethod
    def game_over(board: np.ndarray) -> bool:
        # Check for empty spaces
        if np.any(board == 0):
            return False

        # Check for possible horizontal merges
        if np.any(board[:, :-1] == board[:, 1:]):
            return False

        # Check for possible vertical merges
        if np.any(board[:-1, :] == board[1:, :]):
            return False

        return True
