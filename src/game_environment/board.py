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
        self.board = np.zeros((self.size, self.size), dtype=int)

    def __getitem__(self, key):
        return self.board[key]

    def __setitem__(self, key, value):
        self.board[key] = value

    def get_board(self) -> np.ndarray:
        return self.board

    def set_board(self, board: np.ndarray):
        self.board = board

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

        self.spawn_tile()
        self.spawn_tile()

    # Add new tile to the board.
    def spawn_tile(self):
        # Find all empty cells in board
        empty_cells = np.argwhere(self.board == 0)

        # If there are empty cells on the board
        if empty_cells.size > 0:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4

    def __str__(self):
        return "\n".join(
            [" ".join([f"{tile:4}" for tile in row]) for row in self.board]
        )


class BoardLogic:
    @staticmethod
    def move(board: np.ndarray, action: int) -> tuple[np.ndarray, bool, int]:
        """
        Perform a move on the board.

        Args:
            board (np.ndarray): The current game board.
            action (int): The action to perform (UP, DOWN, LEFT, RIGHT).

        Returns:
            Tuple[np.ndarray, bool, int]: New board state, whether move was valid, and score.
        """

        original_board = board.copy()
        score: int = 0

        def merge(row):
            nonlocal score

            # Remove zeros and get non-zero values
            row = row[row != 0]

            # Merge adjacent equal values
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    row[i] *= 2
                    score += row[i]
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

        is_valid_move: bool = not np.array_equal(original_board, board)

        return board, is_valid_move, score

    @staticmethod
    def game_over(board: np.ndarray) -> bool:
        # Check for empty spaces
        if np.any(board == 0):
            return False

        # Check for possible vertical or horizontal merges
        if np.any(board[:, :-1] == board[:, 1:]) or np.any(
            board[:-1, :] == board[1:, :]
        ):
            return False

        return True
