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
        prev_board = board.copy()
        new_board = board.copy()
        
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
            new_board = np.apply_along_axis(merge, 0, new_board)
        elif action == ACTION_DOWN:
            new_board = np.apply_along_axis(lambda x: merge(x[::-1])[::-1], 0, new_board)
        elif action == ACTION_LEFT:
            new_board = np.apply_along_axis(merge, 1, new_board)
        elif action == ACTION_RIGHT:
            new_board = np.apply_along_axis(lambda x: merge(x[::-1])[::-1], 1, new_board)
        else:
            raise ValueError(f"Invalid action: {action}")

        is_valid_move: bool = not np.array_equal(prev_board, new_board)

        return new_board, is_valid_move, score
    
    @staticmethod
    def count_similar_tiles(board: np.ndarray) -> int:
        count = 0
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if i < board.shape[0] - 1 and board[i, j] == board[i+1, j] and board[i, j] != 0:
                    count += 1
                if j < board.shape[1] - 1 and board[i, j] == board[i, j+1] and board[i, j] != 0:
                    count += 1
        return count
    
    @staticmethod
    def sum_different_tiles(board: np.ndarray) -> int:
        total = 0
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if i < board.shape[0] - 1:
                    total += abs(board[i, j] - board[i+1, j])
                if j < board.shape[1] - 1:
                    total += abs(board[i, j] - board[i, j+1])
                    
        return total

    @staticmethod
    def game_over(board: np.ndarray) -> bool:
        # Check for empty spaces
        if np.any(board == 0):
            return False

        # Check for possible vertical or horizontal merges
        if np.any(board[:, :-1] == board[:, 1:]) or np.any(board[:-1, :] == board[1:, :]):
            return False

        return True
