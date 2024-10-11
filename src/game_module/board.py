import random


class Board:
    def __init__(self, size=4):
        self.size = size
        self._score = 0
        # Create 4x4 grid of zeros
        self._grid = [[0 for _ in range(size)] for _ in range(size)]
        self._colors = {
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

    def get_score(self) -> int:
        return self._score

    def add_score(self, score: int):
        self._score += score

    def get_grid(self) -> list[list[int]]:
        return self._grid

    def set_grid(self, grid: list[list[int]]):
        self._grid = grid

    def get_colors(self) -> dict[int, tuple[int, int, int]]:
        return self._colors

    # Add new tile to the board.
    def spawn_tile(self):
        # Find all empty cells in board
        empty_cells = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self._grid[i][j] == 0
        ]

        if empty_cells:
            i, j = random.choice(empty_cells)
            self._grid[i][j] = 2 if random.random() < 0.9 else 4

    def __str__(self):
        return "\n".join(
            [" ".join([f"{tile:4}" for tile in row]) for row in self._grid]
        )
