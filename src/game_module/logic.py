class GameLogic:
    @staticmethod
    def move(board, direction) -> bool:
        def reverse(row):
            return row[::-1]

        def transpose(grid):
            return [list(row) for row in zip(*grid)]

        def push_left(row):
            new_row = [tile for tile in row if tile != 0]
            new_row += [0] * (board.size - len(new_row))

            return new_row

        def merge(row):
            for i in range(board.size - 1):
                if row[i] == row[i + 1] != 0:
                    row[i] *= 2
                    board.add_score(row[i])
                    row[i + 1] = 0

            return push_left(row)

        moves = {
            "LEFT": lambda grid: [merge(push_left(row)) for row in grid],
            "RIGHT": lambda grid: [
                reverse(merge(push_left(reverse(row)))) for row in grid
            ],
            "UP": lambda grid: transpose(
                [merge(push_left(row)) for row in transpose(grid)]
            ),
            "DOWN": lambda grid: transpose(
                [reverse(merge(push_left(reverse(row)))) for row in transpose(grid)]
            ),
        }

        if direction in moves:
            grid = moves[direction](board.get_grid())

            if grid != board.get_grid():
                board.set_grid(grid)
                return True

        return False

    @staticmethod
    def game_over(board) -> bool:
        for i in range(board.size):
            for j in range(board.size):
                grid = board.get_grid()

                if grid[i][j] == 0:
                    return False
                if i < board.size - 1 and grid[i][j] == grid[i + 1][j]:
                    return False
                if j < board.size - 1 and grid[i][j] == grid[i][j + 1]:
                    return False

        return True
