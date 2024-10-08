class Board:
    def __init__(self, size=4):
        self.size = size
        # Create 4x4 grid of zeros
        self.grid = [[0 for _ in range(size)] for _ in range(size)]

    def __str__(self):
        return "\n".join([" ".join([f"{tile:4}" for tile in row]) for row in self.grid])
