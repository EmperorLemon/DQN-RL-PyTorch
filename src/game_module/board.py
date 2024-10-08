class Board:
    def __init__(self, size=4):
        self.size = size
        # Create 4x4 grid of zeros
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.colors = {
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

    def __str__(self):
        return "\n".join([" ".join([f"{tile:4}" for tile in row]) for row in self.grid])
