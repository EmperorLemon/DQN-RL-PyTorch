from pygame.display import set_mode, flip
from pygame.draw import rect
from pygame.font import Font


class Renderer:
    def __init__(self, width: int, height: int):
        self._screen = set_mode((width, height))
        self._font = Font(None, 36)

    def clear_color(self, color):
        self._screen.fill(color=color)

    def get_width(self) -> int:
        return self._screen.get_width()

    def get_height(self) -> int:
        return self._screen.get_height()

    def draw_grid(
        self,
        grid: list[list[int]],
        colors: dict[int, tuple[int, int, int]],
        size: int,
        padding: int,
    ):
        cell_size = min(self.get_width(), self.get_height()) // (size + 1)

        # Calculate top-left position of grid to center it
        width = size * cell_size + (size - 1) * padding
        height = width  # Using square grid

        start_x = (self.get_width() - width) // 2
        start_y = (self.get_height() - height) // 2

        # Draw background rectangle for the grid
        rect(
            self._screen,
            (187, 173, 160),
            (
                start_x - padding,
                start_y - padding,
                width + 2 * padding,
                height + 2 * padding,
            ),
        )

        for row in range(size):
            for col in range(size):
                value = grid[row][col]

                # Calculate position for this cell
                x = start_x + col * (cell_size + padding)
                y = start_y + row * (cell_size + padding)

                rect(
                    self._screen,
                    colors.get(value, (0, 0, 0)),
                    (x, y, cell_size, cell_size),
                    border_radius=8,
                )

                if value != 0:
                    text_surface = self._font.render(str(value), True, (0, 0, 0))
                    text_rect = text_surface.get_rect(
                        center=(x + cell_size // 2, y + cell_size // 2)
                    )
                    self._screen.blit(text_surface, text_rect)

    def render(self):
        flip()
