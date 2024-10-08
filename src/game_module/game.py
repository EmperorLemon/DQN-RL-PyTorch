from abc import ABC, abstractmethod
from .board import Board

from game_app_module.input import K_UP, K_DOWN, K_LEFT, K_RIGHT


class IGame(ABC):
    @abstractmethod
    def update(self, dt: float):
        pass

    @abstractmethod
    def render(self, renderer):
        pass

    @abstractmethod
    def handle_input(self, input):
        pass


class Game(IGame):
    def __init__(self):
        self.board = Board()

    def update(self, dt: float):
        # TODO: Update game logic here
        pass

    def render(self, renderer):
        renderer.clear_color((252, 251, 244))

        renderer.draw_grid(
            self.board.grid, self.board.colors, len(self.board.grid), padding=5
        )

    def handle_input(self, input):
        if input.get_key_down(K_UP):
            pass
        elif input.get_key_down(K_DOWN):
            pass
        elif input.get_key_down(K_LEFT):
            pass
        elif input.get_key_down(K_RIGHT):
            pass
