from abc import ABC, abstractmethod

from game_app_module.input import K_UP, K_DOWN, K_LEFT, K_RIGHT
from utils.globals import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

from .board import BOARD_COLORS

INPUT_MAPPING = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}


# The 'I' being 'Interactable' ;)
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

    @abstractmethod
    def is_done(self):
        pass


class Game(IGame):
    def __init__(self, env):
        self.env = env

    def update(self, dt: float):
        pass

    def render(self, renderer):
        renderer.clear_color((252, 251, 244))

        # if not self.env.done:
        #     renderer.draw_board(self.env.game_board, BOARD_COLORS, padding=5)
        #     renderer.draw_text(f"Score: {self.env.score}", center=(400, 75))
        # else:
        #     renderer.draw_text(f"Game Over", center=(400, 75))
        #     renderer.draw_text(f"Score: {self.env.score}", center=(400, 110))

    def handle_input(self, input):
        if input.get_key_down(K_UP):
            self.env.step(ACTION_UP)
        elif input.get_key_down(K_DOWN):
            self.env.step(ACTION_DOWN)
        elif input.get_key_down(K_LEFT):
            self.env.step(ACTION_LEFT)
        elif input.get_key_down(K_RIGHT):
            self.env.step(ACTION_RIGHT)

    def is_done(self) -> bool:
        return False
