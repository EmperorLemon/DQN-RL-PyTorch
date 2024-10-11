from abc import ABC, abstractmethod
from .board import Board
from .logic import GameLogic

from game_app_module.input import K_UP, K_DOWN, K_LEFT, K_RIGHT
from game_app_module.input import INPUT_DIRECTIONS

import random


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

        self.board.spawn_tile()
        self.board.spawn_tile()

        self.game_over = False

    def update(self, dt: float):
        pass
        # if not GameLogic.game_over(self.board):
        #     if GameLogic.move(self.board, random.choice(INPUT_DIRECTIONS)):
        #         self.board.spawn_tile()
        # else:
        #     self.game_over = True

    def render(self, renderer):
        renderer.clear_color((252, 251, 244))

        renderer.draw_grid(
            self.board.get_grid(),
            self.board.get_colors(),
            self.board.size,
            padding=5,
            score=self.board.get_score(),
        )

    # If you want to play, you can :)
    def handle_input(self, input):
        if input.get_key_down(K_LEFT):
            if GameLogic.move(self.board, "LEFT"):
                self.board.spawn_tile()
        elif input.get_key_down(K_RIGHT):
            if GameLogic.move(self.board, "RIGHT"):
                self.board.spawn_tile()
        elif input.get_key_down(K_UP):
            if GameLogic.move(self.board, "UP"):
                self.board.spawn_tile()
        elif input.get_key_down(K_DOWN):
            if GameLogic.move(self.board, "DOWN"):
                self.board.spawn_tile()
