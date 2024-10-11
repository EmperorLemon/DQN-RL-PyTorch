from enum import Enum

from pygame import K_UP, K_DOWN, K_LEFT, K_RIGHT
from pygame.key import get_pressed

INPUT_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class Input:
    def get_key_down(self, key: int) -> bool:
        return get_pressed()[key]
