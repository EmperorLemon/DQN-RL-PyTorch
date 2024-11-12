from abc import ABC, abstractmethod

from .input import K_UP, K_DOWN, K_LEFT, K_RIGHT
from utils.globals import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

from ai_module.test import prepare_agent, play_2048

from game_environment.board import BOARD_COLORS

from time import sleep

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


class Game(IGame):
    def __init__(self, env, agent, num_games: int = 5):
        self.env = env
        self.agent = agent
        
        self.move_timer = 0
        self.move_delay = 0.1 # Seconds between moves
        
        self.games_remaining = num_games
        
        # Current state of the game environment
        self.state = self.env.reset()
        
        # Load models for agent
        prepare_agent(self.agent)

    def update(self, dt: float):
        if self.games_remaining > 0:
            if self.env.done:
                self.games_remaining -= 1
                
                self.state = self.env.reset()
                
                # Give the viewer some time to look at the score and max tile
                sleep(1.25) 
            else:
                self.move_timer += dt
                
                # Make AI move after delay
                if self.move_timer >= self.move_delay:
                    self.move_timer = 0
                    
                    play_2048(self.env, self.agent, self.state)
            
                
    def render(self, renderer):
        renderer.clear_color((252, 251, 244))
        
        renderer.draw_text(f"Games Remaining: {self.games_remaining}", center=(400, 25))

        if not self.env.done:
            renderer.draw_board(self.env.board, BOARD_COLORS, padding=5)
            renderer.draw_text(f"Score: {self.env.score}", center=(400, 60))
            renderer.draw_text(f"Max Tile: {self.env.get_max_tile()}", center=(400, 90))
        else:
            renderer.draw_text(f"Game Over", center=(400, 80))
            renderer.draw_text(f"Final Score: {self.env.score}", center=(400, 115))
            renderer.draw_text(f"Max Tile: {self.env.get_max_tile()}", center=(400, 150))

    def handle_input(self, input):
        if input.get_key_down(K_UP):
            self.env.step(ACTION_UP)
        elif input.get_key_down(K_DOWN):
            self.env.step(ACTION_DOWN)
        elif input.get_key_down(K_LEFT):
            self.env.step(ACTION_LEFT)
        elif input.get_key_down(K_RIGHT):
            self.env.step(ACTION_RIGHT)