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
    def __init__(self, env, ai_agent=None):
        self.env = env
        self.ai_agent = ai_agent
        self.ai_mode = ai_agent is not None

        self.ai_step_delay = 0.05  # Delay between AI actions in seconds
        self.time_since_last_action = 0

        self.env.reset()

        self.episode = 0
        self.max_episodes = 10000
        self.episode_reward = 0
        self.best_score = 0

    def update(self, dt: float):
        if self.ai_mode and not self.env.done:
            self.time_since_last_action += dt

            if self.time_since_last_action >= self.ai_step_delay:
                self.perform_ai_action()
                self.time_since_last_action = 0

        if self.env.done:
            self.end_episode()

    def render(self, renderer):
        renderer.clear_color((252, 251, 244))

        if not self.env.done:
            renderer.draw_board(self.env.game_board, BOARD_COLORS, padding=5)
            renderer.draw_text(f"Score: {self.env.score}", center=(400, 75))
            if self.ai_mode:
                renderer.draw_text("[AI Mode]", center=(400, 550))
                renderer.draw_text(f"Steps: {self.env.steps}", center=(200, 580))
                renderer.draw_text(f"Episode: {self.episode}", center=(400, 580))
                renderer.draw_text(f"Best Score: {self.best_score}", center=(600, 580))
        else:
            renderer.draw_text(f"Game Over", center=(400, 75))
            renderer.draw_text(f"Score: {self.env.score}", center=(400, 110))
            renderer.draw_text(f"Best Score: {self.best_score}", center=(400, 145))

    def handle_input(self, input):
        if not self.ai_mode:
            if input.get_key_down(K_UP):
                self.env.step(ACTION_UP)
            elif input.get_key_down(K_DOWN):
                self.env.step(ACTION_DOWN)
            elif input.get_key_down(K_LEFT):
                self.env.step(ACTION_LEFT)
            elif input.get_key_down(K_RIGHT):
                self.env.step(ACTION_RIGHT)

    def is_done(self) -> bool:
        return self.episode >= self.max_episodes if self.ai_mode else self.env.done

    def perform_ai_action(self):
        if self.ai_agent:
            state = self.env.get_state()
            action = self.ai_agent.select_action(state)
            (next_state, reward, done) = self.env.step(action)
            self.ai_agent.store_transition(state, action, next_state, reward, done)
            self.ai_agent.learn()

            self.episode_reward += reward

    def end_episode(self):
        if self.ai_mode:
            self.best_score = max(self.best_score, self.env.score)
            print(
                f"Episode {self.episode} ended. Score: {self.env.score}, Best: {self.best_score}"
            )

            self.episode += 1
            self.episode_reward = 0
            self.env.reset()

            # Update target network periodically
            if self.episode % 10 == 0:
                self.ai_agent.update_target_network()
