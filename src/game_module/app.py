from .time import Time
from .input import Input
from .render import Renderer

from .game import IGame

from pygame import init
from pygame import event, KEYDOWN, QUIT
from pygame.display import set_caption


class App:
    def __init__(self, screen_width: int, screen_height: int, game: IGame):
        # Initialize pygame
        init()

        self.input = Input()
        self.time = Time(fps_limit=165)
        self.renderer = Renderer(screen_width, screen_height)

        set_caption("2048 Game")

        # Inject game into app (dependency injection)
        self.game = game

        self._running = True

    def run(self):
        while self._running:
            self.time.update()
            self._poll_events()
            self.game.update(self.time.dt)
            self.game.render(self.renderer)
            self.renderer.render()

    def _poll_events(self):
        for e in event.get():
            if e.type == QUIT:
                self.exit_app()

            if e.type == KEYDOWN:
                self.game.handle_input(self.input)

    def exit_app(self):
        self._running = False