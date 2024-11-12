from pygame.time import Clock

class Time:
    def __init__(self, fps_limit: int = 60):
        self._clock = Clock()

        # Deltatime
        self.dt: float = 0.0
        self.fps_limit: int = fps_limit

    def update(self):
        self.dt = float(self._clock.tick(self.fps_limit)) / 1000.0