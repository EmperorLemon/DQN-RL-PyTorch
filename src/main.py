from game_app_module.app import App
from game_module.game import Game


def main() -> int:
    game = Game()

    app = App(800, 600, game)
    app.run()

    return 0


if __name__ == "__main__":
    main()
